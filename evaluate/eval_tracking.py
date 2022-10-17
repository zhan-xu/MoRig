import sys
sys.path.append("./")
import argparse, os, cv2, glob, copy, numpy as np, open3d as o3d, matplotlib.pyplot as plt, time
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch
from torch_geometric.utils import add_self_loops
import models
from datasets.dataset_pose import GraphData
from utils.rig_parser import Rig
from utils.os_utils import mkdir_p
from utils.vis_utils import visualize_seg, visualize_track, show_obj_rig
from utils.deform_ik import Deform_IK
from utils.piecewise_ransac import Piecewise_RANSAC


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_deformnet(deformnet_path):
    deformnet = models.__dict__["deformnet"](tau_nce=0.07, num_interp=5)
    deformnet.to(device)
    if torch.cuda.is_available():
        deformnet.load_state_dict(torch.load(deformnet_path)['state_dict'])
    else:
        deformnet.load_state_dict(torch.load(deformnet_path, map_location=torch.device('cpu'))['state_dict'])
    deformnet.eval()
    return deformnet


def run_deform_net_inference(flow_net, vtx_in, pts_in, tpl_e, geo_e):
    vtx = torch.from_numpy(vtx_in).float()
    pts = torch.from_numpy(pts_in).float()
    tpl_e_tensor = torch.from_numpy(tpl_e).long()
    geo_e_tensor = torch.from_numpy(geo_e).long()
    tpl_e_tensor, _ = add_self_loops(tpl_e_tensor, num_nodes=len(vtx))
    geo_e_tensor, _ = add_self_loops(geo_e_tensor, num_nodes=len(vtx))
    data = GraphData(tpl_edge_index=tpl_e_tensor, geo_edge_index=geo_e_tensor)
    data.vtx = torch.from_numpy(vtx_in).float()
    data.pts = torch.from_numpy(pts_in).float()
    data.vtx_batch = torch.zeros(len(vtx), dtype=torch.long)
    data.pts_batch = torch.zeros(len(pts), dtype=torch.long)
    data.to(device)
    with torch.no_grad():
        pred_flow, vtx_feature, pts_feature, pred_vismask, tau = flow_net(data)
    pred_flow = pred_flow.to("cpu").numpy()
    pred_vismask = pred_vismask.to("cpu").numpy()
    vtx_feature = vtx_feature.to("cpu").numpy()
    pts_feature = pts_feature.to("cpu").numpy()
    corr_matrix = np.matmul(vtx_feature, pts_feature.T)
    vert_shift = vtx_in + pred_flow
    return vert_shift, pred_vismask, corr_matrix


def ik_drag(vtx_src, vtx_dst, pts_dst, rig, corrmat, vismask):
    vert_src = np.column_stack((vtx_src, np.ones(len(vtx_src)))).T
    vert_src = torch.from_numpy(vert_src)
    vert_dst = torch.from_numpy(vtx_dst)
    rig_globals_inv = torch.inverse(torch.from_numpy(rig.global_transforms_homogeneous))
    vert_src_local = torch.matmul(rig_globals_inv, vert_src[None, ...])
    deformer = Deform_IK(vismask_thrd=0.3) # vismask_thrd=0.30
    locals_update, globals_update, jpos = \
        deformer.run(locals_in=torch.from_numpy(rig.local_frames).float(),
                     offsets=torch.from_numpy(rig.offset).float(),
                     parent=rig.hierarchy,
                     root_id=rig.root_id,
                     vert_local=vert_src_local.float(),
                     skinning=torch.from_numpy(rig.skins).float(),
                     constraints=vert_dst.float(),
                     vismask=torch.from_numpy(vismask).float(),
                     iter_time=200)
    locals_update = locals_update.detach().numpy()
    jpos = jpos.detach().numpy()
    rig_update = copy.deepcopy(rig)
    rig_update.pos = jpos
    rig_update.local_frames = locals_update
    rig_update.FK()
    vert_update = np.matmul(rig_update.global_transforms_homogeneous, vert_src_local.numpy())
    vtx_update_glb = np.sum(vert_update * rig_update.skins.T[:, None, :], axis=0)[0:3].T

    if corrmat is not None:
        ''' 2nd time: t2 mesh -> gt points IK '''
        max_sim = np.max(corrmat, axis=1)
        nnidx = np.argmax(corrmat, axis=1)
        corr_list = np.zeros((corrmat.shape[1], 3))
        for idx in range(len(nnidx)):
            if (max_sim[idx] > corr_list[nnidx[idx], -1]):
                corr_list[nnidx[idx], 0] = idx
                corr_list[nnidx[idx], 1] = nnidx[idx]
                corr_list[nnidx[idx], 2] = max_sim[idx]
        # cor dis
        thd = 0.5
        nnidx = np.where(corr_list[:, -1] > thd)[0]
        corr_list = corr_list[nnidx, 0:2]

        vert_src = np.column_stack((vtx_update_glb, np.ones(len(vtx_update_glb)))).T
        vert_src = torch.from_numpy(vert_src)
        vert_src_all = copy.deepcopy(vert_src)
        vert_dst = torch.from_numpy(pts_dst)

        vert_src = vert_src[:, corr_list[:, 0]]
        vert_dst = vert_dst[corr_list[:, 1], :]

        # l2 dis
        l2_dis = torch.sum((vert_src[0:3, :].T - vert_dst)**2, dim=-1)
        nnidx = torch.where(l2_dis < 1e-2)[0]
        #print('close correspondent pair len after corr_matrix', corr_list.shape[0], ', after L2 constrain', nnidx.shape[0])
        vert_src = vert_src[:, nnidx]
        vert_dst = vert_dst[nnidx, :]
        corr_list = corr_list[nnidx, :]

        rig_globals_inv = torch.inverse(torch.from_numpy(rig_update.global_transforms_homogeneous))
        vert_src_local = torch.matmul(rig_globals_inv, vert_src[None, ...])

        locals_update, globals_update, jpos = \
            deformer.run(locals_in=torch.from_numpy(rig_update.local_frames).float(),
                         offsets=torch.from_numpy(rig_update.offset).float(),
                         parent=rig_update.hierarchy,
                         root_id=rig_update.root_id,
                         vert_local=vert_src_local.float(),
                         skinning=torch.from_numpy(rig_update.skins).float()[corr_list[:, 0]],
                         constraints=vert_dst.float(),
                         vismask=torch.from_numpy(vismask).float()[corr_list[:, 0]],
                         iter_time=400, lr=1e-3, w_invis=0.0)

        locals_update = locals_update.detach().numpy()
        jpos = jpos.detach().numpy()
        rig_update2 = copy.deepcopy(rig_update)
        rig_update2.pos = jpos
        rig_update2.local_frames = locals_update
        rig_update2.FK()

        vert_src_all_local = torch.matmul(rig_globals_inv, vert_src_all[None, ...])
        vert_update = np.matmul(rig_update2.global_transforms_homogeneous,
                                vert_src_all_local.numpy())
        vtx_update_glb = np.sum(vert_update * rig_update2.skins.T[:, None, :], axis=0)[0:3].T

        # pcd_1 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vtx_src))
        # pcd_1.paint_uniform_color([0.0, 0.0, 1.0])
        # pcd_2 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vtx_update_glb))
        # pcd_2.paint_uniform_color([0.0, 0.0, 1.0])
        # pcd_gt = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_dst))
        # pcd_gt.paint_uniform_color([0.0, 0.0, 1.0])
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_1)
        # vis.add_geometry(pcd_2.translate([1.0, 0.0, 0.0]))
        # vis.add_geometry(pcd_gt.translate([2.0, 0.0, 0.0]))
        # vis.run()
        # vis.destroy_window()
        return vtx_update_glb, Rotation.from_matrix(rig_update2.local_frames).as_quat()
    else:
        return vtx_update_glb, Rotation.from_matrix(rig_update.local_frames).as_quat()


def tracking_one(vtx_ori, rig, pts_traj, tpl_e, geo_e, deformnet):
    seq_length = pts_traj.shape[1]
    pred_vtx_traj, pred_vismask, pred_quats = [vtx_ori], [], []
    for t in tqdm(range(1, seq_length)):
        pts_tar = pts_traj[:, t, :]
        vert_shift, pred_vismask_t, corr_matrix_t = run_deform_net_inference(deformnet, pred_vtx_traj[-1], pts_tar, tpl_e, geo_e)
        vert_shift, quats_t = ik_drag(pred_vtx_traj[0], vert_shift, pts_tar, rig, corr_matrix_t, pred_vismask_t.squeeze(axis=1))
        pred_vtx_traj.append(vert_shift)
        pred_vismask.append(pred_vismask_t.squeeze(axis=1))
        pred_quats.append(quats_t)
    pred_vtx_traj = np.stack(pred_vtx_traj[1:], axis=1)
    pred_vismask = np.stack(pred_vismask, axis=1)
    pred_quats = np.stack(pred_quats, axis=1)
    return pred_vtx_traj, pred_vismask, pred_quats
    
    
    vtx_traj_filelist = glob.glob(os.path.join(testset_folder, f"*_vtx_traj.npy"))
    for vtx_traj_filename in tqdm(vtx_traj_filelist[start_id:end_id]):
        model_id = vtx_traj_filename.split("/")[-1].split("_")[0]
        vtx_traj_filename = os.path.join(testset_folder, f"{model_id}_vtx_traj.npy")
        if os.path.exists(os.path.join(rig_folder, f"tracking_loss/{model_id}_pred_vtx.npy")):
            continue
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{model_id}.obj"))
        if not os.path.exists(os.path.join(rig_folder, f"{model_id}_rig.txt")):
            continue
        rig = Rig(os.path.join(rig_folder, f"{model_id}_rig.txt"))
        tpl_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_tpl_e.txt")).T
        geo_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_geo_e.txt")).T
        gt_vtx_traj = np.load(vtx_traj_filename)
        gt_pts_traj = np.load(vtx_traj_filename.replace("_vtx_traj.npy", "_pts_traj.npy")).reshape(-1, 101, 3)
        gt_vismask = np.load(vtx_traj_filename.replace("_vtx_traj.npy", "_vismask.npy"))

        pred_vtx_traj, pred_vismask, tar_pts, corr_matrix, quats = [], [], [], [], []
        full_flow_error, vis_flow_error = [], []
        pred_vtx_traj.append(gt_vtx_traj[:, 0, :])
        pred_vismask.append(gt_vismask[:, 0])
        tar_pts.append(gt_pts_traj[:, 0, :])
        corr_matrix.append(np.zeros((gt_vtx_traj.shape[0], gt_pts_traj.shape[0])))
        quats.append(Rotation.from_matrix(rig.local_frames).as_quat())

        for t in tqdm(range(1, seq_length + 1)):
            pts_tar = gt_pts_traj[:, t, :]
            vert_shift, pred_vismask_t, corr_matrix_t = run_deform_net_inference(flownet, pred_vtx_traj[-1], pts_tar, tpl_e, geo_e)
        
            if False:  # visualize flow shifting
                mesh_last = copy.deepcopy(mesh)
                mesh_last.vertices = o3d.utility.Vector3dVector(pred_vtx_traj[-1])
                mesh_last_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_last)
                mesh_last_ls.paint_uniform_color([0.0, 0.0, 1.0])
                mesh_shift = copy.deepcopy(mesh)
                mesh_shift.vertices = o3d.utility.Vector3dVector(vert_shift)
                mesh_shift_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_shift)
                mesh_shift_ls.paint_uniform_color([1.0, 0.0, 0.0])
                #pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_tar))
                #pcd.paint_uniform_color([0.0, 0.0, 1.0])
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(mesh_last_ls)
                vis.add_geometry(mesh_shift_ls.translate([1.0, 0.0, 0.0]))
                #vis.add_geometry(pcd.translate([2.0, 0.0, 0.0]))
                vis.run()
                vis.destroy_window()

            pred_vtx_traj.append(vert_shift)
            pred_vismask.append(pred_vismask_t.squeeze(axis=1))
            corr_matrix.append(corr_matrix_t)
            tar_pts.append(pts_tar)
            pred_vtx_traj[t], quats_t = ik_drag(pred_vtx_traj[0], pred_vtx_traj[t], pts_tar, rig, corr_matrix_t, pred_vismask[t])
            quats.append(quats_t)

            if False:  # visualize flow shifting
                mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                mesh_ls.paint_uniform_color([0.0, 0.0, 1.0])
                mesh_shift = copy.deepcopy(mesh)
                mesh_shift.vertices = o3d.utility.Vector3dVector(pred_vtx_traj[t])
                mesh_shift_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_shift)
                mesh_shift_ls.paint_uniform_color([1.0, 0.0, 0.0])
                #pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_tar))
                #pcd.paint_uniform_color([0.0, 0.0, 1.0])
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(mesh_ls)
                vis.add_geometry(mesh_shift_ls.translate([1.0, 0.0, 0.0]))
                #vis.add_geometry(pcd.translate([2.0, 0.0, 0.0]))
                vis.run()
                vis.destroy_window()


def plot(type="full"):
    none_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/none_rigs/tracking_loss_pred_flow/"
    rignet_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/rignet_rigs/tracking_loss_pred_flow/"
    ours_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/ours_rigs_no_dup/tracking_loss_pred_flow/"
    ours_gt_flow_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/ours_rigs_gt_flow/tracking_loss_gt_flow/"
    skerig_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/SkeRig_pred_flow/tracking_loss_pred_flow/"
    skerig_gt_flow_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/SkeRig_gt_flow/tracking_loss_gt_flow/"

    model_list = ['4140', '8184', '18897', '3621', '3618', '8318', '7163', '8306', '8301', '8334', '8234', '2921', '3625', '8328', '17711', '14188', '9871', '10087', '3220', '9876', '7495', '9836', '6383', '18592', '8235', '10104', '3672', '8247', '3725', '15631', '3535', '17224', '3670', '6516', '15671', '10080', '8210', '7983', '18596', '783', '14414', '14051', '3640', '16282', '9870', '9453', '8336', '17736', '16537', '7201', '15687', '9853', '8236', '8233', '3675', '14836', '3724', '781', '8227', '17161', '8193', '496', '7216', '9835', '14455', '8335', '18063', '7154', '7979', '7157', '9852', '12106', '3641', '2586', '510', '7179', '8333', '8248', '3642', '8330', '14466', '15677', '15559', '6387', '10518', '1236', '7771', '4518', '8331', '9479', '3540', '10503', '9484', '1262', '19193', '6384', '14521', '17284', '9466', '11786', '8478', '14509', '10108', '6521', '18020', '14697', '14602', '7191', '432', '12108', '2317', '10560', '15472', '5590', '10559', '8245', '13463', '1317', '3685', '8304', '10557', '1307', '16314', '15930', '14471', '3645', '14373', '16261', '18617', '12121', '10110', '16552', '15458', '1276', '14462', '1280', '16740', '14372', '425', '10514', '9477', '15012', '2923', '7665', '3644', '18608', '2132', '8320', '15906', '14726']

    errors = []
    for folder in [none_folder, ours_gt_flow_folder, ours_folder, rignet_folder, skerig_folder, skerig_gt_flow_folder]:
        error_list = glob.glob(os.path.join(folder, f"*_{type}_flow_error.npy"))
        error = []
        for error_filename in error_list:
            #model_id = error_filename.split("/")[-1].split("_")[0]
            #if model_id in model_list:
            error.append(np.load(error_filename))
        error = np.concatenate(error, axis=0)
        error = error.mean(axis=0)
        errors.append(error)

    t = np.arange(101)  # 101
    plt.plot(t, errors[0], 'm--', label="none")
    #plt.plot(t, errors[1], 'g--', label="ours_gt_flow")
    plt.plot(t, errors[2], 'y--', label="ours_pred_flow")
    plt.plot(t, errors[3], 'b--', label="rignet_pred_flow")
    plt.plot(t, errors[4], 'r--', label="SkeRig_pred_flow")
    #plt.plot(t, errors[5], 'k--', label="SkeRig_gt_flow")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    testset_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/test/"
    mesh_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/obj_remesh/"
    deformnet_seq_path = "checkpoints/deform_p_mr_seq/model_best.pth.tar"
    rig_folder = "results/our_results"
    
    deformnet = load_deformnet(deformnet_path=deformnet_seq_path)
    pts_traj_filelist = glob.glob(os.path.join(testset_folder, f"*_pts_traj.npy"))
    mkdir_p(os.path.join(rig_folder, "tracking_loss/"))
    for pts_traj_filename in tqdm(pts_traj_filelist):
        model_id = pts_traj_filename.split("/")[-1].split("_")[0]
        gt_pts_traj =  np.load(pts_traj_filename).reshape(-1, 101, 3)
        gt_vtx_traj = np.load(pts_traj_filename.replace("_pts_traj.npy", "_vtx_traj.npy"))
        gt_vismask = np.load(pts_traj_filename.replace("_pts_traj.npy", "_vismask.npy"))
        rig = Rig(os.path.join(rig_folder, f"{model_id}_rig2.txt"))
        tpl_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_tpl_e.txt")).T
        geo_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_geo_e.txt")).T
        pred_vtx_traj, pred_vismask, pred_quats = \
            tracking_one(vtx_ori=gt_vtx_traj[:, 0, :], 
                        rig=rig, pts_traj=gt_pts_traj, 
                        tpl_e=tpl_e, geo_e=geo_e, 
                        deformnet=deformnet)        
        # evaluate
        full_flow_error = np.sqrt(np.sum((pred_vtx_traj - gt_vtx_traj[:, 1:, :]) ** 2, axis=2)).mean()
        vis_flow_error = ((np.sqrt(np.sum((pred_vtx_traj - gt_vtx_traj[:, 1:, :]) ** 2, axis=2)) * (gt_vismask[:, 1:] > 0.5)).sum()) / (gt_vismask[:, 1:] > 0.5).sum()
        # save results
        np.savez(os.path.join(rig_folder, f"tracking_loss/{model_id}_tracking.npz"), 
                 pred_quats=pred_quats, pred_vtx_traj=pred_vtx_traj, pred_vismask=pred_vismask, 
                 full_flow_error=full_flow_error, vis_flow_error=vis_flow_error)
        