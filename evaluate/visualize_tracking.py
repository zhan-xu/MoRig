import numpy as np, open3d as o3d, cv2, os, copy, glob
from utils.rig_parser import Rig
from scipy.spatial.transform import Rotation


def visualize_tracking(vtx_traj_pred, vtx_traj_gt, pts_traj, mesh, output_folder=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for frame_id in range(101):
        #print(frame_id)
        #vis = o3d.visualization.Visualizer()
        #vis.create_window()
        mesh_t_gt = copy.deepcopy(mesh)
        mesh_t_gt.vertices = o3d.utility.Vector3dVector(vtx_traj_gt[:, frame_id, :])
        mesh_t_pred = copy.deepcopy(mesh)
        mesh_t_pred.vertices = o3d.utility.Vector3dVector(vtx_traj_pred[:, frame_id, :])
        pts = pts_traj[:, frame_id, :]
        mesh_t_gt_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_t_gt)
        mesh_t_gt_ls.paint_uniform_color([0.0, 0.0, 1.0])
        mesh_t_pred_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_t_pred)
        mesh_t_pred_ls.paint_uniform_color([1.0, 0.0, 0.0])
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
        pcd.paint_uniform_color([0.0, 0.0, 1.0])

        vis.add_geometry(mesh_t_pred_ls)
        vis.add_geometry(mesh_t_gt_ls)
        vis.add_geometry(pcd.translate([1.0, 0.0, 0.0]))
        vis.poll_events()
        vis.update_renderer()
        #vis.run()
        if output_folder is not None:
            image = vis.capture_screen_float_buffer()
            image = np.asarray(image) * 255
            image = image.astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{frame_id:05d}.png", image[:, 400:-400, ::-1]))
        vis.remove_geometry(mesh_t_pred_ls)
        vis.remove_geometry(mesh_t_gt_ls)
        vis.remove_geometry(pcd)
        #vis.destroy_window()
    vis.destroy_window()


def smooth_quats(mesh, rig, quats):
    # smooth
    for t in range(2):
        quats[:, 1:-1, :] = (quats[:, 1:-1, :] + 0.5 * quats[:, 2:, :] + 0.5 * quats[:, :-2, :]) / 2.0

    vtx_src = np.asarray(mesh.vertices)
    vert_src = np.column_stack((vtx_src, np.ones(len(vtx_src)))).T
    rig_globals_inv = np.linalg.inv(rig.global_transforms_homogeneous)
    vert_src_local = np.matmul(rig_globals_inv, vert_src[None, ...])
    vtx_traj = []
    for t in range(quats.shape[1]):
        rig_update = copy.deepcopy(rig)
        locals_t = Rotation.from_quat(quats[:, t, :]).as_matrix()
        rig_update.local_frames = locals_t
        rig_update.FK()
        vert_update = np.matmul(rig_update.global_transforms_homogeneous, vert_src_local)
        vtx_update_glb = np.sum(vert_update * rig_update.skins.T[:, None, :], axis=0)[0:3].T
        vtx_traj.append(vtx_update_glb)
    return np.stack(vtx_traj, axis=1), quats


if __name__ == "__main__":
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX2/zhan/morig/ModelsResources/obj_remesh/"
    sequence_folder = "/mnt/neghvar/mnt/DATA_LINUX2/zhan/morig/ModelsResources/test/"
    res_older = "../results/our_results/"
    output_folder = None

    pred_tracking_filelist = glob.glob(os.path.join(res_older, "tracking_loss/*.npz"))
    for pred_filename in pred_tracking_filelist:
        model_id = pred_filename.split("/")[-1].split("_")[0]
        print("model_id", model_id)
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{model_id}.obj"))
        gt_vtx_traj = np.load(os.path.join(sequence_folder, f"{model_id}_vtx_traj.npy"))
        gt_pts_traj = np.load(os.path.join(sequence_folder, f"{model_id}_pts_traj.npy")).reshape(-1, 101, 3)
        pred_results = np.load(pred_filename)
        pred_vtx_traj = pred_results["pred_vtx_traj"]

        #pred_quats = pred_results["pred_quats"]
        #rig = Rig(os.path.join(res_older, f"{model_id}_rig.txt"))
        #pred_vtx_traj = smooth_quats(mesh, rig, pred_quats)

        visualize_tracking(np.concatenate([gt_vtx_traj[:, 0:1, :], pred_vtx_traj], axis=1), gt_vtx_traj, gt_pts_traj, mesh, output_folder=output_folder)
