import open3d as o3d
import numpy as np
import glob
import os
import copy
import cv2
from utils.vis_utils import create_cityscapes_label_colormap, drawCone, drawSphere


def visualize_flow():
    show_predicion = True
    res_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/deformnet_modelsresources/"
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX2/zhan/morig/ModelsResources/obj_remesh"
    target_frame_ids = [3] # [1, 2, 3, 4, 5]
    pred_flow_filelist = glob.glob(os.path.join(res_folder, "*_pred_flow.npy"))
    for pred_flow in pred_flow_filelist:
        model_id = pred_flow.split('/')[-1].split('_')[0]
        print(model_id)
        mesh_filename = os.path.join(mesh_folder, f"{model_id}.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        mesh_ori_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vtx_0 = np.asarray(mesh.vertices)

        pts_t = np.load(os.path.join(res_folder, f"{model_id}_tar_pts.npy"))
        vtx_t = np.load(os.path.join(res_folder, f"{model_id}_tar_vtx.npy"))
        pred_flow_t = np.load(os.path.join(res_folder, f"{model_id}_pred_flow.npy"))

        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_t))
        pcd.paint_uniform_color([0.4, 0.4, 1.0])

        mesh_gt = copy.deepcopy(mesh)
        mesh_gt.vertices = o3d.utility.Vector3dVector(vtx_t)
        mesh_gt_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_gt)

        mesh_pred = copy.deepcopy(mesh)
        mesh_pred.vertices = o3d.utility.Vector3dVector(vtx_0 + pred_flow_t)
        mesh_pred_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_pred)

        mesh_pred_ls.paint_uniform_color([1.0, 0.4, 0.4])
        mesh_gt_ls.paint_uniform_color([0.4, 0.4, 1.0])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_ori_ls)
        vis.add_geometry(mesh_gt_ls.translate([1.0, 0.0, 0.0]))
        vis.add_geometry(mesh_pred_ls.translate([1.0, 0.0, 0.0]))
        vis.add_geometry(pcd.translate([2.0, 0.0, 0.0]))
        vis.run()
        #vis.poll_events()
        #vis.update_renderer()
        #image = vis.capture_screen_float_buffer()
        #image = np.asarray(image) * 255
        #image = image.astype(np.uint8)
        #cv2.imwrite(vtx_filename.replace("_vtx.npy", "_pred_flow.png"), image[300:-300, 500:-500, ::-1])
        #vis.remove_geometry(vert_ori_pcd)
        #vis.remove_geometry(vert_shift_pcd)
        #vis.remove_geometry(pcd)
        vis.destroy_window()


def visualize_flow_seq():
    #mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX2/zhan/morig/DeformingThings4D/test"
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX2/zhan/morig/ModelsResources/obj_remesh"
    res_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/deformnet_modelsresources_seq/"
    file_list = glob.glob(os.path.join(res_folder, "*_src_vtx.npy"))
    for vtx_filename in file_list:
        model_id = vtx_filename.split("/")[-1].split("_src_vtx.npy")[0]
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{model_id}.obj"))

        vtx_src = np.load(vtx_filename)
        vtx_tar = np.load(vtx_filename.replace("_src_vtx.npy", "_tar_vtx.npy"))
        pts_tar = np.load(vtx_filename.replace("_src_vtx.npy", "_tar_pts.npy"))
        pred_flow = np.load(vtx_filename.replace("_src_vtx.npy", "_pred_flow.npy"))
        vtx_pred = vtx_src + pred_flow

        mesh_src = copy.deepcopy(mesh)
        mesh_src.vertices = o3d.utility.Vector3dVector(vtx_src)
        mesh_src_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_src)
        mesh_src_ls.paint_uniform_color([0.0, 1.0, 0.0])
        mesh_gt = copy.deepcopy(mesh)
        mesh_gt.vertices = o3d.utility.Vector3dVector(vtx_tar)
        mesh_gt_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_gt)
        mesh_gt_ls.paint_uniform_color([0.0, 0.0, 1.0])
        mesh_deform = copy.deepcopy(mesh)
        mesh_deform.vertices = o3d.utility.Vector3dVector(vtx_pred)
        mesh_deform_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_deform)
        mesh_deform_ls.paint_uniform_color([1.0, 0.0, 0.0])
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_tar))
        pcd.paint_uniform_color([0.0, 0.0, 1.0])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd.translate([1.0, 0.0, 0.0]))
        vis.add_geometry(mesh_gt_ls)
        vis.add_geometry(mesh_deform_ls)
        vis.add_geometry(mesh_src_ls)

        #vis.poll_events()
        #vis.update_renderer()
        vis.run()
        image = vis.capture_screen_float_buffer()
        image = np.asarray(image) * 255
        image = image.astype(np.uint8)
        cv2.imwrite(vtx_filename.replace("_src_vtx.npy", "flow.png"), image)
        vis.destroy_window()


if __name__ == "__main__":
    visualize_flow()
    #visualize_flow_seq()
