import copy

import numpy as np
import os
import glob
import open3d as o3d
import cv2
from utils.rot_utils import continuous6d2eular
import matplotlib.pyplot as plt

def visualize_attn():
    mesh_folder = "/media/zhanxu/4T/motionAnim_results/syn_res/"
    pred_attn_folder = "/media/zhanxu/4T/motionAnim_results/syn_res/"
    #pred_attn_folder2 = "/media/zhan/Data/motionAnim/results/ModelsResource_comparison/masknet_motion/"
    pred_attn_filelist = glob.glob(os.path.join(pred_attn_folder, "*_attn.txt"))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    for pred_attn_filename in pred_attn_filelist:
        model_id = pred_attn_filename.split("/")[-1].split("_")[0]
        pred_attn = np.loadtxt(pred_attn_filename).squeeze(axis=1)
        pred_attn = (pred_attn - pred_attn.min()) / (pred_attn.max() - pred_attn.min())
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{model_id}.obj"))
        vtx = np.array(mesh.vertices)
        mesh_subdivide = mesh.subdivide_loop(number_of_iterations=2)
        vtx_subdivide = np.asarray(mesh_subdivide.vertices)
        dist = np.sum((vtx_subdivide[:, None, :] - vtx[None, ...]) ** 2, axis=-1)
        nnidx = np.argmin(dist, axis=1)
        pred_attn_subdivide = pred_attn[nnidx]
        pred_attn_color = plt.cm.YlOrRd(pred_attn_subdivide)[:, :3]
        mesh_subdivide.vertex_colors = o3d.utility.Vector3dVector(pred_attn_color)

        # mesh_subdivide2 = copy.deepcopy(mesh_subdivide)
        # pred_attn2 = np.load(pred_attn_filename.replace(pred_attn_folder, pred_attn_folder2)).squeeze(axis=1)
        # pred_attn2 = (pred_attn2 - pred_attn2.min()) / (pred_attn2.max() - pred_attn2.min())
        # pred_attn_subdivide2 = pred_attn2[nnidx]
        # pred_attn_color2 = plt.cm.YlOrRd(pred_attn_subdivide2)[:, :3]
        # mesh_subdivide2.vertex_colors = o3d.utility.Vector3dVector(pred_attn_color2)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        vis.add_geometry(mesh_subdivide)
        #vis.add_geometry(mesh_subdivide2.translate([1.0, 0.0, 0.0]))

        #ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters('sideview.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        #vis.run()
        vis.poll_events()
        vis.update_renderer()
        #param = ctr.convert_to_pinhole_camera_parameters()
        #o3d.io.write_pinhole_camera_parameters('sideview2.json', param)
        vis.capture_screen_image(f"/media/zhanxu/4T/motionAnim_results/20220204/masknet_motion/{model_id}_pred_attn.png")
        vis.remove_geometry(mesh_subdivide)
    vis.destroy_window()


def visualize_shift_pts():
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/obj_remesh/"
    pts_shift_folder = "/mnt/swarm/mnt/nfs/home/zhanxu/Proj/motion_anim_v3/results/pts_shift_train_nonhuman_test_human/"
    ply_filelist = glob.glob(os.path.join(pts_shift_folder, f"*.ply"))
    for ply_filename in ply_filelist:
        model_id = ply_filename.split("/")[-1].split(".")[0]
        pts_shift = readPly(ply_filename)
        mesh_filename = os.path.join(mesh_folder, f"{model_id}.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_ls.paint_uniform_color([0.8, 0.8, 0.8])
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_shift))
        pcd.paint_uniform_color([1.0, 0.0, 0.0])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_ls)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    visualize_attn()
