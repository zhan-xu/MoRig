import open3d as o3d
import numpy as np
import glob
import os
import copy
import cv2
from utils.vis_utils import create_cityscapes_label_colormap, drawCone, drawSphere
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_color_map(x):
  colours = plt.cm.Spectral(x)
  return colours[:, :3]


def mesh_sphere(pcd, voxel_size, sphere_size=0.6):
  # Create a mesh sphere
  spheres = o3d.geometry.TriangleMesh()
  s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
  s.compute_vertex_normals()

  for i, p in enumerate(pcd.points):
    si = copy.deepcopy(s)
    trans = np.identity(4)
    trans[:3, 3] = p
    si.transform(trans)
    si.paint_uniform_color(pcd.colors[i])
    spheres += si
  return spheres


def get_colored_point_cloud_feature(pcd, feature, voxel_size):
  tsne_results = embed_tsne(feature)

  color = get_color_map(tsne_results)
  pcd.colors = o3d.utility.Vector3dVector(color)
  #spheres = mesh_sphere(pcd, voxel_size)

  #return spheres
  return pcd


def embed_tsne(data):
  """
  N x D np.array data
  """
  tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
  tsne_results = tsne.fit_transform(data)
  tsne_results = np.squeeze(tsne_results)
  tsne_min = np.min(tsne_results)
  tsne_max = np.max(tsne_results)
  return (tsne_results - tsne_min) / (tsne_max - tsne_min)


def visualize_corr():
    from DeformingThings.generate_data import normalize_obj
    mesh = o3d.io.read_triangle_mesh("/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/obj_remesh/9466.obj")
    pts_traj = np.load("/media/zhanxu/4T/motionAnim_results/important_results_backup/real_scan_final/monster2kate/kate_walking_pts_traj.npy")
    pts = pts_traj[:, 40, :]
    vtx = np.asarray(mesh.vertices)
    pts, _, _ = normalize_obj(pts)
    corr = np.load("../DeformingThings/monster2kate_frame40_corrmat.npy")
    vismask = np.load("../DeformingThings/monster2kate_frame40_vismask.npy").squeeze(axis=1)

    vismask = (vismask - np.min(vismask)) / (np.max(vismask) - np.min(vismask))
    vismask_color = plt.cm.YlOrRd(vismask)[:, :3]



    tsne_results = embed_tsne(vtx)
    vtx_color = get_color_map(tsne_results)

    mesh.compute_adjacency_list()
    vtx = np.asarray(mesh.vertices)
    for t in tqdm(range(2)):
        vtx_color_new = copy.deepcopy(vtx_color)
        vismask_color_new = copy.deepcopy(vismask_color)
        for vid in range(len(vtx)):
            if len(list(mesh.adjacency_list[vid])) > 0:
                vtx_color_new[vid] = np.mean(vtx_color[list(mesh.adjacency_list[vid])], axis=0)
                vismask_color_new[vid] = np.mean(vismask_color[list(mesh.adjacency_list[vid])], axis=0)
        vtx_color = vtx_color_new
        vismask_color = vismask_color_new


    mesh_vismask = copy.deepcopy(mesh)
    mesh_vismask.vertex_colors = o3d.utility.Vector3dVector(vismask_color)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vtx_color)
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(vtx_color[np.argmax(corr, axis=0)])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(pcd.translate([1.0, 0.0, 0.0]))
    vis.add_geometry(mesh_vismask.translate([2.0, 0.0, 0.0]))
    vis.run()
    vis.destroy_window()
    exit()

    res_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/corrnet/"
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/obj_remesh"
    seq_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/test"  # human
    vtx_list = glob.glob(os.path.join(res_folder, "*_vtx.npy"))
    for vtx_filename in vtx_list:
        name = int(vtx_filename.split('/')[-1].split('_')[0])
        #frame_id = '_'.join(vtx_filename.split('/')[-1].split('_')[0])
        print(name)
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{name}.obj"))
        vtx_traj = np.load(os.path.join(seq_folder, f"{name}_vtx_traj.npy"))
        vtx_new = vtx_traj[:, 60, :]

        mesh_new = copy.deepcopy(mesh)
        mesh_new.vertices = o3d.utility.Vector3dVector(vtx_new)
        mesh_new_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_new)
        mesh_new_ls.paint_uniform_color([.8, .8, .8])

        pts_filename = os.path.join(res_folder, f"{name}_pcd.npy")
        pts = np.load(pts_filename)
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
        vis_pcd = get_colored_point_cloud_feature(pcd, pts, voxel_size=0.025)

        pred_nnind = np.load(os.path.join(res_folder, f"{name}_nnind.npy"))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[pred_nnind])

        pred_vismask = np.load(os.path.join(res_folder, f"{name}_pred_vismask.npy")).squeeze(axis=1)
        pred_vismask = (pred_vismask - np.min(pred_vismask)) / (np.max(pred_vismask) - np.min(pred_vismask))
        pred_vismask_color = plt.cm.YlOrRd(pred_vismask)[:, :3]
        mesh_vismask = copy.deepcopy(mesh)
        mesh_vismask.vertex_colors = o3d.utility.Vector3dVector(pred_vismask_color)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_new_ls.translate([-1.0, 0.0, 0.0]))
        vis.add_geometry(vis_pcd)
        vis.add_geometry(mesh.translate([1.0, 0.0, 0.0]))
        vis.add_geometry(mesh_vismask.translate([2.0, 0.0, 0.0]))
        vis.run()
        # vis.poll_events()
        # vis.update_renderer()
        # image = vis.capture_screen_float_buffer()
        # image = np.asarray(image) * 255
        # image = image.astype(np.uint8)
        # cv2.imwrite(os.path.join(res_folder, "{:s}_{:d}_corr.png".format(name, frame_id)), image[200:-200, 500:-450,::-1])
        vis.destroy_window()


def visualize_corr_seq():
    res_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/corrnet_seq_deformingthings/"
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/DeformingThings4D/test"
    seq_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/DeformingThings4D/test"
    vtx_list = glob.glob(os.path.join(res_folder, "*_vtx.npy"))
    for vtx_filename in vtx_list:
        name = vtx_filename.split('/')[-1].split('_vtx.npy')[0]
        print(name)
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{name}.obj"))
        vtx_traj = np.load(os.path.join(seq_folder, f"{name}_vtx_traj.npy"))
        vtx_old = vtx_traj[:, 10, :]
        vtx_new = vtx_traj[:, 11, :]
        vtx = np.load(vtx_filename)

        mesh_new = copy.deepcopy(mesh)
        mesh_new.vertices = o3d.utility.Vector3dVector(vtx_new)
        mesh_new_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_new)
        mesh_new_ls.paint_uniform_color([.8, .8, .8])

        pts_filename = os.path.join(res_folder, f"{name}_pcd.npy")
        pts = np.load(pts_filename)
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
        vis_pcd = get_colored_point_cloud_feature(pcd, pts, voxel_size=0.025)

        pred_nnind = np.load(os.path.join(res_folder, f"{name}_nnind.npy"))
        mesh.vertices = o3d.utility.Vector3dVector(vtx_old)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[pred_nnind])

        pred_vismask = np.load(os.path.join(res_folder, f"{name}_pred_vismask.npy")).squeeze(axis=1)
        pred_vismask = (pred_vismask - np.min(pred_vismask)) / (np.max(pred_vismask) - np.min(pred_vismask))
        pred_vismask_color = plt.cm.YlOrRd(pred_vismask)[:, :3]
        mesh_vismask = copy.deepcopy(mesh)
        mesh_vismask.vertex_colors = o3d.utility.Vector3dVector(pred_vismask_color)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_new_ls.translate([-1.0, 0.0, 0.0]))
        vis.add_geometry(vis_pcd)
        vis.add_geometry(mesh.translate([1.0, 0.0, 0.0]))
        vis.add_geometry(mesh_vismask.translate([2.0, 0.0, 0.0]))
        vis.run()
        # vis.poll_events()
        # vis.update_renderer()
        image = vis.capture_screen_float_buffer()
        image = np.asarray(image) * 255
        image = image.astype(np.uint8)
        cv2.imwrite(os.path.join(res_folder, "{:s}_corr.png".format(name)), image[200:-200, 500:-450,::-1])
        vis.destroy_window()

if __name__ == "__main__":
    visualize_corr()
    #visualize_corr_seq()
