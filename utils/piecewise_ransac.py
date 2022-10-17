import numpy as np
import open3d as o3d
import itertools
import copy
from utils.rig_parser import Rig
from utils.vis_utils import drawSphere

class Piecewise_RANSAC:
    def __init__(self, vismask_threshold):
        self.vismask_threshold = vismask_threshold

    def icp(self, src_pts, tar_pts):
        sptb = src_pts - np.mean(src_pts, axis=0, keepdims=True)
        tptb = tar_pts - np.mean(tar_pts, axis=0, keepdims=True)
        M = np.matmul(tptb.T, sptb)
        U, s, Vh = np.linalg.svd(M)
        R = np.matmul(U, Vh)
        if np.linalg.det(R) < 0:
            Vh[-1, :] *= -1
            R = np.matmul(U, Vh)
        t = np.mean(tar_pts - np.matmul(src_pts, R.T), axis=0, keepdims=True)
        return R, t

    def ransac_voting(self, src_pts, tar_pts):
        R_best = None
        T_best = None
        max_inlier = 0
        best_inliers = None
        error_best = 1e10
        for i in range(100):
            sample_id = np.random.choice(range(len(src_pts)), 3, replace=False)
            R, t = self.icp(src_pts[sample_id], tar_pts[sample_id])
            tar_pred = np.matmul(src_pts, R.T) + t
            inliers = np.argwhere(np.sqrt(np.sum((tar_pred - tar_pts) ** 2, axis=1)) < 5e-2)  # 1e-3
            fitting_error = np.sqrt(np.sum((tar_pred - tar_pts) ** 2, axis=1)).sum()
            if len(inliers) > max_inlier:
                max_inlier = len(inliers)
                best_inliers = inliers.squeeze(axis=1)
            if fitting_error < error_best: # no qualified transformation found, use the one with least sum of fitting error
                R_best = R
                T_best = t
                error_best = fitting_error
        if best_inliers is not None and len(best_inliers) > 0.35 * len(src_pts):
            R_best, T_best = self.icp(src_pts[best_inliers], tar_pts[best_inliers])
        return R_best, T_best

    def visualize_handles(self, verts_ori, verts_tar, handles):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd_ori = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts_ori))
        pcd_tar = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts_tar))
        # color_ori = np.repeat(np.array([[1.0, 0.0, 0.0]]), len(verts_ori), axis=0)
        # color_tar = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(verts_tar), axis=0)
        pcd_ori.paint_uniform_color([0.95, 0.8, 0.8])
        pcd_tar.paint_uniform_color([0.8, 0.8, 0.95])
        if handles is not None:
            corr_pts = np.concatenate((verts_ori[handles], verts_tar[handles]), axis=0)
            corr_lines = np.array([[i, i + len(handles)] for i in range(len(handles))])
            corr_line_set = o3d.geometry.LineSet()
            corr_line_set.points = o3d.utility.Vector3dVector(corr_pts)
            corr_line_set.lines = o3d.utility.Vector2iVector(corr_lines)
            colors = [[1.0, 0.0, 0.0] for i in range(len(corr_lines))]
            corr_line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(corr_line_set)
        vis.add_geometry(pcd_ori)
        vis.add_geometry(pcd_tar)
        vis.run()
        vis.destroy_window()

    def renumber_label(self, label_in):
        label_out = np.zeros_like(label_in)
        new_l = 0
        for l in np.unique(label_in):
            label_out[np.argwhere(label_in == l).squeeze(axis=1)] = new_l
            new_l += 1
        return label_out

    def run(self, vert_src, vert_dst, vismask, seg):
        seg = self.renumber_label(seg)
        for l in np.unique(seg):
            handles_p = np.logical_and(vismask >= self.vismask_threshold, seg == l)
            handles_p = np.argwhere(handles_p).squeeze(axis=1)
            #self.visualize_handles(vert_src, vert_dst, handles_p)
            if len(handles_p) < 4:
                vert_src[seg == l] = vert_dst[seg == l]
                continue
            src_pts = vert_src[handles_p]
            tar_pts = vert_dst[handles_p]
            R, t = self.ransac_voting(src_pts, tar_pts)
            vert_src[seg == l] = np.matmul(vert_src[seg == l], R.T) + t
            #self.visualize_handles(vert_src, vert_dst, handles_p)
        return vert_src


if __name__ == "__main__":
    mesh_1 = o3d.io.read_triangle_mesh("/mnt/neghvar/mnt/DATA_LINUX/zhan/models_resource/test/15135_0.obj")
    mesh_2 = o3d.io.read_triangle_mesh("/mnt/neghvar/mnt/DATA_LINUX/zhan/models_resource/test/15135_5.obj")
    vert_1 = np.asarray(mesh_1.vertices)
    vert_2 = np.asarray(mesh_2.vertices)
    rig = Rig("/mnt/neghvar/mnt/DATA_LINUX/zhan/models_resource/test/15135_0_rig.txt")
    seg = np.argmax(rig.skins, axis=1)
    vismask = np.load("/mnt/neghvar/mnt/DATA_LINUX/zhan/models_resource/test/15135_5_vismask.npy")
    deformer = Piecewise_RANSAC(vismask_threshold=0.3)
    #deformer.visualize_handles(vert_1, vert_2, None)
    deformer.visualize_handles(vert_1, vert_2, np.argwhere(vismask>0.3).squeeze())
    vert_out = deformer.run(vert_1, vert_2, vismask, seg)
    deformer.visualize_handles(vert_out, vert_2, np.argwhere(vismask>0.3).squeeze())