import sys
sys.path.append("./")
import glob, os, numpy as np, cv2, sys, math, scipy
import open3d as o3d
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import estimate_bandwidth
from utils import binvox_rw
from utils.os_utils import mkdir_p
from utils.io_utils import readPly
from utils.vis_utils import draw_shifted_pts, draw_joints, visualize_seg, visualize_seg_joints, show_obj_rig, drawSphere
from utils.mst_utils import flip, inside_check
from utils.rig_parser import Info
from utils.eval_utils import chamfer_dist
from utils.cluster_utils import meanshift_cluster, nms_meanshift

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_featuresize(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
    fs_dict = {}
    for li in lines:
        words = li.strip().split()
        fs_dict[words[0]] = float(words[1])
    return fs_dict


def get_joint_with_name(skel):
    joints = []
    names = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            joint_ = np.array(p_node.pos)
            joint_ = joint_[np.newaxis, :]
            joints.append(joint_)
            names.append(p_node.name)
            next_level += p_node.children
        this_level = next_level
    joints = np.concatenate(joints, axis=0)
    return joints, names


def eval_rig(bandwidth_quantile=0.04, threshold1=0.1, threshold2=0.02):
    mesh_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/obj_remesh/"
    info_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/rig_info_remesh/"
    vox_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/vox/"
    featuresize_folder = '/mnt/DATA_LINUX2/zhan/morig/ModelsResources/joint_featuresize/'
    ply_folder = "results/our_results/"
    attn_folder = "results/our_results/"
    output_folder = f"results/our_results/"

    mkdir_p(output_folder)
    ply_list = glob.glob(os.path.join(ply_folder, '*.ply'))
    chamfer_j2j_total = 0.0
    joint_IoU_total = 0.0
    joint_precision_total = 0.0
    joint_recall_total = 0.0
    num_invalid = 0

    for ply_filename in tqdm(ply_list):
        model_id = ply_filename.split('/')[-1].split('.')[0]
        attn_filename = os.path.join(attn_folder, f"{model_id}_attn.npy")
        mesh_filename = os.path.join(mesh_folder, '{:s}.obj'.format(model_id))
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        vtx = np.asarray(mesh.vertices)
        attn = np.load(attn_filename)
        attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))
        vox_file = os.path.join(vox_folder, '{:s}.binvox'.format(model_id))
        with open(vox_file, 'rb') as fvox:
            vox = binvox_rw.read_as_3d_array(fvox)
        shifted_pts = readPly(ply_filename)
        #img = draw_shifted_pts(mesh, shifted_pts, weights=attn)
        #cv2.imwrite(os.path.join(res_folder, "{:s}_pts.png".format(model_id)), img[:,:,::-1])

        shifted_pts, index_inside = inside_check(shifted_pts, vox)
        attn = attn[index_inside, :]
        shifted_pts = shifted_pts[attn.squeeze() > threshold1]
        attn = attn[attn.squeeze() > threshold1]

        # symmetrize points by reflecting
        shifted_pts_reflect = shifted_pts * np.array([[-1, 1, 1]])
        shifted_pts = np.concatenate((shifted_pts, shifted_pts_reflect), axis=0)
        attn = np.tile(attn, (2, 1))
        bandwidth = estimate_bandwidth(shifted_pts, quantile=bandwidth_quantile)
        #print(f"bandwidth: {bandwidth}")
        shifted_pts = meanshift_cluster(shifted_pts, bandwidth, attn, max_iter=30)
        #img = draw_shifted_pts(mesh, shifted_pts, weights=attn)

        pred_joints = nms_meanshift(shifted_pts, attn=attn, bandwidth=bandwidth, thrd_density=threshold2)
        pred_joints, _ = flip(pred_joints)
        #img = draw_joints(mesh_filename, pred_joints)
        #cv2.imwrite(os.path.join(res_folder, "{:s}_joint.png".format(model_id)), img[:,:,::-1])
        np.save(os.path.join(output_folder, "{:s}_joint.npy".format(model_id)), pred_joints)

        fs_file = os.path.join(featuresize_folder, f'{model_id}.txt')
        fs_dict = load_featuresize(fs_file)
        gt_skel = Info(os.path.join(info_folder, f'{model_id}.txt'))
        gt_joint, gt_joint_name = get_joint_with_name(gt_skel)
        fs = [fs_dict[i] for i in gt_joint_name]
        fs = np.array(fs)
        # print(len(gt_joint), len(pred_joint))
        if len(pred_joints) == 0:
            num_invalid += 1
            continue

        chamfer_j2j = chamfer_dist(pred_joints, gt_joint)
        chamfer_j2j_total += chamfer_j2j
        dist_matrix = np.sqrt(np.sum((pred_joints[np.newaxis, ...] - gt_joint[:, np.newaxis, :]) ** 2, axis=2))
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        fs_threshod = fs[row_ind]
        joint_IoU = 2 * np.sum(dist_matrix[row_ind, col_ind] < fs_threshod) / (len(pred_joints) + len(gt_joint))
        joint_IoU_total += joint_IoU
        joint_precision = np.sum(dist_matrix[row_ind, col_ind] < fs_threshod) / len(pred_joints)
        joint_precision_total += joint_precision
        joint_recall = np.sum(dist_matrix[row_ind, col_ind] < fs_threshod) / len(gt_joint)
        joint_recall_total += joint_recall
    print("num_invalid:", num_invalid)
    chamfer_j2j_total /= (len(ply_list) - num_invalid)
    joint_precision_total /= (len(ply_list) - num_invalid)
    joint_recall_total /= (len(ply_list) - num_invalid)
    joint_IoU_total /= (len(ply_list) - num_invalid)
    print('{:s}\n'.format(attn_folder),
          '\tJ2J_chamfer_distance {:.03f}%\n'.format(chamfer_j2j_total * 100),
          '\tjoint_IoU {:.03f}%\n'.format(joint_IoU_total * 100),
          '\tjoint_precision {:.03f}%\n'.format(joint_precision_total * 100),
          '\tjoint_recall {:.03f}%\n'.format(joint_recall_total * 100))


if __name__ == '__main__':
    #bandwidth_quantile, threshold1, threshold_density = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    #print(f"bandwidth_quantile: {bandwidth_quantile}, threshold1: {threshold1}, threshold_density: {threshold_density}")
    eval_rig() #bandwidth_quantile, threshold1, threshold_density