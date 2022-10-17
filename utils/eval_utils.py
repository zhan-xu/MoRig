#-------------------------------------------------------------------------------
# © 2019-2020  Zhan Xu
# Name:        eval_utils.py
# Purpose:     utilize functions for evaluation.
# Licence:     GNU General Public License v3
#-------------------------------------------------------------------------------
import numpy as np


def getJointNum(skel):
    this_level = [skel.root]
    n_joint = 1
    while this_level:
        next_level = []
        for p_node in this_level:
            n_joint += len(p_node.children)
            next_level += p_node.children
        this_level = next_level
    return n_joint


def dist_pts2bone(pts, pos_1, pos_2):
    l2 = np.sum((pos_2 - pos_1) ** 2)
    if l2 < 1e-10:
        dist_to_lineseg = np.linalg.norm(pts - pos_1, axis=1)
        dist_proj = np.linalg.norm(pts - pos_1, axis=1)
    else:
        t_ = np.sum((pts - pos_1[np.newaxis, :]) * (pos_2 - pos_1), axis=1) / l2
        t = np.clip(t_, 0, 1)
        t_pos = pos_1[np.newaxis, :] + t[:, np.newaxis] * (pos_2 - pos_1)[np.newaxis, :]
        lineseg_len = np.linalg.norm(pos_2 - pos_1)
        dist_proj = np.zeros(len(t_))
        dist_proj[np.argwhere(t_ < 0.5).squeeze()] = np.abs(t_[np.argwhere(t_ < 0.5).squeeze()] - 0.0) * lineseg_len
        dist_proj[np.argwhere(t_ >= 0.5).squeeze()] = np.abs(t_[np.argwhere(t_ >= 0.5).squeeze()] - 1.0) * lineseg_len
        dist_to_lineseg = np.linalg.norm(pts - t_pos, axis=1)
    return dist_to_lineseg, dist_proj


def chamfer_dist(pt1, pt2):
    pt1 = pt1[np.newaxis, :, :]
    pt2 = pt2[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    min_left = np.mean(np.min(dist, axis=0))
    min_right = np.mean(np.min(dist, axis=1))
    #print(min_left, min_right)
    return (min_left + min_right) / 2


def oneway_chamfer(pt_src, pt_dst):
    pt1 = pt_src[np.newaxis, :, :]
    pt2 = pt_dst[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    avg_dist = np.mean(np.min(dist, axis=0))
    return avg_dist


def getJointArr(skel):
    joints = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            joint_ = np.array(p_node.pos)
            joint_ = joint_[np.newaxis, :]
            joints.append(joint_)
            next_level += p_node.children
        this_level = next_level
    joints = np.concatenate(joints, axis=0)
    return joints


def sample_bone(p_pos, ch_pos):
    ray = ch_pos - p_pos
    bone_length = np.sqrt(np.sum((p_pos - ch_pos) ** 2))
    num_step = np.round(bone_length / 0.005)
    i_step = np.arange(0, num_step + 1)
    unit_step = ray / (num_step + 1e-30)
    unit_step = np.repeat(unit_step, num_step+1, axis=0)
    res = p_pos + unit_step * i_step[:, np.newaxis]
    return res


def sample_skel(skel):
    bone_sample = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            p_pos = np.array([p_node.pos])
            next_level += p_node.children
            for c_node in p_node.children:
                ch_pos = np.array([c_node.pos])
                res = sample_bone(p_pos, ch_pos)
                bone_sample.append(res)
        this_level = next_level
    bone_sample = np.concatenate(bone_sample, axis=0)
    return bone_sample


def bone2bone_chamfer_dist(skel_1, skel_2):
    bone_sample_1 = sample_skel(skel_1)
    bone_sample_2 = sample_skel(skel_2)
    pt1 = bone_sample_1[np.newaxis, :, :]
    pt2 = bone_sample_2[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    min_left = np.mean(np.min(dist, axis=0))
    min_right = np.mean(np.min(dist, axis=1))
    # print(min_left, min_right)
    return (min_left + min_right) / 2


def joint2bone_chamfer_dist(skel1, skel2):
    bone_sample_1 = sample_skel(skel1)
    bone_sample_2 = sample_skel(skel2)
    joint_1 = getJointArr(skel1)
    joint_2 = getJointArr(skel2)
    dist1 = oneway_chamfer(joint_1, bone_sample_2)
    dist2 = oneway_chamfer(joint_2, bone_sample_1)
    return (dist1 + dist2) / 2