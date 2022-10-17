#-------------------------------------------------------------------------------
# Name:        mst_utils.py
# Purpose:     utilize functions for skeleton generation
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
import numpy as np
from utils.rig_parser import Rig, TreeNode
from utils.vis_utils import visualize_seg, drawSphere
import open3d as o3d

def inside_check(pts, vox):
    """
    Check where points are inside or outside the mesh based on its voxelization.
    :param pts: points to be checked
    :param vox: voxelized mesh
    :return: internal points, and index of them in the input array.
    """
    vc = (pts - vox.translate) / vox.scale * vox.dims[0]
    vc = np.round(vc).astype(int)
    ind1 = np.logical_and(np.all(vc >= 0, axis=1), np.all(vc < 88, axis=1))
    vc = np.clip(vc, 0, 87)
    ind2 = vox.data[vc[:, 0], vc[:, 1], vc[:, 2]]
    ind = np.logical_and(ind1, ind2)
    pts = pts[ind]
    return pts, np.argwhere(ind).squeeze()


def sample_on_bone(p_pos, ch_pos, step_size=0.01):
    """
    sample points on a bone
    :param p_pos: parent joint position
    :param ch_pos: child joint position
    :return: a array of samples on this bone.
    """
    ray = ch_pos - p_pos
    bone_length = np.sqrt(np.sum((p_pos - ch_pos) ** 2))
    num_step = np.round(bone_length / step_size)
    i_step = np.arange(1, num_step + 1)
    unit_step = ray / (num_step + 1e-30)
    unit_step = np.repeat(unit_step[np.newaxis, :], num_step, axis=0)
    res = p_pos + unit_step * i_step[:, np.newaxis]
    return res

def get_all_bone_samples(rig, step_size=0.01):
    bone_samples = []
    this_level = [rig.root_id]
    while this_level:
        next_level = []
        for p_id in this_level:
            ch_list = np.argwhere(rig.hierarchy == p_id).squeeze(axis=1)
            for ch_id in ch_list:
                bone_samples_this = sample_on_bone(rig.pos[p_id], rig.pos[ch_id], step_size=step_size)
                bone_samples.append(bone_samples_this)
            next_level += ch_list.tolist()
        this_level = next_level
    return np.concatenate(bone_samples, axis=0)


def minKey(key, mstSet, nV):
    # Initilaize min value
    min = sys.maxsize
    for v in range(nV):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v
    return min_index


def primMST(graph, init_id):
    """
    Original prim MST algorithm https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
    """
    nV = graph.shape[0]
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = [False] * nV
    # Make key init_id so that this vertex is picked as first vertex
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    for cout in range(nV):
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet, nV)

        # Put the minimum distance vertex in
        # the shortest path tree
        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(nV):
            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if graph[u,v] > 0 and mstSet[v] == False and key[v] > graph[u,v]:
                key[v] = graph[u,v]
                parent[v] = u

    return parent, key


def primMST_symmetry(graph, init_id, joints):
    """
    my modified prim algorithm to generate a tree as symmetric as possible.
    Not guaranteed to be symmetric. All heuristics.
    :param graph: pairwise cost matrix
    :param init_id: init node ID as root
    :param joints: joint positions J*3
    :return:
    """
    joint_mapping = {}
    left_joint_ids = np.argwhere(joints[:, 0] < -2e-2).squeeze(1).tolist()
    middle_joint_ids = np.argwhere(np.abs(joints[:, 0]) <= 2e-2).squeeze(1).tolist()
    right_joint_ids = np.argwhere(joints[:, 0] > 2e-2).squeeze(1).tolist()

    joints_left = joints[left_joint_ids]
    joints_middle = joints[middle_joint_ids]
    joints_right = joints[right_joint_ids]

    for i in range(len(left_joint_ids)):
        joint_map_id = np.argmin(np.sum((joints_right * np.array([[-1, 1, 1]]) - joints_left[i][None,:]) ** 2, axis=-1))
        if np.linalg.norm(joints_right[joint_map_id] * np.array([[-1, 1, 1]]) - joints_left[i]) < 1e-3:
            joint_mapping[left_joint_ids[i]] = right_joint_ids[joint_map_id]
    for i in range(len(right_joint_ids)):
        joint_map_id = np.argmin(np.sum((joints_left * np.array([[-1, 1, 1]]) - joints_right[i][None, :]) ** 2, axis=-1))
        if np.linalg.norm(joints_left[joint_map_id] * np.array([[-1, 1, 1]]) - joints_right[i]) < 1e-3:
            joint_mapping[right_joint_ids[i]] = left_joint_ids[joint_map_id]

    if init_id not in middle_joint_ids:
        #find nearest joint in the middle to be root
        if len(middle_joint_ids) > 0:
            nearest_id = np.argmin(np.linalg.norm(joints[middle_joint_ids, :] - joints[init_id, :][np.newaxis, :], axis=1))
            init_id = middle_joint_ids[nearest_id]

    nV = graph.shape[0]
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = [False] * nV
    # Make key init_id so that this vertex is picked as first vertex
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    while not all(mstSet):
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet, nV)
        # left cases
        if u in left_joint_ids and u in joint_mapping.keys() and parent[u] in middle_joint_ids:
            if u in joint_mapping.keys():
                u2 = joint_mapping[u]
                if mstSet[u2] is False:
                    mstSet[u2] = True
                    parent[u2] = parent[u]
                    key[u2] = graph[u2, parent[u2]]
        elif u in left_joint_ids and u in joint_mapping.keys() and parent[u] in left_joint_ids and parent[u] in joint_mapping.keys():
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
        elif u in left_joint_ids and u in joint_mapping.keys() and parent[u] in right_joint_ids and parent[u] in joint_mapping.keys():
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]

        # right cases
        elif u in right_joint_ids and u in joint_mapping.keys() and parent[u] in middle_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = parent[u]
                key[u2] = graph[u2, parent[u2]]
        elif u in right_joint_ids and u in joint_mapping.keys() and parent[u] in right_joint_ids and parent[u] in joint_mapping.keys():
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
        elif u in right_joint_ids and u in joint_mapping.keys() and parent[u] in left_joint_ids and parent[u] in joint_mapping.keys():
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
        # middle case
        else:
            u2 = None

        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(nV):
            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if graph[u,v] > 0 and mstSet[v] == False and key[v] > graph[u,v]:
                key[v] = graph[u,v]
                parent[v] = u
            if u2 is not None and graph[u2,v] > 0 and mstSet[v] == False and key[v] > graph[u2,v]:
                key[v] = graph[u2, v]
                parent[v] = u2

    return parent, key, init_id


def primMST_middle_first(graph, init_id, joints):
    left_joint_ids = np.argwhere(joints[:, 0] < -2e-2).squeeze(1).tolist()
    middle_joint_ids = np.argwhere(np.abs(joints[:, 0]) <= 2e-2).squeeze(1).tolist()
    right_joint_ids = np.argwhere(joints[:, 0] > 2e-2).squeeze(1).tolist()

    if init_id not in middle_joint_ids:
        #find nearest joint in the middle to be root
        if len(middle_joint_ids) > 0:
            nearest_id = np.argmin(np.linalg.norm(joints[middle_joint_ids, :] - joints[init_id, :][np.newaxis, :], axis=1))
            init_id = middle_joint_ids[nearest_id]

    nV = graph.shape[0]
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = np.array([False] * nV, dtype=np.bool)
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    # first connect middle joints
    while not np.all(mstSet[middle_joint_ids]):
        min = sys.maxsize
        for v in middle_joint_ids:
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                u = v
        mstSet[u] = True
        for v in range(nV):
            if graph[u, v] > 0 and mstSet[v] == False and key[v] > graph[u, v]:
                key[v] = graph[u, v]
                parent[v] = u

    # second connect other joints
    while not np.all(mstSet):
        min = sys.maxsize
        for v in range(nV):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                u = v
        mstSet[u] = True
        for v in range(nV):
            if graph[u, v] > 0 and mstSet[v] == False and key[v] > graph[u, v]:
                key[v] = graph[u, v]
                parent[v] = u

    return parent, key, init_id


def increase_cost_for_outside_bone(cost_matrix, joint_pos, vox):
    """
    increase connectivity cost for bones outside the meshs
    """
    for i in range(len(joint_pos)):
        for j in range(i+1, len(joint_pos)):
            bone_samples = sample_on_bone(joint_pos[i], joint_pos[j])
            bone_samples_vox = (bone_samples - vox.translate) / vox.scale * vox.dims[0]
            bone_samples_vox = np.round(bone_samples_vox).astype(int)

            ind1 = np.logical_and(np.all(bone_samples_vox >= 0, axis=1), np.all(bone_samples_vox < vox.dims[0], axis=1))
            bone_samples_vox = np.clip(bone_samples_vox, 0, vox.dims[0]-1)
            ind2 = vox.data[bone_samples_vox[:, 0], bone_samples_vox[:, 1], bone_samples_vox[:, 2]]
            in_flags = np.logical_and(ind1, ind2)
            outside_bone_sample = np.sum(in_flags == False)

            if outside_bone_sample > 1:
                cost_matrix[i, j] = 2 * outside_bone_sample
                cost_matrix[j, i] = 2 * outside_bone_sample
            if np.abs(joint_pos[i, 0]) < 2e-2 and np.abs(joint_pos[j, 0]) < 2e-2:
                cost_matrix[i, j] *= 0.5
                cost_matrix[j, i] *= 0.5
    return cost_matrix


def flip(pred_joints):
    """
    symmetrize the predicted joints by reflecting joints on the left half space to the right
    :param pred_joints: raw predicted joints
    :return: symmetrized predicted joints
    """
    pred_joints_left = pred_joints[np.argwhere(pred_joints[:, 0] < -2e-2).squeeze(), :]
    pred_joints_middle = pred_joints[np.argwhere(np.abs(pred_joints[:, 0]) <= 2e-2).squeeze(), :]

    if pred_joints_left.ndim == 1:
        pred_joints_left = pred_joints_left[np.newaxis, :]
    if pred_joints_middle.ndim == 1:
        pred_joints_middle = pred_joints_middle[np.newaxis, :]

    pred_joints_middle[:, 0] = 0.0
    pred_joints_right = np.copy(pred_joints_left)
    pred_joints_right[:, 0] = -pred_joints_right[:, 0]
    pred_joints_res = np.concatenate((pred_joints_left, pred_joints_middle, pred_joints_right), axis=0)
    side_indicator = np.concatenate((-np.ones(len(pred_joints_left)), np.zeros(len(pred_joints_middle)), np.ones(len(pred_joints_right))), axis=0)
    return pred_joints_res, side_indicator


def chamfer_dist(pts1, pts2):
    dist_mat = np.sqrt(np.sum((pts1[np.newaxis, ...] - pts2[:, np.newaxis, :])**2, axis=-1))
    dist1 = np.min(dist_mat, axis=0)
    dist2 = np.min(dist_mat, axis=1)
    dist = 0.5 * ((np.mean(dist1)) + (np.mean(dist2)))
    return dist


def determin_flip_src_tar(label, verts):
    vid_left = np.argwhere(verts[:, 0] <= 0).squeeze(axis=1)
    vid_right = np.argwhere(verts[:, 0] > 0).squeeze(axis=1)
    verts_left = verts[vid_left]
    verts_right = verts[vid_right]
    chf_dist_left = chamfer_dist(verts_left, np.array(
        [np.mean(verts_left[label[vid_left]] == l, axis=0) for l in np.unique(label[vid_left])]))
    chf_dist_right = chamfer_dist(verts_right, np.array(
        [np.mean(verts_right[label[vid_right]] == l, axis=0) for l in np.unique(label[vid_right])]))
    if chf_dist_left < chf_dist_right:
        return "left"
    else:
        return "right"


def gen_tpl_adj(mesh):
    nv = len(mesh.vertices)
    tri = np.asarray(mesh.triangles, dtype=np.int)
    adj = np.zeros((nv, nv))
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        adj[tri[:, i], tri[:, j]] = 1
    adj = np.maximum(adj, adj.T)
    return adj


def flip_seg(label, mesh):
    num_label = np.max(label)
    adj = gen_tpl_adj(mesh)
    verts = np.asarray(mesh.vertices)
    preserve_side = determin_flip_src_tar(label, verts)
    if preserve_side == "left":
       vid_src = np.argwhere(verts[:, 0] <= 0).squeeze(axis=1)
       vid_tar = np.argwhere(verts[:, 0] > 0).squeeze(axis=1)
    else:
        vid_tar = np.argwhere(verts[:, 0] <= 0).squeeze(axis=1)
        vid_src = np.argwhere(verts[:, 0] > 0).squeeze(axis=1)
    verts_src = verts[vid_src]
    verts_tar = verts[vid_tar]
    label_src = label[vid_src]

    verts_src_reflect = verts_src.copy()
    verts_src_reflect[:, 0] *= -1
    dist = np.sqrt(np.sum((verts_tar[:, None, :] - verts_src_reflect[None, ...]) ** 2, axis=-1))
    nnidx = np.argmin(dist, axis=1)
    valid_pair = np.argwhere(dist.min(axis=1) < 0.05).squeeze(axis=1)
    label[vid_tar[valid_pair]] = label_src[nnidx[valid_pair]] + num_label + 1
    #visualize_seg(verts, label, mesh)
    for l_src in np.unique(label_src):
        vid_src = np.argwhere(label==l_src).squeeze(axis=1)
        vid_tar = np.argwhere(label==(l_src+num_label+1)).squeeze(axis=1)
        if len(vid_src) == 0 or len(vid_tar) == 0:
            continue
        adj_sub = adj[vid_src,:][:,vid_tar]

        '''pcd_src = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts[vid_src]))
        pcd_src.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_tar = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts[vid_tar]))
        pcd_tar.paint_uniform_color([0.0, 0.0, 1.0])
        mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_ls.paint_uniform_color([0.5, 0.5, 0.5])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_src)
        vis.add_geometry(pcd_tar)
        vis.add_geometry(mesh_ls)
        vis.run()
        vis.destroy_window()'''

        if np.abs(adj_sub.max()-1) < 1e-4:
            label[vid_tar] = l_src
    #visualize_seg(verts, label, mesh)
    return label


def get_pivot(j_parent, j_child, v_parent, v_children):
    dist = np.sqrt(np.sum((v_children[np.newaxis, ...] - v_parent[:, np.newaxis, :]) ** 2, axis=-1))
    #ch_boundary_pids = np.argwhere(np.min(dist, axis=0) <= np.percentile(np.min(dist, axis=0), 5)).squeeze(axis=1)
    #pa_boundary_pids = np.argwhere(np.min(dist, axis=1) <= np.percentile(np.min(dist, axis=1), 5)).squeeze(axis=1)
    adjacent_posid = np.argwhere(dist < np.percentile(dist, 5))
    if len(adjacent_posid) == 0:
        return np.mean(np.concatenate([v_children, v_parent], axis=0), axis=0)
    pa_boundary_pids = np.unique(adjacent_posid[:, 0])
    ch_boundary_pids = np.unique(adjacent_posid[:, 1])
    adjacent_pos_p1 = v_parent[pa_boundary_pids]
    adjacent_pos_p2 = v_children[ch_boundary_pids]
    mean_pos = np.mean(np.concatenate([adjacent_pos_p1, adjacent_pos_p2], axis=0), axis=0)
    #mean_pos = np.mean(adjacent_pos_p2, axis=0)
    '''vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_ch = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts_children))
    color_ch = np.ones((len(verts_children), 3)) * np.array([[1.0, 0.1, 0.1]])
    color_ch[ch_boundary_pids] = np.array([0.6, 0.0 ,0.0])
    pcd_ch.colors = o3d.utility.Vector3dVector(color_ch)
    pcd_pa = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts_parent))
    color_pa = np.ones((len(verts_parent), 3)) * np.array([[0.1, 0.1, 1.0]])
    color_pa[pa_boundary_pids] = np.array([0.0, 0.0 ,0.6])
    pcd_pa.colors = o3d.utility.Vector3dVector(color_pa)
    vis.add_geometry(pcd_pa)
    vis.add_geometry(pcd_ch)
    vis.run()
    vis.destroy_window()'''
    return mean_pos


def move_joints_to_boundary(mesh, skel, label):
    vtx = np.asarray(mesh.vertices)
    samples = np.asarray(mesh.sample_points_poisson_disk(number_of_points=4000).points)
    #np.save("debug_samples.npy", samples)
    #samples = np.load("debug_samples.npy")
    dist_s2v = np.sqrt(np.sum((samples[np.newaxis, ...] - vtx[:, np.newaxis, :]) ** 2, axis=-1))
    label_samples = label[np.argmin(dist_s2v, axis=0)]
    if (label_samples==skel.root_id).sum() == 0:
        skel.pos[skel.root_id] = np.mean(vtx[label == skel.root_id], axis=0)
    else:
        skel.pos[skel.root_id] = np.mean(samples[label_samples==skel.root_id], axis=0)
    this_level = [skel.root_id]
    while this_level:
        next_level = []
        for p_id in this_level:
            ch_list = np.argwhere(skel.hierarchy == p_id).squeeze(axis=1)
            for ch_id in ch_list:
                if len(samples[label_samples==ch_id]) == 0 or len(samples[label_samples==p_id])==0:
                    pivot = get_pivot(skel.pos[p_id], skel.pos[ch_id], vtx[label == p_id], vtx[label == ch_id])
                else:
                    pivot = get_pivot(skel.pos[p_id], skel.pos[ch_id], samples[label_samples==p_id], samples[label_samples==ch_id])
                skel.pos[ch_id] = pivot
            next_level += ch_list.tolist()
        this_level = next_level
    return skel

def loadSkel_recur(p_node, parent_id, joint_name, joint_pos, parent):
    """
    Converst prim algorithm result to our skel/info format recursively
    :param p_node: Root node
    :param parent_id: parent name of current step of recursion.
    :param joint_name: list of joint names
    :param joint_pos: joint positions
    :param parent: parent index returned by prim alg.
    :return: p_node (root) will be expanded to linked with all joints
    """
    for i in range(len(parent)):
        if parent[i] == parent_id:
            if joint_name is not None:
                ch_node = TreeNode(joint_name[i], tuple(joint_pos[i]))
            else:
                ch_node = TreeNode('joint_{}'.format(i), tuple(joint_pos[i]))
            p_node.children.append(ch_node)
            ch_node.parent = p_node
            loadSkel_recur(ch_node, i, joint_name, joint_pos, parent)