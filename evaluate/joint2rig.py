import sys
sys.path.append("./")
import glob, os, open3d as o3d, numpy as np, itertools as it, cv2, copy, sys, time, trimesh
from tqdm import tqdm
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from utils import binvox_rw
from utils.vis_utils import draw_shifted_pts, show_obj_rig, drawCone, drawSphere
from utils.rig_parser import Info, Rig, Node, TreeNode
from utils.mst_utils import inside_check, sample_on_bone, increase_cost_for_outside_bone, primMST, primMST_symmetry

from data_proc.common_ops import calc_surface_geodesic
from data_proc.gen_skin_data import get_bones

from models.rootnet import ROOTNET
from models.bonenet import PairCls as BONENET
from models.rignet import SkinMotion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getInitId(data, model):
    """
    predict root joint ID via rootnet
    :param data:
    :param model:
    :return:
    """
    with torch.no_grad():
        root_prob, _ = model(data, shuffle=False)
        root_prob = torch.sigmoid(root_prob).data.cpu().numpy()
    root_id = np.argmax(root_prob)
    return root_id


def pts2line(pts, lines):
    '''
    Calculate points-to-bone distance. Point to line segment distance refer to
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    :param pts: N*3
    :param lines: N*6, where [N,0:3] is the starting position and [N, 3:6] is the ending position
    :return: origins are the neatest projected position of the point on the line.
             ends are the points themselves.
             dist is the distance in between, which is the distance from points to lines.
             Origins and ends will be used for generate rays.
    '''
    l2 = np.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, axis=1)
    origins = np.zeros((len(pts) * len(lines), 3))
    ends = np.zeros((len(pts) * len(lines), 3))
    dist = np.zeros((len(pts) * len(lines)))
    for l in range(len(lines)):
        if np.abs(l2[l]) < 1e-8:  # for zero-length edges
            origins[l * len(pts):(l + 1) * len(pts)] = lines[l][0:3]
        else:  # for other edges
            t = np.sum((pts - lines[l][0:3][np.newaxis, :]) * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :], axis=1) / \
                l2[l]
            t = np.clip(t, 0, 1)
            t_pos = lines[l][0:3][np.newaxis, :] + t[:, np.newaxis] * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :]
            origins[l * len(pts):(l + 1) * len(pts)] = t_pos
        ends[l * len(pts):(l + 1) * len(pts)] = pts
        dist[l * len(pts):(l + 1) * len(pts)] = np.linalg.norm(
            origins[l * len(pts):(l + 1) * len(pts)] - ends[l * len(pts):(l + 1) * len(pts)], axis=1)
    return origins, ends, dist


def calc_pts2bone_visible_mat(mesh, origins, ends):
    '''
    Check whether the surface point is visible by the internal bone.
    Visible is defined as no occlusion on the path between.
    :param mesh:
    :param surface_pts: points on the surface (n*3)
    :param origins: origins of rays
    :param ends: ends of the rays, together with origins, we can decide the direction of the ray.
    :return: binary visibility matrix (n*m), where 1 indicate the n-th surface point is visible to the m-th ray
    '''
    ray_dir = ends - origins
    RayMeshIntersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = RayMeshIntersector.intersects_location(origins, ray_dir + 1e-15)
    locations_per_ray = [locations[index_ray == i] for i in range(len(ray_dir))]
    min_hit_distance = []
    for i in range(len(locations_per_ray)):
        if len(locations_per_ray[i]) == 0:
            min_hit_distance.append(np.linalg.norm(ray_dir[i]))
        else:
            min_hit_distance.append(np.min(np.linalg.norm(locations_per_ray[i] - origins[i], axis=1)))
    min_hit_distance = np.array(min_hit_distance)
    distance = np.linalg.norm(ray_dir, axis=1)
    vis_mat = (np.abs(min_hit_distance - distance) < 1e-4)
    return vis_mat


def add_duplicate_joints(rig):
    this_level = [rig.root_id]
    pos_new = [rig.pos[rig.root_id]]
    hier_new = [-1]
    names_new = [rig.root_name]

    while this_level:
        next_level = []
        for pid in this_level:
            ch_ids = np.argwhere(rig.hierarchy == pid).squeeze(axis=1)
            if len(ch_ids) > 1:
                for dup_id, ch_id in enumerate(ch_ids):
                    # duplicate parent node
                    pos_new.append(rig.pos[pid] + 0.01 * (rig.pos[ch_id] - rig.pos[pid]))
                    names_new.append(rig.names[pid] + f"_dup_{dup_id}")
                    hier_new.append(names_new.index(rig.names[pid]))
                    # add child node
                    pos_new.append(rig.pos[ch_id])
                    names_new.append(rig.names[ch_id])
                    hier_new.append(names_new.index(rig.names[pid] + f"_dup_{dup_id}"))
            elif len(ch_ids) == 1:
                ch_id = ch_ids[0]
                pos_new.append(rig.pos[ch_id])
                names_new.append(rig.names[ch_id])
                hier_new.append(names_new.index(rig.names[pid]))
            else:  # no child
                pass
            next_level += ch_ids.tolist()
        this_level = next_level

    rig_new = Rig()
    rig_new.pos = np.array(pos_new)
    rig_new.hierarchy = np.array(hier_new)
    rig_new.root_id = 0
    rig_new.root_name = rig.root_name
    rig_new.names = names_new
    rig_new.calc_frames_and_offsets()
    return rig_new


def mapping_bone_index(bones_old, bones_new):
    bone_map = {}
    for i in range(len(bones_old)):
        bone_old = bones_old[i][np.newaxis, :]
        dist = np.linalg.norm(bones_new - bone_old, axis=1)
        ni = np.argmin(dist)
        bone_map[i] = ni
    return bone_map


def assemble_skel_skin(skel, attachment):
    bones_old, bone_names_old, _ = get_bones(skel)
    rig_new = add_duplicate_joints(skel)
    bones_new, bone_names_new, _ = get_bones(rig_new)
    bone_mapping = mapping_bone_index(bones_old, bones_new)
    for v in range(len(attachment)):
        skw = attachment[v]
        skw_new = np.zeros(len(rig_new.names))
        for i in range(len(skw)):
            if skw[i] > 1e-5:
                bind_joint_name = bone_names_new[bone_mapping[i]][0]
                bind_weight = skw[i]
                skw_new[rig_new.names.index(bind_joint_name)] = bind_weight
        rig_new.skins.append(skw_new)
    rig_new.skins = np.stack(rig_new.skins, axis=0)
    return rig_new


def post_filter(skin_weights, topology_edge, num_ring=1):
    skin_weights_new = np.zeros_like(skin_weights)
    for v in range(len(skin_weights)):
        adj_verts_multi_ring = []
        current_seeds = [v]
        for r in range(num_ring):
            adj_verts = []
            for seed in current_seeds:
                adj_edges = topology_edge[:, np.argwhere(topology_edge == seed)[:, 1]]
                adj_verts_seed = list(set(adj_edges.flatten().tolist()))
                adj_verts_seed.remove(seed)
                adj_verts += adj_verts_seed
            adj_verts_multi_ring += adj_verts
            current_seeds = adj_verts
        adj_verts_multi_ring = list(set(adj_verts_multi_ring))
        if len(adj_verts_multi_ring) == 0:
            skin_weights_new[v, :] = skin_weights[v]
        else:
            if v in adj_verts_multi_ring:
                adj_verts_multi_ring.remove(v)
            skin_weights_neighbor = [skin_weights[int(i), :][np.newaxis, :] for i in adj_verts_multi_ring]
            skin_weights_neighbor = np.concatenate(skin_weights_neighbor, axis=0)
            #max_bone_id = np.argmax(skin_weights[v, :])
            #if np.sum(skin_weights_neighbor[:, max_bone_id]) < 0.17 * len(skin_weights_neighbor):
            #    skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
            #else:
            #    skin_weights_new[v, :] = skin_weights[v, :]
            skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
    #skin_weights_new[skin_weights_new.sum(axis=1) == 0, :] = skin_weights[skin_weights_new.sum(axis=1) == 0, :]
    return skin_weights_new


def predict_skeleton(input_data, vox, root_pred_net, bone_pred_net):
    """
    Predict skeleton structure based on joints
    :param input_data: wrapped data
    :param vox: voxelized mesh
    :param root_pred_net: network to predict root
    :param bone_pred_net: network to predict pairwise connectivity cost
    :param mesh_filename: meshfilename for debugging
    :return: predicted skeleton structure
    """
    root_id = getInitId(input_data, root_pred_net)
    pred_joints = input_data.joints.data.cpu().numpy()

    with torch.no_grad():
        connect_prob, _ = bone_pred_net(input_data, permute_joints=False)
        connect_prob = torch.sigmoid(connect_prob)
    pair_idx = input_data.pairs.long().data.cpu().numpy()
    prob_matrix = np.zeros((len(input_data.joints), len(input_data.joints)))
    prob_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.data.cpu().numpy().squeeze()
    prob_matrix = prob_matrix + prob_matrix.transpose()
    cost_matrix = -np.log(prob_matrix + 1e-10)
    cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)

    parent, key = primMST(cost_matrix, root_id)
    #parent, key, root_id = primMST_symmetry(cost_matrix, root_id, pred_joints)
    pred_skel = Rig()
    pred_skel.pos = pred_joints
    pred_skel.root_id = root_id
    pred_skel.names = [f'joint_{i}' for i in range(len(pred_joints))]
    pred_skel.root_name = pred_skel.names[root_id]
    pred_skel.hierarchy = parent
    pred_skel.calc_frames_and_offsets()
    return pred_skel #, img


def create_one_data(v, joints, tpl_e, geo_e, vox, motion=None, flow=None):
    # prepare and add new data members
    pairs = list(it.combinations(range(joints.shape[0]), 2))
    pair_attr = []
    for pr in pairs:
        dist = np.linalg.norm(joints[pr[0]] - joints[pr[1]])
        bone_samples = sample_on_bone(joints[pr[0]], joints[pr[1]], step_size=0.01)
        bone_samples_inside, _ = inside_check(bone_samples, vox)
        outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
        attr = np.array([dist, outside_proportion, 1])
        pair_attr.append(attr)
    pairs = np.array(pairs)
    pair_attr = np.array(pair_attr)
    pairs = torch.from_numpy(pairs).float()
    pair_attr = torch.from_numpy(pair_attr).float()
    joints = torch.from_numpy(joints).float()
    joints_batch = torch.zeros(len(joints), dtype=torch.long)
    pairs_batch = torch.zeros(len(pairs), dtype=torch.long)

    v = torch.from_numpy(v).float()
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))
    if motion is not None:
        motion = torch.from_numpy(motion).float()
    if flow is not None:
        flow = torch.from_numpy(flow).float()
    # batch
    batch = torch.zeros(len(v), dtype=torch.long)
    data = Data(pos=v, tpl_edge_index=tpl_e, geo_edge_index=geo_e, batch=batch, joints=joints, motion=motion,
                flow=flow, pairs=pairs, pair_attr=pair_attr, joints_batch=joints_batch, pairs_batch=pairs_batch)
    return data


def pred_skel_func(res_folder):
    mesh_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/obj_remesh/"
    vox_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/vox/"
    testset_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/test/"
    rootNet = ROOTNET()
    rootNet.to(device)
    rootNet.eval()
    rootNet_checkpoint = torch.load('checkpoints/rootnet/model_best.pth.tar')
    rootNet.load_state_dict(rootNet_checkpoint['state_dict'])
    print("     root prediction network loaded.")
    boneNet = BONENET()
    boneNet.to(device)
    boneNet.eval()
    boneNet_checkpoint = torch.load('checkpoints/bonenet/model_best.pth.tar')
    boneNet.load_state_dict(boneNet_checkpoint['state_dict'])
    print("     bone prediction network loaded.")

    joint_filelist = glob.glob(os.path.join(res_folder, "*_joint.npy"))
    for joint_filename in tqdm(joint_filelist):
        model_id = joint_filename.split("/")[-1].split("_")[0]
        if os.path.exists(os.path.join(res_folder, f"{model_id}_skel.txt")):
            continue
        mesh_filename = os.path.join(mesh_folder, f"{model_id}.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        v = np.asarray(mesh.vertices)
        tpl_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_tpl_e.txt")).T
        geo_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_geo_e.txt")).T
        with open(os.path.join(vox_folder, f"{model_id}.binvox"), 'rb') as fvox:
            vox = binvox_rw.read_as_3d_array(fvox)
        joints = np.load(joint_filename)
        #img = draw_shifted_pts(mesh_filename, joints)

        data = create_one_data(v, joints, tpl_e, geo_e, vox)
        data.to(device)
        pred_skeleton = predict_skeleton(data, vox, rootNet, boneNet)
        #show_obj_rig(mesh, pred_skeleton)
        pred_skeleton.save(os.path.join(res_folder, f"{model_id}_skel.txt"))
        #cv2.imwrite(os.path.join(res_folder, f"{model_id}_skel.png"), img[:,:,::-1])


def calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=False):
    """
    calculate volumetric geodesic distance from vertices to each bones
    :param bones: B*6 numpy array where each row stores the starting and ending joint position of a bone
    :param mesh_v: V*3 mesh vertices
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: an approaximate volumetric geodesic distance matrix V*B, were (v,b) is the distance from vertex v to bone b
    """

    if subsampling:
        mesh0 = o3d.io.read_triangle_mesh(mesh_filename)
        mesh0 = mesh0.simplify_quadric_decimation(3000)
        o3d.io.write_triangle_mesh(mesh_filename.replace(".obj", "_simplified.obj"), mesh0)
        mesh_trimesh = trimesh.load(mesh_filename.replace(".obj", "_simplified.obj"))
        subsamples_ids = np.random.choice(len(mesh_v), np.min((len(mesh_v), 1500)), replace=False)
        subsamples = mesh_v[subsamples_ids, :]
        surface_geodesic = surface_geodesic[subsamples_ids, :][:, subsamples_ids]
    else:
        mesh_trimesh = trimesh.load(mesh_filename)
        subsamples = mesh_v
    origins, ends, pts_bone_dist = pts2line(subsamples, bones)
    pts_bone_visibility = calc_pts2bone_visible_mat(mesh_trimesh, origins, ends)
    pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(subsamples)).transpose()
    pts_bone_dist = pts_bone_dist.reshape(len(bones), len(subsamples)).transpose()
    # remove visible points which are too far
    for b in range(pts_bone_visibility.shape[1]):
        visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
        if len(visible_pts) == 0:
            continue
        threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
        pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False

    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                visible_matrix[r, c] = 8.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]
    if subsampling:
        nn_dist = np.sum((mesh_v[:, np.newaxis, :] - subsamples[np.newaxis, ...])**2, axis=2)
        nn_ind = np.argmin(nn_dist, axis=1)
        visible_matrix = visible_matrix[nn_ind, :]
        os.remove(mesh_filename.replace(".obj", "_simplified.obj"))
    return visible_matrix


def remove_dup_joints(rig_ori):
    this_level = [rig_ori.root_id]
    joint_res = []
    skin_res = []
    hier_res = [-1]
    names_res = [rig_ori.root_name]
    while this_level:
        next_level = []
        for p_id in this_level:
            ch_ids = np.argwhere(rig_ori.hierarchy == p_id).squeeze(axis=1)
            for ch_id in ch_ids:
                if "_dup" in rig_ori.names[ch_id]:
                    ch_id_of_ch = np.argwhere(rig_ori.hierarchy == ch_id).squeeze(axis=1)[0]
                    next_level.append(ch_id_of_ch)
                    names_res.append(rig_ori.names[ch_id_of_ch])
                    rig_ori.skins[:, p_id] += rig_ori.skins[:, ch_id]
                else:
                    next_level.append(ch_id)
                    names_res.append(rig_ori.names[ch_id])
                hier_res.append(names_res.index(rig_ori.names[p_id]))
            joint_res.append(rig_ori.pos[p_id])
            skin_res.append(rig_ori.skins[:, p_id])
        this_level = next_level
    rig_res = Rig()
    rig_res.pos = np.stack(joint_res, axis=0)
    rig_res.hierarchy = np.array(hier_res)
    rig_res.names = names_res
    rig_res.root_id = 0
    rig_res.root_name = rig_ori.root_name
    rig_res.calc_frames_and_offsets()
    rig_res.skins = np.stack(skin_res, axis=1)
    return rig_res
    

def predict_skinning(input_data, pred_skel, skin_pred_net, surface_geodesic, mesh_filename, subsampling=False):
    """
    predict skinning
    :param input_data: wrapped input data
    :param pred_skel: predicted skeleton
    :param skin_pred_net: network to predict skinning weights
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: predicted rig with skinning weights information
    """
    global device
    num_nearest_bone = 5
    bones, bone_names, bone_isleaf = get_bones(pred_skel)
    mesh_v = input_data.pos.data.cpu().numpy()
    print("     calculating volumetric geodesic distance from vertices to bone. This step takes some time...")
    geo_dist = calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=subsampling)
    input_samples = []  # joint_pos (x, y, z), (bone_id, 1/D)*5
    loss_mask = []
    skin_nn = []
    for v_id in range(len(mesh_v)):
        geo_dist_v = geo_dist[v_id]
        bone_id_near_to_far = np.argsort(geo_dist_v)
        this_sample = []
        this_nn = []
        this_mask = []
        for i in range(num_nearest_bone):
            if i >= len(bones):
                this_sample += bones[bone_id_near_to_far[0]].tolist()
                this_sample.append(1.0 / (geo_dist_v[bone_id_near_to_far[0]] + 1e-10))
                this_sample.append(bone_isleaf[bone_id_near_to_far[0]])
                this_nn.append(0)
                this_mask.append(0)
            else:
                skel_bone_id = bone_id_near_to_far[i]
                this_sample += bones[skel_bone_id].tolist()
                this_sample.append(1.0 / (geo_dist_v[skel_bone_id] + 1e-10))
                this_sample.append(bone_isleaf[skel_bone_id])
                this_nn.append(skel_bone_id)
                this_mask.append(1)
        input_samples.append(np.array(this_sample)[np.newaxis, :])
        skin_nn.append(np.array(this_nn)[np.newaxis, :])
        loss_mask.append(np.array(this_mask)[np.newaxis, :])

    skin_input = np.concatenate(input_samples, axis=0)
    loss_mask = np.concatenate(loss_mask, axis=0)
    skin_nn = np.concatenate(skin_nn, axis=0)
    skin_input = torch.from_numpy(skin_input).float()
    input_data.skin_input = skin_input
    input_data.to(device)

    motion_all, motion_aggr, skin_pred = skin_pred_net(input_data, input_data.flow)
    skin_pred = skin_pred * torch.from_numpy(loss_mask).to(skin_pred.device)
    skin_pred = torch.softmax(skin_pred, dim=1)
    skin_pred = skin_pred.data.cpu().numpy()
    motion_aggr = motion_aggr.detach().cpu().numpy()
    skin_nn = skin_nn[:, 0:num_nearest_bone]
    skin_pred_full = np.zeros((len(skin_pred), len(bone_names)))
    for v in range(len(skin_pred)):
        for nn_id in range(len(skin_nn[v, :])):
            if loss_mask[v, nn_id] == 1:
                skin_pred_full[v, skin_nn[v, nn_id]] = skin_pred[v, nn_id]
    print("     filtering skinning prediction")
    tpl_e = input_data.tpl_edge_index.data.cpu().numpy()
    skin_pred_full = post_filter(skin_pred_full, tpl_e, num_ring=1)
    skin_pred_full[skin_pred_full < np.max(skin_pred_full, axis=1, keepdims=True) * 0.35] = 0.0
    skin_pred_full = skin_pred_full / (skin_pred_full.sum(axis=1, keepdims=True) + 1e-10)
    skel_res = assemble_skel_skin(pred_skel, skin_pred_full)
    return skel_res


def pred_rig_func(res_folder):
    mesh_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/obj_remesh/"
    vox_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/vox/"
    testset_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/test/"
    surface_geodesic_folder = "/mnt/DATA_LINUX2/zhan/morig/ModelsResources/surface_geodesic/"

    skinNet = SkinMotion(nearest_bone=5, use_motion=True, use_Dg=False, use_Lf=False, motion_dim=32, num_keyframes=5)
    skinNet_checkpoint = torch.load('checkpoints/skin_motion/model_best.pth.tar')
    skinNet.load_state_dict(skinNet_checkpoint['state_dict'])
    skinNet.to(device)
    skinNet.eval()
    print("     skinning prediction network loaded.")

    skel_filelist = glob.glob(os.path.join(res_folder, "*_skel.txt"))
    for skel_filename in tqdm(skel_filelist):
        model_id = skel_filename.split("/")[-1].split("_")[0]
        if os.path.exists(os.path.join(res_folder, f"{model_id}_rig.txt")):
            continue
        mesh_filename = os.path.join(mesh_folder, f"{model_id}.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        mesh.compute_vertex_normals()
        pred_skeleton = Rig(skel_filename)
        #show_obj_rig(mesh, pred_skeleton)
        v = np.asarray(mesh.vertices)
        tpl_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_tpl_e.txt")).T
        geo_e = np.loadtxt(os.path.join(testset_folder, f"{model_id}_geo_e.txt")).T
        with open(os.path.join(vox_folder, f"{model_id}.binvox"), 'rb') as fvox:
            vox = binvox_rw.read_as_3d_array(fvox)
        joints = np.load(skel_filename.replace("_skel.txt", "_joint.npy"))
        pred_flow = []
        for t in range(1, 6):
            pred_flow.append(np.load(os.path.join(testset_folder, f"pred_flow/{model_id}_{t}_pred_flow.npy")))
        pred_flow = np.concatenate(pred_flow, axis=1)
        data = create_one_data(v, joints, tpl_e, geo_e, vox, flow=pred_flow)
        data.to(device)
        if os.path.exists(os.path.join(surface_geodesic_folder, f"{model_id}.npy")):
            surface_geodesic = np.load(os.path.join(surface_geodesic_folder, f"{model_id}.npy"))
        else:
            surface_geodesic = calc_surface_geodesic(mesh, number_of_points=4000)
            np.save(os.path.join(surface_geodesic_folder, f"{model_id}.npy"), surface_geodesic)
        pred_rig = predict_skinning(data, pred_skeleton, skinNet, surface_geodesic, mesh_filename, subsampling=True)
        # remove duplicate joints
        pred_rig = remove_dup_joints(pred_rig)
        pred_rig.save(os.path.join(res_folder, f"{model_id}_rig.txt"))


if __name__ == "__main__":
    res_folder_ours = "results/our_results/"
    #pred_skel_func(res_folder_ours)  # step 1
    pred_rig_func(res_folder_ours)  # step 2
