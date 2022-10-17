import numpy as np
import open3d as o3d
import time
import copy
import os
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from utils.rot_utils import mat2continuous6d
from scipy.ndimage import binary_dilation
from utils.mst_utils import sample_on_bone
import multiprocessing as mp
from scipy.spatial import KDTree


def get_tpl_edges(obj_v, obj_f):
    edge_index = []
    for v in range(len(obj_v)):
        face_ids = np.argwhere(obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if obj_f[face_id, v_id] != v:
                    neighbor_ids.append(obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        if len(neighbor_ids) > 0:
            neighbor_ids = np.concatenate(neighbor_ids, axis=0)
            edge_index.append(neighbor_ids)
        else:
            continue
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def find_tpl_neighbors(mesh_0):
    vtx = np.asarray(mesh_0.vertices)
    faces = np.asarray(mesh_0.triangles)
    neighbors_all = []
    for v in range(len(vtx)):
        face_ids = np.argwhere(faces == v)[:, 0]
        neighbor_ids = np.unique(faces[face_ids].flatten())
        neighbor_ids = np.delete(neighbor_ids, np.where(neighbor_ids == v))
        neighbors_all.append(neighbor_ids)
    return neighbors_all


def get_gt_rotation_icp(mesh_0, vtx_0, vtx_t, nnids):
    if nnids is None:
        # form multi ring neighbors
        vtx_distmat = np.sqrt(np.sum((vtx_0[:, None, :] - vtx_0[None, ...]) ** 2, axis=-1))
        nnids = find_tpl_neighbors(mesh_0)
        for ring in range(2):
            nnids_aug = []
            for v in range(len(vtx_0)):
                nn_aug = np.unique(np.concatenate([nnids[i] for i in nnids[v]]))
                nnids_aug.append(nn_aug)
            nnids = nnids_aug
        for vid in range(len(vtx_0)):
            dist_v = vtx_distmat[vid, nnids[vid]]
            thd = 0.04
            while len(np.argwhere(dist_v < thd).squeeze(axis=1)) < 5:
                thd *= 1.25
                if thd > 0.06:
                    break
            nnids[vid] = nnids[vid][np.argwhere(dist_v < thd).squeeze(axis=1)]

    R_all = []
    T_all = []
    for vid in range(len(vtx_0)):
        patches_0 = vtx_0[nnids[vid]]
        patches_t = vtx_t[nnids[vid]]
        R, t = icp(patches_0[None, ...], patches_t[None, ...])
        R_all.append(mat2continuous6d(R))
        #R_all.append(R.squeeze())
        T_all.append(t.squeeze())
    R_all = np.concatenate(R_all, axis=0)
    T_all = np.stack(T_all, axis=0)
    return R_all, T_all, nnids


def generate_3d():
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M


def batch_fps(batch_pts, K):
    """ Found here:
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    """
    calc_distances = lambda p0, pts: ((p0 - pts) ** 2).sum(axis=1)

    def fps(x):
        pts, K = x
        farthest_idx = np.zeros(K, dtype=np.int)
        farthest_idx[0] = np.random.randint(len(pts))
        distances = calc_distances(pts[farthest_idx[0]], pts)

        for i in range(1, K):
            farthest_idx[i] = np.argmax(distances, axis=0)
            farthest_pts = pts[farthest_idx[i]]
            distances = np.minimum(distances, calc_distances(farthest_pts, pts))
        pts_sampled = pts[farthest_idx, :]
        return pts_sampled, farthest_idx

    fps_res = list(map(fps, [(pts, K) for pts in batch_pts]))
    batch_pts = [i[0] for i in fps_res]
    batch_id = [i[1] for i in fps_res]
    return np.stack(batch_pts, axis=0), np.stack(batch_id, axis=0)


def normalize(mesh, pivot=None, scale=None):
    vtx = np.asarray(mesh.vertices)
    if scale is None:
        dims = [max(vtx[:, 0]) - min(vtx[:, 0]),
                max(vtx[:, 1]) - min(vtx[:, 1]),
                max(vtx[:, 2]) - min(vtx[:, 2])]
        scale = 1.0 / max(dims)
    if pivot is None:
        pivot = np.array([(min(vtx[:, 0]) + max(vtx[:, 0])) / 2, min(vtx[:, 1]),
                          (min(vtx[:, 2]) + max(vtx[:, 2])) / 2])
    vtx[:, 0] -= pivot[0]
    vtx[:, 1] -= pivot[1]
    vtx[:, 2] -= pivot[2]
    vtx *= scale
    mesh.vertices = o3d.utility.Vector3dVector(vtx)
    return mesh, pivot, scale


def random_small_rotate():
    alpha, beta, gamma = np.random.uniform(low=-0.1, high=0.1, size=3)
    R_x = lambda x: np.array([[1, 0, 0], [0, np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)],
                              [0, -np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)]])
    R_y = lambda x: np.array([[np.cos(2 * np.pi * x), 0, -np.sin(2 * np.pi * x)], [0, 1, 0],
                              [np.sin(2 * np.pi * x), 0, np.cos(2 * np.pi * x)]])
    R_z = lambda x: np.array([[np.cos(2 * np.pi * x), np.sin(2 * np.pi * x), 0],
                              [-np.sin(2 * np.pi * x), np.cos(2 * np.pi * x), 0], [0, 0, 1]])
    R = lambda x, y, z: np.matmul(np.matmul(R_x(x), R_y(y)), R_z(z))

    rot_mat = R(alpha, beta, gamma)
    return rot_mat


def icp(src_pts, tar_pts):
    """
    Iterative closest point method (https://en.wikipedia.org/wiki/Iterative_closest_point)
    Used to find the best rotation and translation to align two sets of point cloud
    src_pts: B, N, 3
    tar_pts: B, N, 3
    """
    sptb = src_pts - np.mean(src_pts, axis=1, keepdims=True)
    tptb = tar_pts - np.mean(tar_pts, axis=1, keepdims=True)
    M = np.matmul(tptb.transpose(0, 2, 1), sptb)
    U, s, Vh = np.linalg.svd(M)
    R = np.matmul(U, Vh)
    if len(np.argwhere(np.linalg.det(R) < 0)) > 0:
        invalid_ids = np.argwhere(np.linalg.det(R) < 0).squeeze(axis=1)
        Vh[invalid_ids, -1, :] *= -1
        R[invalid_ids] = np.matmul(U[invalid_ids], Vh[invalid_ids])
    t = np.mean(tar_pts - np.matmul(src_pts, R.transpose(0, 2, 1)), axis=1, keepdims=True)
    return R, t


def calc_surface_geodesic(mesh, number_of_points=4000):
    # We denselu sample 4000 points to be more accuracy.
    samples = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    samples.estimate_normals()
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    time1 = time.time()
    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                    return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    time2 = time.time()
    #print('surface geodesic calculation: {} seconds'.format((time2 - time1)))
    return surface_geodesic


def get_geo_edges(remeshed_obj, radius=0.06, max_nn=15):
    remesh_obj_v = np.asarray(remeshed_obj.vertices)
    surface_geodesic = calc_surface_geodesic(remeshed_obj)
    edge_index = []
    surface_geodesic += 10.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= radius).squeeze(1)
        if len(geodesic_ball_samples) > max_nn:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, max_nn, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def transfer_rig_to_remesh(mesh_ori, mesh_remesh, rig_ori):
    vtx_ori = np.asarray(mesh_ori.vertices)
    vtx_remesh = np.asarray(mesh_remesh.vertices)
    dist = np.sqrt(np.sum((vtx_remesh[:, None, :] - vtx_ori[None, ...]) ** 2, axis=-1))
    skin_new = np.zeros((len(vtx_remesh), rig_ori.skins.shape[1]))
    filled_flag = np.zeros(len(vtx_remesh), dtype=bool)
    overlap_ids = np.argwhere(np.min(dist, axis=1) == 0).squeeze(axis=1)
    filled_flag[overlap_ids] = True
    skin_new[overlap_ids] = rig_ori.skins[np.argmin(dist[overlap_ids], axis=1)]
    faces = np.asarray(mesh_remesh.triangles)
    dist2 = np.sqrt(np.sum((vtx_remesh[:, None, :] - vtx_remesh[None, ...]) ** 2, axis=-1))
    while not np.all(filled_flag):
        unfilled_pos = np.argwhere(filled_flag == False).squeeze(axis=1)
        for v in unfilled_pos:
            neighbor_ids = np.unique(faces[np.argwhere(faces == v)])
            neighbor_ids = np.delete(neighbor_ids, np.where(neighbor_ids == v))
            if filled_flag[neighbor_ids].sum() > 0:
                closest_nn = -1
                closest_dist = 1e8
                for nv in neighbor_ids:
                    if filled_flag[nv] and dist2[v, nv] < closest_dist:
                        closest_dist = dist2[v, nv]
                        closest_nn = nv
                skin_new[v] = skin_new[closest_nn]
                filled_flag[v] = True

    skin_new = skin_new / (skin_new.sum(axis=1, keepdims=True) + 1e-8)
    rig_remesh = copy.deepcopy(rig_ori)
    rig_remesh.skins = skin_new
    #rig_remesh.save(os.path.join(info_remesh_folder, f"{model_id}.txt"))
    return rig_remesh


def pts2lines(pts, lines):
    """
    pts: Nx3
    lines: Mx6
    """
    l2 = np.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, axis=1)
    t = np.sum((pts[:, None, :] - lines[None, :, 0:3]) * (lines[:, 3:6] - lines[:, 0:3])[None, ...], axis=-1) / (l2[None, :] + 1e-6)
    t = np.clip(t, 0, 1)
    t_pos = lines[:, 0:3][None, :] + t[..., None] * (lines[:, 3:6] - lines[:, 0:3])[None, ...]
    dist = np.sqrt(np.sum((t_pos - pts[:, None, :])**2, axis=-1))
    return dist


def one_bone(vtx_vox, vox, bone):
    kernel = np.ones((3, 3, 3))
    vox_reached = np.zeros_like(vox.data, dtype=bool)
    distmap_bone = np.zeros_like(vox.data, dtype=np.int64)
    bone_samples = sample_on_bone(bone[0:3], bone[3:6])[:-1, :]
    bone_samples = np.vstack([bone_samples, bone[0:3]])

    bone_sample_vox = (bone_samples - vox.translate) / vox.scale * vox.dims[0]
    bone_sample_vox = np.round(bone_sample_vox).astype(int)
    bone_sample_vox = np.clip(bone_sample_vox, 0, 87)

    vox_reached[bone_sample_vox[:, 0], bone_sample_vox[:, 1], bone_sample_vox[:, 2]] = True
    dist_bone = 1
    num_unfilled_last = (vox.data * (vox_reached == False)).sum()
    while num_unfilled_last > 0:
        #print((vox.data * (vox_reached == False)).sum())
        vox_reached_new = binary_dilation(vox_reached, kernel, mask=vox.data)
        vox_changed = np.argwhere(vox_reached_new != vox_reached)
        distmap_bone[vox_changed[:, 0], vox_changed[:, 1], vox_changed[:, 2]] = dist_bone
        dist_bone += 1
        vox_reached = vox_reached_new

        num_unfilled_this = (vox.data * (vox_reached == False)).sum()
        if num_unfilled_this == num_unfilled_last:
            # address disconnected components.
            vox_unreached = vox.data * (vox_reached == False)
            pos_unreached = np.argwhere(vox_unreached)
            pos_reached = np.argwhere(vox_reached)
            kdtree = KDTree(pos_reached)
            nndist, nnids = kdtree.query(pos_unreached)
            closest_ids = np.where(nndist == np.min(nndist))[0]
            for i in range(len(closest_ids)):
                unreached_i = closest_ids[i]
                reached_i = nnids[closest_ids[i]]
                pos_xx, pos_yy = pos_unreached[unreached_i], pos_reached[reached_i]
                distmap_bone[pos_xx[0], pos_xx[1], pos_xx[2]] = distmap_bone[pos_yy[0], pos_yy[1], pos_yy[2]] + 1
                vox_reached[pos_xx[0], pos_xx[1], pos_xx[2]] = True
        num_unfilled_last = num_unfilled_this

    return distmap_bone[vtx_vox[:, 0], vtx_vox[:, 1], vtx_vox[:, 2]]


def calc_volumetric_geodesic(vtx, vox, bones):
    ## bones: Mx6
    #time0 = time.time()
    vtx_vox = (vtx - vox.translate) / vox.scale * vox.dims[0]
    vtx_vox = np.round(vtx_vox).astype(int)
    vtx_vox = np.clip(vtx_vox, 0, 87)
    pool = mp.Pool(8)  # mp.cpu_count()
    results = pool.starmap(one_bone, [(vtx_vox, vox, bones[i]) for i in range(len(bones))])
    results = np.stack(results, axis=1)
    #time1 = time.time()
    #print(time1 - time0)
    return results


def get_obb_for_parts(vtx, seg, num_parts, minimal_num_vtx=6):
    obb_pts_all = -np.ones((num_parts, 8, 3))
    for seg_id in range(num_parts):
        vid = np.argwhere(seg == seg_id).squeeze(axis=1)
        if len(vid) > minimal_num_vtx:
            pts_seg = vtx[vid] + 1e-5 * np.random.randn(vtx[vid].shape[0], vtx[vid].shape[1])  # avoid pts all in lower dimension
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(pts_seg))
            obb_pts_all[seg_id] = np.asarray(obb.get_box_points())
        else:
            continue
        # # visualize
        # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vtx))
        # pcd_color = np.zeros((len(vtx), 3))
        # pcd_color[vid] = np.array([1.0, 0.0, 0.0])
        # pcd.colors = o3d.utility.Vector3dVector(pcd_color)
        # o3d.visualization.draw_geometries([pcd, obb])
    return obb_pts_all