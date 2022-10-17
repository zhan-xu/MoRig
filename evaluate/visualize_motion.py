import glob, os, shutil, open3d as o3d, cv2, copy
from utils.rig_parser import Rig
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy import linalg
from utils.vis_utils import visualize_seg


def spectral_clustering_numpy(A, d=-1, cut_thres=0.993):
    '''
    input similarity matrix A: N,N
    return pred segmentation: N,S
    '''
    D = np.diag(np.sum(A, axis=0))
    L = D - A
    w, v = linalg.eig(L)
    w = w.astype(np.float64)
    v = v.astype(np.float64)

    if d != -1:
        w_leading = np.abs(w[1:48])
        total_est_points = np.mean(w_leading)
        e_th = total_est_points * 0.993
        d = np.clip(np.sum(np.abs(w) < e_th), 1, 48)

    _, midx = kmeans2(v[:, np.argsort(np.abs(w))[1:d + 1]], d, iter=2000)
    return midx


mesh_folder = "/media/zhanxu/4T/motionAnim_results/syn_res/"
res_folder = "/media/zhanxu/4T/motionAnim_results/syn_res/"
output_folder = "/media/zhanxu/4T/motionAnim_results/syn_res/"
filelist = glob.glob(os.path.join(res_folder, "*_embedding.npy"))
tsne = TSNE(n_components=3, perplexity=30.0, random_state=1)


# teaser
# for mesh_filename in glob.glob(os.path.join(mesh_folder, "*_remesh.obj")):
#     mesh_filename = os.path.join(mesh_folder, "10559_remesh.obj")
#     model_id = mesh_filename.split("/")[-1].split(".")[0]
#     mesh = o3d.io.read_triangle_mesh(mesh_filename)
#     vtx = np.asarray(mesh.vertices)
#     embedding = np.load(mesh_filename.replace("_remesh.obj", "_embedding.npy"))
#
#     mesh.compute_adjacency_list()
#     vtx = np.asarray(mesh.vertices)
#     for t in range(2):
#         embedding_new = copy.deepcopy(embedding)
#         for vid in range(len(vtx)):
#             if len(list(mesh.adjacency_list[vid])) > 0:
#                 embedding_new[vid] = np.mean(embedding[list(mesh.adjacency_list[vid])], axis=0)
#         embedding = embedding_new
#
#     embedding_z = tsne.fit_transform(embedding)
#     embedding_z = (embedding_z - np.min(embedding_z, axis=0, keepdims=True)) / (np.max(embedding_z, axis=0, keepdims=True) - np.min(embedding_z, axis=0, keepdims=True))
#     np.save(os.path.join(output_folder, f"{model_id}_motion_colormap.npy"), embedding_z)
#     mesh.vertex_colors = o3d.utility.Vector3dVector(embedding_z)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#
#     vis.add_geometry(mesh)
#     vis.run()
#     #vis.capture_screen_image(os.path.join(output_folder, f"{model_id}_motion.png"))
#     vis.destroy_window()
#     exit()

mesh_filelist = ["/media/zhanxu/4T/data/killingFusion/Duck_convert/mesh_simplified.obj",
                 "/media/zhanxu/4T/data/killingFusion/Snoopy_convert/mesh_simplified.obj",
                 "/media/zhanxu/4T/data/Dynamic_FAUST/human_mesh/6387.obj",
                 "/media/zhanxu/4T/data/Dynamic_FAUST/human_mesh/6388.obj"]

motion_filelist = ["../DFaust/duck_motion.npy",
                   "../DFaust/snoopy_motion.npy",
                   "../DFaust/6387_motion.npy",
                   "../DFaust/6388_motion.npy",]

for i in range(len(mesh_filelist)):
    mesh_filename = mesh_filelist[i]
    motion_filename = motion_filelist[i]
    model_id = motion_filename.split("/")[-1].split("_")[0]

    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    vtx = np.asarray(mesh.vertices)
    embedding = np.load(motion_filename)
    mesh.compute_adjacency_list()
    vtx = np.asarray(mesh.vertices)
    for t in range(2):
        embedding_new = copy.deepcopy(embedding)
        for vid in range(len(vtx)):
            if len(list(mesh.adjacency_list[vid])) > 0:
                embedding_new[vid] = np.mean(embedding[list(mesh.adjacency_list[vid])], axis=0)
        embedding = embedding_new
    embedding_z = tsne.fit_transform(embedding)
    embedding_z = (embedding_z - np.min(embedding_z, axis=0, keepdims=True)) / (np.max(embedding_z, axis=0, keepdims=True) - np.min(embedding_z, axis=0, keepdims=True))
    np.save(f"{model_id}_motion_colormap.npy", embedding_z)
    mesh.vertex_colors = o3d.utility.Vector3dVector(embedding_z)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(mesh)
    vis.run()
    #vis.capture_screen_image(os.path.join(output_folder, f"{model_id}_motion.png"))
    vis.destroy_window()
exit()

for model_filename in filelist:
    model_id = model_filename.split("/")[-1].split("_")[0]
    mesh_filename = os.path.join(mesh_folder, f"{model_id}.obj")
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    vtx = np.asarray(mesh.vertices)
    embedding = np.load(model_filename)

    # rig = Rig(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/rig_info_remesh/{model_id}.txt")
    # gt_seg = np.argmax(rig.skins, axis=1)
    # print("GT seg:", len(np.unique(gt_seg)))
    # sim_mat = (np.matmul(embedding, embedding.T) + 1.0) / 2.0
    # pred_seg = spectral_clustering_numpy(sim_mat, d=20)
    # print("Pred seg:", len(np.unique(pred_seg)))
    # #pred_seg = KernelKMeans(n_clusters=20, w_euc=0.0).fit_predict(embedding, vtx)
    # visualize_seg(vtx, pred_seg)
    # continue

    embedding_z = tsne.fit_transform(embedding)
    embedding_z = (embedding_z - np.min(embedding_z, axis=0, keepdims=True)) / (np.max(embedding_z, axis=0, keepdims=True) - np.min(embedding_z, axis=0, keepdims=True))
    mesh.vertex_colors = o3d.utility.Vector3dVector(embedding_z)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(mesh)
    vis.run()
    vis.capture_screen_image(os.path.join(output_folder, f"{model_id}_motion.png"))
    vis.destroy_window()
