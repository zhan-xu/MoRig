import sys
sys.path.append("./")
import glob, os
import open3d as o3d
import numpy as np
from tqdm import tqdm
from utils import binvox_rw
from utils.rig_parser import Rig
from utils.vis_utils import visualize_seg_skel
from utils.os_utils import mkdir_p
from data_proc.common_ops import calc_volumetric_geodesic


def get_bones(rig):
    bones = []
    bone_name = []
    leaf_bones = []
    this_level = [rig.root_id]
    while this_level:
        next_level = []
        for pid in this_level:
            ch_ids = np.argwhere(rig.hierarchy == pid).squeeze(axis=1)
            p_pos = rig.pos[pid]
            for cid in ch_ids:
                c_pos = rig.pos[cid]
                bones.append(np.concatenate((p_pos, c_pos))[np.newaxis, :])
                bone_name.append([rig.names[pid], rig.names[cid]])
                leaf_bones.append(False)
                if len(np.argwhere(rig.hierarchy == cid).squeeze(axis=1)) == 0:
                    bones.append(np.concatenate((c_pos, c_pos))[np.newaxis, :])
                    bone_name.append([rig.names[cid], rig.names[cid] + "_leaf"])
                    leaf_bones.append(True)
            next_level += ch_ids.tolist()

        this_level = next_level
    bones = np.concatenate(bones, axis=0)
    return bones, bone_name, leaf_bones


if __name__ == "__main__":
    dataset_folder = "/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/"
    mesh_folder = os.path.join(dataset_folder, "obj_remesh/")
    rig_folder = os.path.join(dataset_folder, "rig_info_remesh/")
    volumetric_geodesic_folder = os.path.join(dataset_folder, "volumetric_geodesic/")
    train_list = np.loadtxt(os.path.join(dataset_folder, "train_final.txt"))
    val_list = np.loadtxt(os.path.join(dataset_folder, "val_final.txt"))
    test_list = np.loadtxt(os.path.join(dataset_folder, "test_final.txt"))
    obj_filelist = glob.glob(os.path.join(mesh_folder, "*.obj"))
    for obj_filename in tqdm(obj_filelist):
        model_id = obj_filename.split("/")[-1].split(".")[0]
        if int(model_id) in train_list:
            split_name = "train"
        elif int(model_id) in val_list:
            split_name = "val"
        else:
            split_name = "test"
        mesh = o3d.io.read_triangle_mesh(obj_filename)
        vtx = np.asarray(mesh.vertices)
        rig_filename = os.path.join(rig_folder, f"{model_id}.txt")
        rig = Rig(rig_filename)
        bones, bone_names, bone_isleaf = get_bones(rig)
        vol_geodesic_dist = np.load(os.path.join(volumetric_geodesic_folder, f"{model_id}_volumetric_geo.npy"))
        
        # gt_skinning_full = np.zeros((len(vtx), len(bones)))
        # for vid in range(len(vtx)):
        #     gt_skin = rig.skins[vid]
        #     skin_w = {}
        #     for i in np.arange(len(gt_skin)):
        #         if gt_skin[i] > 0:
        #             skin_w[rig.names[i]] = float(gt_skin[i])
        #     bone_id_near_to_far = np.argsort(vol_geodesic_dist[vid, :])
        #     for i in range(len(bone_id_near_to_far)):
        #         if i >= len(bone_id_near_to_far):
        #             gt_skinning_full[vid, i] = 0.0
        #         else:
        #             bone_id = bone_id_near_to_far[i]
        #             start_joint_name = bone_names[bone_id][0]
        #             if start_joint_name in skin_w:
        #                 gt_skinning_full[vid, bone_id] = skin_w[start_joint_name]
        #                 del skin_w[start_joint_name]
        #             else:
        #                 gt_skinning_full[vid, bone_id] = 0.0
        # np.save(f"/home/zhan/Proj/motion_anim_v3/results/test/skinning_comparison/{model_id}_skinning_gt.npy", gt_skinning_full)
        # continue
        

        # skinning information
        num_nearest_bone = 20
        input_samples = []  # mesh_vertex_id, (bone_id, 1 / D_g, is_leaf) * N
        ground_truth_labels = []  # w_1, w_2, ..., w_N
        for vid in range(len(vtx)):
            this_sample = [vid]
            this_label = []
            gt_skin = rig.skins[vid]
            
            skin_w = {}
            for i in np.arange(len(gt_skin)):
                if gt_skin[i] > 0:
                    skin_w[rig.names[i]] = float(gt_skin[i])
            
            bone_id_near_to_far = np.argsort(vol_geodesic_dist[vid, :])
            for i in range(num_nearest_bone):
                if i >= len(bone_id_near_to_far):
                    this_sample += [-1, 0, 0]
                    this_label.append(0.0)
                    continue
                bone_id = bone_id_near_to_far[i]
                this_sample.append(bone_id)
                this_sample.append(1.0 / (vol_geodesic_dist[vid, bone_id] + 1e-10))
                this_sample.append(bone_isleaf[bone_id])
                start_joint_name = bone_names[bone_id][0]
                if start_joint_name in skin_w:
                    this_label.append(skin_w[start_joint_name])
                    del skin_w[start_joint_name]
                else:
                    this_label.append(0.0)
            input_samples.append(this_sample)
            ground_truth_labels.append(this_label)

        with open(os.path.join(dataset_folder, '{:s}/{:s}_skin.txt'.format(split_name, model_id)), 'w') as fout:
            for i in range(len(bones)):
                fout.write('bones {:s} {:s} {:.6f} {:.6f} {:.6f} '
                           '{:.6f} {:.6f} {:.6f}\n'.format(bone_names[i][0], bone_names[i][1],
                                                           bones[i, 0], bones[i, 1], bones[i, 2],
                                                           bones[i, 3], bones[i, 4], bones[i, 5]))
            for i in range(len(input_samples)):
                fout.write('bind {:d} '.format(input_samples[i][0]))
                for j in np.arange(1, len(input_samples[i]), 3):
                    fout.write('{:d} {:.6f} {:d} '.format(input_samples[i][j], input_samples[i][j + 1],
                                                          input_samples[i][j + 2]))
                fout.write('\n')
            for i in range(len(ground_truth_labels)):
                fout.write('influence ')
                for j in range(len(ground_truth_labels[i])):
                    fout.write('{:.3f} '.format(ground_truth_labels[i][j]))
                fout.write('\n')