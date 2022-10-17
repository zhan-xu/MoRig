import os
import torch
import numpy as np
import glob
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops
from utils.rig_parser import Rig


class RigDataset(InMemoryDataset):
    def __init__(self, root):
        super(RigDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_v_filelist = glob.glob(os.path.join(self.root, '*_vtx_traj.npy'))
        return raw_v_filelist

    @property
    def processed_file_names(self):
        return '{:s}_rig_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass
    
    def load_skin(self, filename):
        with open(filename, 'r') as fin:
            lines = fin.readlines()
        bones = []
        bone_names = []
        input = []
        label = []
        nearest_bone_ids = []
        loss_mask_all = []
        for li in lines:
            words = li.strip().split()
            if words[0] == 'bones':
                bone_names.append([words[1], words[2]])
                bones.append([float(w) for w in words[3:]])
            elif words[0] == 'bind':
                words = [float(w) for w in words[1:]]
                sample_input = []
                sample_nearest_bone_ids = []
                loss_mask = []
                for i in range(self.num_nearest_bone):
                    if int(words[3 * i + 1]) == -1:
                        ## walk-round. however words[3] may also be invalid.
                        sample_nearest_bone_ids.append(int(words[1]))
                        sample_input += bones[int(words[1])]
                        sample_input.append(words[2])
                        sample_input.append(int(words[3]))
                        loss_mask.append(0)
                    else:
                        sample_nearest_bone_ids.append(int(words[3 * i + 1]))
                        sample_input += bones[int(words[3 * i + 1])]
                        sample_input.append(words[3 * i + 2])
                        sample_input.append(int(words[3 * i + 3]))
                        loss_mask.append(1)
                input.append(np.array(sample_input)[np.newaxis, :])
                nearest_bone_ids.append(np.array(sample_nearest_bone_ids)[np.newaxis, :])
                loss_mask_all.append(np.array(loss_mask)[np.newaxis, :])
            elif words[0] == 'influence':
                sample_label = np.array([float(w) for w in words[1:]])[np.newaxis, :]
                label.append(sample_label)

        input = np.concatenate(input, axis=0)
        nearest_bone_ids = np.concatenate(nearest_bone_ids, axis=0)
        label = np.concatenate(label, axis=0)
        loss_mask_all = np.concatenate(loss_mask_all, axis=0)

        return input, nearest_bone_ids, label, loss_mask_all, bone_names
    
    def process(self):
        data_list = []
        self.num_nearest_bone = 20
        num_max_joint = 48
        for vtx_filename in tqdm(self.raw_paths):
            v_traj = np.load(vtx_filename)
            m = np.loadtxt(vtx_filename.replace('_vtx_traj.npy', '_attn.txt'))
            tpl_e = np.loadtxt(vtx_filename.replace('_vtx_traj.npy', '_tpl_e.txt')).T
            geo_e = np.loadtxt(vtx_filename.replace('_vtx_traj.npy', '_geo_e.txt')).T
            rig = Rig(vtx_filename.replace('_vtx_traj.npy', '_rig.txt'))
            joints = rig.pos
            name = int(vtx_filename.split('/')[-1].split('_')[0])
            nearest_jid = np.argmin(np.sum((joints[:, None, :] - v_traj[:, 0, :][None, ...])**2, axis=-1), axis=0)
            offsets = joints[nearest_jid] - v_traj[:, 0, :]
            gt_skin = np.zeros((rig.skins.shape[0], num_max_joint))
            gt_skin[:, 0:rig.skins.shape[1]] = rig.skins
            skin_input, skin_nn, skin_label, loss_mask, bone_names = self.load_skin(vtx_filename.replace('_vtx_traj.npy', '_skin.txt'))
            # get nearest joint IDs for alignment
            skin_nnjids = []
            for vid in range(len(v_traj)):
                skin_nnjids_v = []
                for n in skin_nn[vid]:
                    skin_nnjids_v.append(rig.names.index(bone_names[n][0]))
                skin_nnjids.append(np.array(skin_nnjids_v))
            skin_nnjids = np.stack(skin_nnjids, 0)
            
            # gt flow
            gt_flow = []
            for key_t in np.arange(20, 110, 20):
                gt_flow.append(v_traj[:, key_t, :] - v_traj[:, 0, :])
            gt_flow = np.concatenate(gt_flow, axis=1)

            # pred flow: you need to first train a deformation module to get them ;)
            pred_flow = []
            for key_t in np.arange(1, 6):
                pred_flow_t = np.load(os.path.join(self.root, f"pred_flow/{name}_{key_t}_pred_flow.npy"))
                pred_flow.append(pred_flow_t)
            pred_flow = np.concatenate(pred_flow, axis=1)

            pos = torch.from_numpy(v_traj[:, 0, :]).float()
            m = torch.from_numpy(m).float()
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=pos.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=pos.size(0))
            offsets = torch.from_numpy(offsets).float()
            gt_flow = torch.from_numpy(gt_flow).float()
            pred_flow = torch.from_numpy(pred_flow).float()
            joints = torch.from_numpy(joints).float()
            skin_input = torch.from_numpy(skin_input).float()
            skin_label = torch.from_numpy(skin_label).float()
            skin_nn = torch.from_numpy(skin_nn).long()
            skin_nnjids = torch.from_numpy(skin_nnjids).long()
            loss_mask = torch.from_numpy(loss_mask).long()
            gt_skin = torch.from_numpy(gt_skin).float()

            data_list.append(Data(pos=pos, tpl_edge_index=tpl_e, geo_edge_index=geo_e,
                                  pred_flow=pred_flow, gt_flow=gt_flow, name=name, 
                                  mask=m, joints=joints, offsets=offsets, gt_skin=gt_skin,
                                  skin_input=skin_input, skin_label=skin_label, skin_nn=skin_nn, 
                                  skin_nnjids=skin_nnjids, loss_mask=loss_mask))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
