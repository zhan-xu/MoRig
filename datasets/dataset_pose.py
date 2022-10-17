import os, glob, numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops


class GraphData(Data):
    def __init__(self, vtx_traj=None, pts_traj=None, corr_v2p_all=None, corr_p2v_all=None,
                 vismask_all=None, tpl_edge_index=None, geo_edge_index=None, name=None):
        super(GraphData, self).__init__()
        self.vtx_traj = vtx_traj
        self.pts_traj = pts_traj
        self.corr_v2p_all = corr_v2p_all
        self.corr_p2v_all = corr_p2v_all
        self.vismask_all = vismask_all
        self.tpl_edge_index = tpl_edge_index
        self.geo_edge_index = geo_edge_index
        self.name = name

    def __inc__(self, key, value, *args, **kwargs):
        if "edge_index" in key:
            return self.vtx_traj.size(0)
        else:
            return super(GraphData, self).__inc__(key, value)


class ModelsResourcesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ModelsResourcesDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_filelist = glob.glob(os.path.join(self.root, f'*_vtx_traj.npy'))
        return raw_filelist

    @property
    def processed_file_names(self):
        return '{:s}_mr_pose_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in tqdm(range(len(self.raw_paths))):
            name = self.raw_paths[i].split('/')[-1].split('_')[0]
            vtx_traj = np.load(self.raw_paths[i])
            vtx_traj = vtx_traj.reshape(-1, 303)
            pts_traj = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_pts_traj.npy"))
            corr_v2p = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_v2p.npy"))
            corr_p2v = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_p2v.npy"))
            vismask = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_vismask.npy"))
            tpl_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_tpl_e.txt')).T
            geo_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_geo_e.txt')).T

            # pick frames
            vtx_frames = []
            pts_frames = []
            corr_v2p_frames = []
            corr_p2v_frames = []
            vismask_frames = []
            for key_t in np.arange(0, 110, 20):
                vtx_frames.append(vtx_traj[:, 3*key_t:3*(key_t+1)])
                pts_frames.append(pts_traj[:, 3*key_t:3*(key_t+1)])
                corr_v2p_frames.append(corr_v2p[corr_v2p[:, -1] == key_t])
                corr_p2v_frames.append(corr_p2v[corr_p2v[:, -1] == key_t])
                vismask_frames.append(vismask[:, key_t])

            vtx_frames = np.concatenate(vtx_frames, axis=1)
            pts_frames = np.concatenate(pts_frames, axis=1)
            corr_v2p_frames = np.concatenate(corr_v2p_frames, axis=0)
            corr_p2v_frames = np.concatenate(corr_p2v_frames, axis=0)
            corr_v2p_frames[:, -1] = corr_v2p_frames[:, -1] / 20
            corr_p2v_frames[:, -1] = corr_p2v_frames[:, -1] / 20
            vismask_frames = np.stack(vismask_frames, axis=1)

            # convert to tensor
            vtx_traj = torch.from_numpy(vtx_frames).float()
            pts_traj = torch.from_numpy(pts_frames).float()
            corr_v2p = torch.from_numpy(corr_v2p_frames).long()
            corr_p2v = torch.from_numpy(corr_p2v_frames).long()
            vismask = torch.from_numpy(vismask_frames).float()
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=vtx_traj.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=vtx_traj.size(0))
            # add to data class
            data = GraphData(vtx_traj=vtx_traj, pts_traj=pts_traj,
                             corr_v2p_all=corr_v2p, corr_p2v_all=corr_p2v, vismask_all=vismask,
                             tpl_edge_index=tpl_e, geo_edge_index=geo_e, name=name)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SeqModelsResourcesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SeqModelsResourcesDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_filelist = glob.glob(os.path.join(self.root, f'*_vtx_traj.npy'))
        return raw_filelist

    @property
    def processed_file_names(self):
        return '{:s}_mr_seq_pose_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in tqdm(range(len(self.raw_paths))):
            name = self.raw_paths[i].split('/')[-1].split('_')[0]
            vtx_traj = np.load(self.raw_paths[i])
            vtx_traj = vtx_traj.reshape(-1, 303)
            pts_traj = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_pts_traj.npy"))
            corr_v2p = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_v2p.npy"))
            corr_p2v = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_p2v.npy"))
            vismask = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_vismask.npy"))
            tpl_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_tpl_e.txt')).T
            geo_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_geo_e.txt')).T

            # pick frames
            vtx_frames = []
            pts_frames = []
            corr_v2p_frames = []
            corr_p2v_frames = []
            vismask_frames = []
            for key_t in np.arange(0, 21):
                vtx_frames.append(vtx_traj[:, 3*key_t:3*(key_t+1)])
                pts_frames.append(pts_traj[:, 3*key_t:3*(key_t+1)])
                corr_v2p_frames.append(corr_v2p[corr_v2p[:, -1] == key_t])
                corr_p2v_frames.append(corr_p2v[corr_p2v[:, -1] == key_t])
                vismask_frames.append(vismask[:, key_t])

            vtx_frames = np.concatenate(vtx_frames, axis=1)
            pts_frames = np.concatenate(pts_frames, axis=1)
            corr_v2p_frames = np.concatenate(corr_v2p_frames, axis=0)
            corr_p2v_frames = np.concatenate(corr_p2v_frames, axis=0)
            corr_v2p_frames[:, -1] = corr_v2p_frames[:, -1]
            corr_p2v_frames[:, -1] = corr_p2v_frames[:, -1]
            vismask_frames = np.stack(vismask_frames, axis=1)

            # convert to tensor
            vtx_traj = torch.from_numpy(vtx_frames).float()
            pts_traj = torch.from_numpy(pts_frames).float()
            corr_v2p = torch.from_numpy(corr_v2p_frames).long()
            corr_p2v = torch.from_numpy(corr_p2v_frames).long()
            vismask = torch.from_numpy(vismask_frames).float()
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=vtx_traj.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=vtx_traj.size(0))
            # add to data class
            data = GraphData(vtx_traj=vtx_traj, pts_traj=pts_traj,
                             corr_v2p_all=corr_v2p, corr_p2v_all=corr_p2v, vismask_all=vismask,
                             tpl_edge_index=tpl_e, geo_edge_index=geo_e, name=name)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
class DeformingThingsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DeformingThingsDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_filelist = glob.glob(os.path.join(self.root, f'*_vtx_traj.npy'))
        return raw_filelist

    @property
    def processed_file_names(self):
        return '{:s}_dt_pose_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in tqdm(range(len(self.raw_paths))):
            name = self.raw_paths[i].split('/')[-1].split('_vtx_traj.npy')[0]
            vtx_traj = np.load(self.raw_paths[i])
            pts_traj = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_pts_traj.npy"))
            vtx_traj = vtx_traj.reshape(-1, 300)
            pts_traj = pts_traj.reshape(-1, 300)
            corr_v2p = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_v2p.npy"))
            corr_p2v = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_p2v.npy"))
            vismask = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_vismask.npy"))
            tpl_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_tpl_e.txt')).T
            geo_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_geo_e.txt')).T

            # pick frames
            vtx_frames = []
            pts_frames = []
            corr_v2p_frames = []
            corr_p2v_frames = []
            vismask_frames = []
            for key_t in np.arange(0, 100, 19):
                vtx_frames.append(vtx_traj[:, 3*key_t:3*(key_t+1)])
                pts_frames.append(pts_traj[:, 3*key_t:3*(key_t+1)])
                corr_v2p_frames.append(corr_v2p[corr_v2p[:, -1] == key_t])
                corr_p2v_frames.append(corr_p2v[corr_p2v[:, -1] == key_t])
                vismask_frames.append(vismask[:, key_t])

            vtx_frames = np.concatenate(vtx_frames, axis=1)
            pts_frames = np.concatenate(pts_frames, axis=1)
            corr_v2p_frames = np.concatenate(corr_v2p_frames, axis=0)
            corr_p2v_frames = np.concatenate(corr_p2v_frames, axis=0)
            corr_v2p_frames[:, -1] = corr_v2p_frames[:, -1] / 19
            corr_p2v_frames[:, -1] = corr_p2v_frames[:, -1] / 19
            vismask_frames = np.stack(vismask_frames, axis=1)

            # convert to tensor
            vtx_traj = torch.from_numpy(vtx_frames).float()
            pts_traj = torch.from_numpy(pts_frames).float()
            corr_v2p = torch.from_numpy(corr_v2p_frames).long()
            corr_p2v = torch.from_numpy(corr_p2v_frames).long()
            vismask = torch.from_numpy(vismask_frames).float()
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=vtx_traj.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=vtx_traj.size(0))
            # add to data class
            data = GraphData(vtx_traj=vtx_traj, pts_traj=pts_traj,
                             corr_v2p_all=corr_v2p, corr_p2v_all=corr_p2v, vismask_all=vismask,
                             tpl_edge_index=tpl_e, geo_edge_index=geo_e, name=name)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SeqDeformingThingsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SeqDeformingThingsDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_filelist = glob.glob(os.path.join(self.root, f'*_vtx_traj.npy'))
        return raw_filelist

    @property
    def processed_file_names(self):
        return '{:s}_dt_seq_pose_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in tqdm(range(len(self.raw_paths))):
            name = self.raw_paths[i].split('/')[-1].split('_vtx_traj')[0]
            vtx_traj = np.load(self.raw_paths[i])
            pts_traj = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_pts_traj.npy"))
            vtx_traj = vtx_traj.reshape(-1, 300)
            pts_traj = pts_traj.reshape(-1, 300)
            corr_v2p = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_v2p.npy"))
            corr_p2v = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_corr_p2v.npy"))
            vismask = np.load(self.raw_paths[i].replace("_vtx_traj.npy", "_vismask.npy"))
            tpl_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_tpl_e.txt')).T
            geo_e = np.loadtxt(self.raw_paths[i].replace('_vtx_traj.npy', '_geo_e.txt')).T

            # pick frames
            vtx_frames = []
            pts_frames = []
            corr_v2p_frames = []
            corr_p2v_frames = []
            vismask_frames = []
            for key_t in np.arange(0, 21):
                vtx_frames.append(vtx_traj[:, 3*key_t:3*(key_t+1)])
                pts_frames.append(pts_traj[:, 3*key_t:3*(key_t+1)])
                corr_v2p_frames.append(corr_v2p[corr_v2p[:, -1] == key_t])
                corr_p2v_frames.append(corr_p2v[corr_p2v[:, -1] == key_t])
                vismask_frames.append(vismask[:, key_t])

            vtx_frames = np.concatenate(vtx_frames, axis=1)
            pts_frames = np.concatenate(pts_frames, axis=1)
            corr_v2p_frames = np.concatenate(corr_v2p_frames, axis=0)
            corr_p2v_frames = np.concatenate(corr_p2v_frames, axis=0)
            corr_v2p_frames[:, -1] = corr_v2p_frames[:, -1]
            corr_p2v_frames[:, -1] = corr_p2v_frames[:, -1]
            vismask_frames = np.stack(vismask_frames, axis=1)

            # convert to tensor
            vtx_traj = torch.from_numpy(vtx_frames).float()
            pts_traj = torch.from_numpy(pts_frames).float()
            corr_v2p = torch.from_numpy(corr_v2p_frames).long()
            corr_p2v = torch.from_numpy(corr_p2v_frames).long()
            vismask = torch.from_numpy(vismask_frames).float()
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=vtx_traj.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=vtx_traj.size(0))
            # add to data class
            data = GraphData(vtx_traj=vtx_traj, pts_traj=pts_traj,
                             corr_v2p_all=corr_v2p, corr_p2v_all=corr_p2v, vismask_all=vismask,
                             tpl_edge_index=tpl_e, geo_edge_index=geo_e, name=name)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])