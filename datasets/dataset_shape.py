import os
import torch
import numpy as np
import glob
import open3d as o3d
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops


class GraphData(Data):
    def __init__(self, vtx=None, pts=None, corr_v2p=None, corr_p2v=None, vismask=None, flow=None,
                 tpl_edge_index=None, geo_edge_index=None, name=None):
        super(GraphData, self).__init__()
        self.vtx = vtx
        self.pts = pts
        self.corr_v2p = corr_v2p
        self.corr_p2v = corr_p2v
        self.vismask = vismask
        self.flow = flow
        self.tpl_edge_index = tpl_edge_index
        self.geo_edge_index = geo_edge_index
        self.name = name

    def __inc__(self, key, value, *args, **kwargs):
        if "edge_index" in key:
            return self.vtx.size(0)
        else:
            return super(GraphData, self).__inc__(key, value)
        
        
class ModelsResourcesShapeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ModelsResourcesShapeDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_filelist = glob.glob(os.path.join(self.root, f'*_0.obj'))
        return raw_filelist

    @property
    def processed_file_names(self):
        return '{:s}_mr_shape_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in tqdm(range(len(self.raw_paths))):
            name = self.raw_paths[i].split('/')[-1].split('_')[0]
            # load everything
            mesh = o3d.io.read_triangle_mesh(self.raw_paths[i])
            vtx = np.asarray(mesh.vertices)
            pts = np.load(self.raw_paths[i].replace("_0.obj", "_pts.npy"))
            flow = np.load(self.raw_paths[i].replace("_0.obj", "_flow.npy"))
            corr_v2p = np.load(self.raw_paths[i].replace("_0.obj", "_corr_v2p.npy"))
            corr_p2v = np.load(self.raw_paths[i].replace("_0.obj", "_corr_p2v.npy"))
            vismask = np.load(self.raw_paths[i].replace("_0.obj", "_vismask.npy"))
            tpl_e = np.loadtxt(self.raw_paths[i].replace('_0.obj', '_tpl_e.txt')).T
            geo_e = np.loadtxt(self.raw_paths[i].replace('_0.obj', '_geo_e.txt')).T

            vtx = torch.from_numpy(vtx).float()
            pts = torch.from_numpy(pts).float()
            flow = torch.from_numpy(flow).float()
            corr_v2p = torch.from_numpy(corr_v2p).long()
            corr_p2v = torch.from_numpy(corr_p2v).long()
            vismask = torch.from_numpy(vismask).float()
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=vtx.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=vtx.size(0))
            # add to data class
            data = GraphData(vtx=vtx, pts=pts, corr_v2p=corr_v2p, corr_p2v=corr_p2v, vismask=vismask, flow=flow,
                             tpl_edge_index=tpl_e, geo_edge_index=geo_e, name=name)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
