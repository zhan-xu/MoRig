import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_scatter import scatter_max, scatter_add
from models.corrnet import CorrNet
from models.basic_modules import MLP, GCUMotion
from torch_geometric.nn import knn

__all__ = ['deformnet']


class GCNDeform(torch.nn.Module):
    def __init__(self, chn_in, chn_output, aggr='max'):
        super(GCNDeform, self).__init__()
        self.gcu_1 = GCUMotion(in_channels=chn_in, out_channels=128, aggr=aggr)
        self.gcu_2 = GCUMotion(in_channels=128, out_channels=256, aggr=aggr)
        self.gcu_3 = GCUMotion(in_channels=256, out_channels=512, aggr=aggr)
        self.mlp_glb = MLP([(128 + 256 + 512), 1024])
        self.mlp_tramsform = Sequential(MLP([1024 + 3 + chn_in + 128 + 256 + 512, 1024, 256]), Linear(256, chn_output))

    def forward(self, pos, feature, geo_edge_index, tpl_edge_index, batch):
        x_1 = self.gcu_1(pos, feature, tpl_edge_index, geo_edge_index)
        x_2 = self.gcu_2(pos, x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(pos, x_2, tpl_edge_index, geo_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))
        x_global, _ = scatter_max(x_4, batch, dim=0)
        x_global = torch.repeat_interleave(x_global, torch.bincount(batch), dim=0)
        x_5 = torch.cat([x_global, pos, feature, x_1, x_2, x_3], dim=1)
        out = self.mlp_tramsform(x_5)
        return out


class DeformNet(torch.nn.Module):
    def __init__(self, tau_nce, num_interp):
        super(DeformNet, self).__init__()
        self.corr_extractor = CorrNet(3, 64, temprature=tau_nce)
        self.completing = GCNDeform(chn_in=4, chn_output=3)
        self.num_interp = num_interp

    def forward(self, data):
        vtx_feature, pts_feature, pred_vismask, tau = self.corr_extractor(data, train_vismask=True)
        pred_vismask = torch.sigmoid(pred_vismask)
        for i in range(len(torch.unique(data.vtx_batch))):
            pred_vismask[data.vtx_batch == i] = (pred_vismask[data.vtx_batch == i] - pred_vismask[data.vtx_batch == i].min()) / \
                                                (pred_vismask[data.vtx_batch == i].max() - pred_vismask[data.vtx_batch == i].min())

        # for visible part
        assign_index = knn(pts_feature, vtx_feature, self.num_interp, data.pts_batch, data.vtx_batch, cosine=True)
        euclidean_dist = data.pts[assign_index[1]] - data.vtx[assign_index[0]]
        # In practice, we find the following simplification is more efficient. feature_sim is always positive since we only consider top-5 most similar pairs.
        feature_sim = torch.sum(pts_feature[assign_index[1]] * vtx_feature[assign_index[0]], dim=-1, keepdim=True)
        feature_sim = feature_sim * pred_vismask.repeat_interleave(self.num_interp, dim=0)  # maybe remove pred_vismask here??
        flow_init = scatter_add(euclidean_dist * feature_sim, assign_index[0], dim=0) / scatter_add(feature_sim, assign_index[0], dim=0)
        
        # for invisible part
        vis_vids = (pred_vismask >= 0.5).squeeze(dim=1)
        invis_vids = (pred_vismask < 0.5).squeeze(dim=1)
        vis_vtx_feature = vtx_feature[vis_vids]
        invis_vtx_feature = vtx_feature[invis_vids]
        vis_vtx_batch = data.vtx_batch[vis_vids]
        invis_vtx_batch = data.vtx_batch[invis_vids]
        vis_flow_init = flow_init[vis_vids]

        # debug here
        # import open3d as o3d
        # from utils.vis_utils import drawSphere
        # for i in range(len(torch.unique(data.vtx_batch))):
        #     vis_vtx_i = data.vtx[vis_vids][vis_vtx_batch == i].detach().to("cpu").numpy()
        #     invis_vtx_i = data.vtx[invis_vids][invis_vtx_batch == i].detach().to("cpu").numpy()
        #     invis_vtx_feature_i = invis_vtx_feature[invis_vtx_batch == i].detach().to("cpu").numpy()
        #     vis_vtx_feature_i = vis_vtx_feature[vis_vtx_batch == i].detach().to("cpu").numpy()
        #     sim = np.matmul(invis_vtx_feature_i, vis_vtx_feature_i.T)
        #     nnidx = np.argsort(sim, axis=1)[:, -10:]
        #     pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(data.vtx[data.vtx_batch==i].detach().to("cpu").numpy()))
        #     pcd.paint_uniform_color([0.8, 0.8, 0.8])
        #     for t in range(3):
        #         vis = o3d.visualization.Visualizer()
        #         vis.create_window()
        #         vis.add_geometry(pcd)
        #         vis.add_geometry(drawSphere(invis_vtx_i[len(invis_vtx_i)//4*t], color=[1.0, 0.0, 0.0], radius=0.007))
        #         nnidx_j = nnidx[len(invis_vtx_i) // 4 * t]
        #         for j in range(len(nnidx_j)):
        #             vis.add_geometry(drawSphere(vis_vtx_i[nnidx_j[j]], color=[0.0, 0.0, 1.0], radius=0.007))
        #         vis.run()
        #         #vis.capture_screen_image(f"{i}_{t}.png")
        #         vis.destroy_window()
        
        # We here originally find nearest geodesic neighbors, which requires pre-computing geodesic distance among vertices.
        # To make the dataset and training procedure less painful, we change to euclidean nearest neighbors.
        # We found the influence on performance is negligible.
        assign_index2 = knn(vis_vtx_feature, invis_vtx_feature, self.num_interp, vis_vtx_batch, invis_vtx_batch, cosine=True)
        feature_sim = torch.sum(vis_vtx_feature[assign_index2[1]] * invis_vtx_feature[assign_index2[0]], dim=-1, keepdim=True)
        invis_flow_init = scatter_add(vis_flow_init[assign_index2[1]] * feature_sim, assign_index2[0], dim=0) / scatter_add(feature_sim, assign_index2[0], dim=0)
        flow_init[invis_vids] = invis_flow_init

        l1_points = torch.cat((flow_init, pred_vismask), dim=-1)
        pred_flow = self.completing(data.vtx, l1_points, data.geo_edge_index, data.tpl_edge_index, data.vtx_batch)
        return pred_flow, vtx_feature, pts_feature, pred_vismask, tau
    
    
def deformnet(**kwargs):
    model = DeformNet(tau_nce=kwargs["tau_nce"], num_interp=kwargs["num_interp"])
    return model
