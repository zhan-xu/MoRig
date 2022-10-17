import torch
from torch.nn import Sequential as Seq, Linear as Lin, Parameter
from torch_geometric.nn import knn
from torch_scatter import scatter_max
from models.basic_modules import MLP, SAModule, GlobalSAModule, FPModule, GCU

__all__ = ['corrnet']


class CorrNet(torch.nn.Module):
    def __init__(self, input_feature, output_feature, temprature, aggr='max'):
        super(CorrNet, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.temprature = Parameter(torch.Tensor([temprature]))

        self.vtx_gcu_1 = GCU(in_channels=3, out_channels=32, aggr=aggr)
        self.vtx_gcu_2 = GCU(in_channels=32, out_channels=64, aggr=aggr)
        self.vtx_gcu_3 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.vtx_gcu_4 = GCU(in_channels=256, out_channels=512, aggr=aggr)
        self.vtx_mlp_glb = MLP([(32 + 64 + 256 + 512), 1024])
        self.vtx_mlp = Seq(MLP([1024 + 3 + 32 + 64 + 256 + 512, 1024, 256]), Lin(256, output_feature))
    
        self.pts_sa1_module = SAModule(0.5, 0.12, MLP([input_feature, 32, 32, 64]), max_num_neighbors=64)
        self.pts_sa2_module = SAModule(0.25, 0.25, MLP([64 + 3, 64, 64, 128]), max_num_neighbors=64)
        self.pts_sa3_module = SAModule(0.25, 0.5, MLP([128 + 3, 256, 256, 256]), max_num_neighbors=64)
        self.pts_sa4_module = GlobalSAModule(MLP([256 + 3, 256, 256, 512]))

        self.pts_fp4_module = FPModule(1, MLP([512 + 256, 256, 256]))
        self.pts_fp3_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.pts_fp2_module = FPModule(3, MLP([128 + 64, 128, 64]))
        self.pts_fp1_module = FPModule(3, MLP([64, 64, 64]))
        self.pts_mlp = Seq(MLP([64, 64]), Lin(64, output_feature))

        self.lin_vismask = Seq(MLP([2 * output_feature + 1, 256, 128, 64]), Lin(64, 1))

    def forward(self, data, train_vismask, random_start=True):
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.vtx_batch
        x_1 = self.vtx_gcu_1(data.vtx, tpl_edge_index, geo_edge_index)
        x_2 = self.vtx_gcu_2(x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.vtx_gcu_3(x_2, tpl_edge_index, geo_edge_index)
        x_4 = self.vtx_gcu_4(x_3, tpl_edge_index, geo_edge_index)
        x_5 = self.vtx_mlp_glb(torch.cat([x_1, x_2, x_3, x_4], dim=1))
        x_global, _ = scatter_max(x_5, data.vtx_batch, dim=0)
        x_global = torch.repeat_interleave(x_global, torch.bincount(data.vtx_batch), dim=0)
        x_6 = torch.cat([x_global, data.vtx, x_1, x_2, x_3, x_4], dim=1)
        out_vtx = self.vtx_mlp(x_6)
        out_vtx = torch.nn.functional.normalize(out_vtx, dim=1)

        pts_sa0_out = (None, data.pts, data.pts_batch)
        pts_sa1_out = self.pts_sa1_module(*pts_sa0_out, random_start)
        pts_sa2_out = self.pts_sa2_module(*pts_sa1_out, random_start)
        pts_sa3_out = self.pts_sa3_module(*pts_sa2_out, random_start)
        pts_sa4_out = self.pts_sa4_module(*pts_sa3_out)
        pts_fp4_out = self.pts_fp4_module(*pts_sa4_out, *pts_sa3_out)
        pts_fp3_out = self.pts_fp3_module(*pts_fp4_out, *pts_sa2_out)
        pts_fp2_out = self.pts_fp2_module(*pts_fp3_out, *pts_sa1_out)
        out_pts, _, _ = self.pts_fp1_module(*pts_fp2_out, *pts_sa0_out)
        out_pts = self.pts_mlp(out_pts)
        out_pts = torch.nn.functional.normalize(out_pts, dim=1)

        if train_vismask:  # torch_geometric.nn.knn only works on cuda
            if torch.cuda.is_available():
                assign_index = knn(out_pts, out_vtx, 1, data.pts_batch, data.vtx_batch, cosine=True)
                out_combine = torch.cat([out_vtx[assign_index[0]], out_pts[assign_index[1]],torch.sum(out_vtx[assign_index[0]] * out_pts[assign_index[1]], dim=1)[:, None]], dim=1)
            else:
                out_combine = []
                for i in range(len(torch.unique(data.vtx_batch))):
                    with torch.no_grad():
                        feature_similarity_i = torch.matmul(out_vtx[data.vtx_batch==i], out_pts[data.pts_batch==i].transpose(0, 1))
                        max_sim, nnidx = torch.max(feature_similarity_i, dim=1)
                    out_combine.append(torch.cat([out_vtx[data.vtx_batch==i], out_pts[data.pts_batch==i][nnidx], max_sim.unsqueeze(dim=1)], dim=1))
                out_combine = torch.cat(out_combine, dim=0)
            out_vismask = self.lin_vismask(out_combine)
        else:
            out_vismask = None
        return out_vtx, out_pts, out_vismask, self.temprature


def corrnet(**kwargs):
    model = CorrNet(input_feature=kwargs['input_feature'], output_feature=kwargs['output_feature'], temprature=kwargs['temprature'])
    return model