import numpy as np
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import knn_interpolate, fps, radius, global_max_pool, PointConv


def radius_cpu(x, y, r, max_num_neighbors):
    """ Finds for each element in: obj: `y` all points in: obj:`x` within distance: obj:`r`."""
    dist = torch.cdist(y.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
    valid_positions = (dist <= r)
    reduced_rows = torch.where(torch.sum(valid_positions, dim=1) > max_num_neighbors)[0]
    reserve_rows = torch.where(torch.sum(valid_positions, dim=1) <= max_num_neighbors)[0]
    if len(reduced_rows) > 0:
        col_for_reduced_rows = torch.multinomial(valid_positions[reduced_rows].float(), max_num_neighbors)
        edge_for_reduced_rows = torch.stack([col_for_reduced_rows.flatten(),
                                             torch.repeat_interleave(reduced_rows, max_num_neighbors, dim=0).to(
                                                 valid_positions.device)], dim=0)
    _, col_for_reserve_rows = torch.where(valid_positions[reserve_rows])
    num_edge_for_reserve_rows = valid_positions[reserve_rows].sum(dim=1)
    edge_for_reserve_rows = torch.stack([col_for_reserve_rows,
                                         torch.repeat_interleave(reserve_rows.to(valid_positions.device),
                                                                 num_edge_for_reserve_rows, dim=0)], dim=0)
    if len(reduced_rows) > 0:
        edges = torch.cat((edge_for_reserve_rows, edge_for_reduced_rows), dim=1)
    else:
        edges = edge_for_reserve_rows
    return edges

def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i], momentum=0.1))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])


def batch_fps(pts, K):
    calc_distances = lambda p0, pts: ((p0 - pts) ** 2).sum(dim=1)
    #np.random.seed(0)

    def fps(x):
        pts, K = x
        farthest_idx = torch.LongTensor(K)
        farthest_idx.zero_()
        farthest_idx[0] = np.random.randint(len(pts))
        distances = calc_distances(pts[farthest_idx[0]], pts)
        for i in range(1, K):
            farthest_idx[i] = torch.max(distances, dim=0)[1]
            farthest_pts = pts[farthest_idx[i]]
            distances = torch.min(distances, calc_distances(farthest_pts, pts))
        pts_sampled = pts[farthest_idx, :]
        return pts_sampled, farthest_idx

    fps_res = list(map(fps, [(pts[i].to('cpu'), K) for i in range(len(pts))]))
    batch_pts = [i[0] for i in fps_res]
    batch_pts = torch.stack(batch_pts, dim=0).to(pts[0].device)
    batch_id = [i[1] for i in fps_res]
    batch_id = torch.stack(batch_id, dim=0).long().to(pts[0].device)
    return batch_pts, batch_id



"""PointNet2 components"""
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, max_num_neighbors):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch, random_start=True):
        idx = fps(pos, batch, ratio=self.ratio, random_start=random_start) #, random_start=False
        if torch.cuda.is_available():
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.max_num_neighbors)
            edge_index = torch.stack([col, row], dim=0)
        else:
            edge_index = radius_cpu(pos, pos[idx], self.r, max_num_neighbors=self.max_num_neighbors)
        if x is None:
            x = self.conv((None, None), (pos, pos[idx]), edge_index)
        else:
            x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class SAModule_msg(torch.nn.Module):
    def __init__(self, ratio, r_list, mlp_list, max_num_neighbors_list):
        super(SAModule_msg, self).__init__()
        self.ratio = ratio
        self.r_list = r_list
        self.max_num_neighbors_list = max_num_neighbors_list
        self.mlp_list = torch.nn.ModuleList()
        for nn in mlp_list:
            self.mlp_list.append(PointConv(nn))

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio) #, random_start=False
        new_points_list = []
        for i in range(len(self.mlp_list)):
            if torch.cuda.is_available():
                row, col = radius(pos, pos[idx], self.r_list[i], batch, batch[idx], max_num_neighbors=self.max_num_neighbors_list[i])
                edge_index = torch.stack([col, row], dim=0)
            else:
                edge_index = radius_cpu(pos, pos[idx], self.r, max_num_neighbors=self.max_num_neighbors)
            if x is None:
                new_points_list.append(self.mlp_list[i]((None, None), (pos, pos[idx]), edge_index))
            else:
                new_points_list.append(self.mlp_list[i]((x, x[idx]), (pos, pos[idx]), edge_index))
        x = torch.cat(new_points_list, dim=1)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


"""GCN components"""
class EdgeConv(MessagePassing):
    def __init__(self, nn_pos, aggr='max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn_pos = nn_pos

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        pos_feat = self.nn_pos(torch.cat([x_i, (x_j - x_i)], dim=1))
        return pos_feat

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, aggr_out.shape[-1])
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GCU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(GCU, self).__init__()
        self.edge_conv_tpl = EdgeConv(nn_pos=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        self.edge_conv_geo = EdgeConv(nn_pos=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, pos, tpl_edge_index, geo_edge_index):
        x_tpl = self.edge_conv_tpl(pos, tpl_edge_index)
        x_geo = self.edge_conv_geo(pos, geo_edge_index)
        x_out = torch.cat([x_tpl, x_geo], dim=1)
        x_out = self.mlp(x_out)
        return x_out

class EdgeConvMotion(MessagePassing):
    def __init__(self, nn_x, nn_pos, aggr='max', **kwargs):
        super(EdgeConvMotion, self).__init__(aggr=aggr, **kwargs)
        self.nn_x = nn_x
        self.nn_pos = nn_pos

    def forward(self, pos, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, pos=pos, x=x)

    def message(self, pos_i, pos_j, x_i, x_j):
        x_feat = self.nn_x(torch.cat([x_i, (x_j - x_i)], dim=1))
        pos_feat = self.nn_pos(torch.cat([pos_i, (pos_j - pos_i)], dim=1))
        return torch.cat([x_feat, pos_feat], dim=1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, aggr_out.shape[-1])
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GCUMotion(torch.nn.Module):
    def __init__(self, in_channels, out_channels, in_channel_pos=3, dim_pos_feat=16, aggr='max'):
        super(GCUMotion, self).__init__()
        self.edge_conv_tpl = EdgeConvMotion(nn_x=MLP([in_channels * 2, out_channels // 2, out_channels // 2]),
                                            nn_pos=MLP([in_channel_pos * 2, dim_pos_feat, dim_pos_feat]), aggr=aggr)
        self.edge_conv_geo = EdgeConvMotion(nn_x=MLP([in_channels * 2, out_channels // 2, out_channels // 2]),
                                            nn_pos=MLP([in_channel_pos * 2, dim_pos_feat, dim_pos_feat]), aggr=aggr)
        self.mlp = MLP([out_channels + dim_pos_feat * 2, out_channels])

    def forward(self, pos, x, tpl_edge_index, geo_edge_index):
        x_tpl = self.edge_conv_tpl(pos, x, tpl_edge_index)
        x_geo = self.edge_conv_geo(pos, x, geo_edge_index)
        x_out = torch.cat([x_tpl, x_geo], dim=1)
        x_out = self.mlp(x_out)
        return x_out
