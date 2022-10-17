import torch
from torch.nn import Sequential, Linear
from torch_scatter import scatter_max
from models.basic_modules import MLP, GCUMotion
import numpy as np

__all__ = ['jointnet_motion', 'masknet_motion', 'skinnet_motion']


class TemporalAttn(torch.nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, dim_feedforward, output_size):
        """Different from transformer encoder layer,  hidden size is defined for a single head, not for all heads"""
        super(TemporalAttn, self).__init__()
        self.num_heads = num_heads
        self.w_qs = Linear(input_size, hidden_size * num_heads, bias=False)
        self.w_ks = Linear(input_size, hidden_size * num_heads, bias=False)
        self.w_vs = Linear(input_size, hidden_size * num_heads, bias=False)
        self.w_o = Linear(hidden_size * num_heads, hidden_size, bias=False)
        self.feedforward = MLP([hidden_size, dim_feedforward, output_size])
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, input_size))

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        """From https://d2l.ai/chapter_attention-mechanisms/multihead-attention.html"""
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of `transpose_qkv`."""
        """From https://d2l.ai/chapter_attention-mechanisms/multihead-attention.html"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, x):
        cls_token = self.cls_token.expand([x.shape[0], -1, -1])
        x_expand = torch.cat([cls_token, x], dim=1)
        q, k, v = self.w_qs(x_expand), self.w_ks(x_expand), self.w_vs(x_expand)
        q, k, v = self.transpose_qkv(q), self.transpose_qkv(k), self.transpose_qkv(v)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = torch.nn.functional.softmax(attn / np.sqrt(k.size(-1)), dim=-1)
        res = torch.bmm(attn, v)
        res = self.w_o(self.transpose_output(res))
        res = self.feedforward(res[:, 0, :])  # index only the cls token for classification
        return res


class GCNRig(torch.nn.Module):
    def __init__(self, chn_feature, chn_output, aggr='max'):
        super(GCNRig, self).__init__()
        self.gcu_1 = GCUMotion(in_channels=chn_feature, out_channels=64, dim_pos_feat=16, aggr=aggr)
        self.gcu_2 = GCUMotion(in_channels=64, out_channels=256, dim_pos_feat=16, aggr=aggr)
        self.gcu_3 = GCUMotion(in_channels=256, out_channels=512, dim_pos_feat=16, aggr=aggr)
        self.mlp_glb = MLP([(64 + 256 + 512), 1024])
        self.mlp_transform = Sequential(MLP([1024 + 3 + chn_feature + 64 + 256 + 512, 1024, 256]), Linear(256, chn_output))

    def forward(self, pos, feature, tpl_edge_index, geo_edge_index, batch):
        x_1 = self.gcu_1(pos, feature, tpl_edge_index, geo_edge_index)
        x_2 = self.gcu_2(pos, x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(pos, x_2, tpl_edge_index, geo_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))
        x_global, _ = scatter_max(x_4, batch, dim=0)
        x_global = torch.repeat_interleave(x_global, torch.bincount(batch), dim=0)
        x_5 = torch.cat([x_global, pos, feature, x_1, x_2, x_3], dim=1)
        x_out = self.mlp_transform(x_5)
        return x_out


class JointNetMotion(torch.nn.Module):
    def __init__(self, num_keyframes, chn_output, aggr_method, aggr='max'):
        super(JointNetMotion, self).__init__()
        self.num_keyframes = num_keyframes
        self.aggr_method = aggr_method
        self.motionNet = GCNRig(chn_feature=3, chn_output=32, aggr=aggr)
        if self.aggr_method == "attn":
            self.aggragator = TemporalAttn(input_size=32, num_heads=2, hidden_size=64, dim_feedforward=512, output_size=64)
            self.jointnet = GCNRig(chn_feature=64, chn_output=chn_output, aggr=aggr)
        else:
            self.jointnet = GCNRig(chn_feature=32, chn_output=chn_output, aggr=aggr)

    def forward(self, data, input_flow):
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch
        motion_all = []
        for t in range(self.num_keyframes):
            motion_t = self.motionNet(data.pos, input_flow[:, 3 * t:3 * t + 3], tpl_edge_index, geo_edge_index, batch)
            motion_t = torch.nn.functional.normalize(motion_t, dim=1)
            motion_all.append(motion_t)
        motion_all = torch.stack(motion_all, dim=1)
        if self.aggr_method == "attn":
            motion_aggr = self.aggragator(motion_all)
        elif self.aggr_method == "mean":
            motion_aggr = torch.mean(motion_all, dim=1)
        elif self.aggr_method == "max":
            motion_aggr = torch.max(motion_all, dim=1)[0]
        else:
            raise NotImplementedError
        motion_aggr = torch.nn.functional.normalize(motion_aggr, dim=1)
        pred_shift = self.jointnet(data.pos, motion_aggr, tpl_edge_index, geo_edge_index, batch)
        return motion_all, motion_aggr, pred_shift
    

class MaskNetMotion(torch.nn.Module):
    def __init__(self, num_keyframes, chn_output, aggr_method, aggr='max'):
        super(MaskNetMotion, self).__init__()        
        self.num_keyframes = num_keyframes
        self.aggr_method = aggr_method
        self.motionNet = GCNRig(chn_feature=3, chn_output=32, aggr=aggr)
        if self.aggr_method == "attn":
            self.aggragator = TemporalAttn(input_size=32, num_heads=2, hidden_size=64, dim_feedforward=512, output_size=64)
            self.masknet = GCNRig(chn_feature=64, chn_output=chn_output, aggr=aggr)
        else:
            self.masknet = GCNRig(chn_feature=32, chn_output=chn_output, aggr=aggr)

    def forward(self, data, input_flow):
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch
        motion_all = []
        for t in range(self.num_keyframes):
            motion_t = self.motionNet(data.pos, input_flow[:, 3 * t:3 * t + 3], tpl_edge_index, geo_edge_index, batch)
            motion_t = torch.nn.functional.normalize(motion_t, dim=1)
            motion_all.append(motion_t)
        motion_all = torch.stack(motion_all, dim=1)
        if self.aggr_method == "attn":
            motion_aggr = self.aggragator(motion_all)
        elif self.aggr_method == "mean":
            motion_aggr = torch.mean(motion_all, dim=1)
        elif self.aggr_method == "max":
            motion_aggr = torch.max(motion_all, dim=1)[0]
        else:
            raise NotImplementedError
        motion_aggr = torch.nn.functional.normalize(motion_aggr, dim=1)
        pred_mask = self.masknet(data.pos, motion_aggr, tpl_edge_index, geo_edge_index, batch)
        return motion_all, motion_aggr, pred_mask


class SkinNet_inner(torch.nn.Module):
    def __init__(self, nearest_bone, use_Dg, use_Lf, motion_dim, use_motion, aggr='max'):
        super(SkinNet_inner, self).__init__()
        self.use_Dg = use_Dg
        self.use_Lf = use_Lf
        self.num_nearest_bone = nearest_bone
        
        if self.use_Dg and self.use_Lf:
            input_dim = 3 + self.num_nearest_bone * 8
        elif self.use_Dg and not self.use_Lf:
            input_dim = 3 + self.num_nearest_bone * 7
        elif self.use_Lf and not self.use_Dg:
            input_dim = 3 + self.num_nearest_bone * 7
        else:
            input_dim = 3 + self.num_nearest_bone * 6
        
        self.gcu1 = GCUMotion(in_channels=motion_dim, out_channels=256, in_channel_pos=input_dim, dim_pos_feat=64, aggr=aggr)
        self.gcu2 = GCUMotion(in_channels=256, out_channels=256, in_channel_pos=input_dim, dim_pos_feat=64, aggr=aggr)
        self.gcu3 = GCUMotion(in_channels=256, out_channels=256, in_channel_pos=input_dim, dim_pos_feat=64, aggr=aggr)
        self.multi_layer_tranform2 = MLP([256, 512, 1024])
        self.cls_branch = Sequential(MLP([1024 + 256, 1024, 512]), Linear(512, self.num_nearest_bone))
        
    def forward(self, data, motion):
        samples = data.skin_input
        if self.use_Dg and self.use_Lf:
            samples = samples[:, 0: 8 * self.num_nearest_bone]
        elif self.use_Dg and not self.use_Lf:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 7]
            samples = samples[:, 0: 7 * self.num_nearest_bone]
        elif self.use_Lf and not self.use_Dg:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 6]
            samples = samples[:, 0: 7 * self.num_nearest_bone]
        else:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 7]
            samples = samples[:, np.arange(samples.shape[1]) % 7 != 6]
            samples = samples[:, 0: 6 * self.num_nearest_bone]
            
        raw_input = torch.cat([data.pos, samples], dim=1)
        x_1 = self.gcu1(raw_input, motion, data.tpl_edge_index, data.geo_edge_index)
        x_global = self.multi_layer_tranform2(x_1)
        x_global, _ = scatter_max(x_global, data.batch, dim=0)
        x_2 = self.gcu2(raw_input, x_1, data.tpl_edge_index, data.geo_edge_index)
        x_3 = self.gcu3(raw_input, x_2, data.tpl_edge_index, data.geo_edge_index)
        x_global = torch.repeat_interleave(x_global, torch.bincount(data.batch), dim=0)
        x_4 = torch.cat([x_3, x_global], dim=1)
        skin_cls_pred = self.cls_branch(x_4)
        return skin_cls_pred
    

class SkinMotion(torch.nn.Module):
    def __init__(self, nearest_bone, use_Dg, use_Lf, num_keyframes, use_motion, motion_dim, aggr='max'):
        super(SkinMotion, self).__init__()
        self.num_keyframes = num_keyframes
        self.motion_dim = motion_dim
        self.motionNet = GCNRig(chn_feature=3, chn_output=motion_dim, aggr=aggr)
        self.aggragator = TemporalAttn(input_size=motion_dim, num_heads=2, hidden_size=64, dim_feedforward=512, output_size=motion_dim)
        self.skinNet = SkinNet_inner(nearest_bone, use_Dg, use_Lf, motion_dim, use_motion, aggr)
        
    def forward(self, data, input_flow):
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch
        motion_all = []
        for t in range(self.num_keyframes):
            motion_t = self.motionNet(data.pos, input_flow[:, 3 * t:3 * t + 3], tpl_edge_index, geo_edge_index, batch)
            motion_t = torch.nn.functional.normalize(motion_t, dim=1)
            motion_all.append(motion_t)
        motion_all = torch.stack(motion_all, dim=1)
        motion_aggr = self.aggragator(motion_all)
        motion_aggr = torch.nn.functional.normalize(motion_aggr, dim=1)
        skin_cls_pred = self.skinNet(data, motion_aggr)
        return motion_all, motion_aggr, skin_cls_pred


def jointnet_motion(**kwargs):
    model = JointNetMotion(num_keyframes=kwargs["num_keyframes"], chn_output=kwargs["chn_output"], aggr_method=kwargs["aggr_method"])
    return model

def masknet_motion(**kwargs):
    model = MaskNetMotion(num_keyframes=kwargs["num_keyframes"], chn_output=kwargs["chn_output"], aggr_method=kwargs["aggr_method"])
    return model

def skinnet_motion(**kwargs):
    model = SkinMotion(nearest_bone=kwargs["nearest_bone"], use_Dg=kwargs["use_Dg"], 
                       use_Lf=kwargs["use_Lf"], num_keyframes=kwargs["num_keyframes"], 
                       use_motion=kwargs["use_motion"], motion_dim=kwargs["motion_dim"])
    return model