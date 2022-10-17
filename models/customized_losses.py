import numpy as np
from sympy import im
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_scatter import scatter_max
from torch_cluster import fps
from itertools import combinations


def log_ratio_loss(pred_feature, gt_skin, batch):
    num_sample = 50
    epsilon = 1e-6
    pairs = np.array(list(combinations(np.arange(num_sample), 2)))
    
    feature_sample = []
    gt_skin_sample = []
    for i in range(len(torch.unique(batch))):
        sample_ids = np.random.choice((batch == i).sum().item(), num_sample, replace=False)
        feature_i = pred_feature[batch == i][sample_ids]
        gt_skin_i = gt_skin[batch == i][sample_ids]
        feature_sample.append(feature_i)
        gt_skin_sample.append(gt_skin_i)
    feature_sample = torch.stack(feature_sample, dim=0)
    gt_skin_sample = torch.stack(gt_skin_sample, dim=0)
    
    dist = torch.sum((feature_sample[:, pairs[:, 0], None, :] - feature_sample[:, None, pairs[:, 1], :])**2, dim=-1)
    gt_dist = torch.sum((gt_skin_sample[:, pairs[:, 0], None, :] - gt_skin_sample[:, None, pairs[:, 1], :])**2, dim=-1)
    
    log_dist = torch.log(dist + epsilon)
    log_gt_dist = torch.log(gt_dist + epsilon)
    
    diff_log_dist = log_dist.permute((0, 2, 1)) - log_dist
    diff_log_gt_dist = log_gt_dist.permute((0, 2, 1)) - log_gt_dist
    log_ratio_loss = (diff_log_dist - diff_log_gt_dist).pow(2)
    
    # uniform weight coefficients 
    idxs = torch.arange(len(pairs)).cuda()
    indc = idxs.repeat(len(pairs),1).t() < idxs.repeat(len(pairs), 1)
    wgt = indc.clone().float()
    wgt = wgt.div(wgt.sum())

    loss = log_ratio_loss.mul(wgt).sum()
    return loss  / len(torch.unique(batch))      
    
    
def hungarian_matching(pred_seg, gt_seg):
    interset = np.matmul(pred_seg.T, gt_seg)
    matching_cost = 1-np.divide(interset, np.expand_dims(np.sum(pred_seg,0), 1)+np.sum(gt_seg, axis=0, keepdims=True)-interset+1e-8)
    row_ind, col_ind = linear_sum_assignment(matching_cost)
    return np.vstack((row_ind, col_ind))


def motionLoss(pred_Rs, pred_ts, xyz, gt_flow, gt_seg):
    """ pred_Rs: B x nsmp x 3 x 3,
        pred_ts: B x nsmp x 1 x 3,
        xyz: B x nsmp x 3,
        gt_flow: B x nsmp x 3
        gt_seg: B x nsmp x nstep """
    ppdist = torch.unsqueeze(xyz, 1) - torch.unsqueeze(xyz, 2) # B x nsmp x nsmp x 3
    ppdist = torch.matmul(ppdist, pred_Rs) + pred_ts + torch.unsqueeze(gt_flow, 2) # B x nsmp x nsmp x 3
    loss_motion = ppdist-torch.unsqueeze(gt_flow,1)
    gt_seg = torch.sum(torch.mul(torch.unsqueeze(gt_seg, 2), torch.unsqueeze(gt_seg, 1)), dim=-1)
    gt_seg_normalized = torch.div(gt_seg, gt_seg.sum(dim=2, keepdim=True)+1e-8)
    loss_motion = torch.sum(torch.square(loss_motion), dim=-1) # B x nsmp x nsmp
    loss_motion = torch.mul(loss_motion, gt_seg_normalized)
    loss_motion = torch.div(torch.sum(loss_motion), torch.sum(gt_seg_normalized))
    return loss_motion


def groupingLoss(pred_support_matrix, seg_sub):
    """ pred_support_matrix: B x nsmp x nsmp,
        gt_seg: B x nsmp x nstep """
    gt_seg = torch.sum(torch.mul(torch.unsqueeze(seg_sub, 2), torch.unsqueeze(seg_sub, 1)), dim=-1)
    loss_group = torch.nn.functional.binary_cross_entropy_with_logits(pred_support_matrix, gt_seg.float())
    return loss_group


def iouLoss(pred_seg, gt_seg, batch):
    """
    pred_seg: B, nsmp
    gt_seg: B,
    batch: B
    """
    pred_seg_np = pred_seg.data.to("cpu").numpy()
    gt_seg_np = gt_seg.data.to("cpu").numpy()
    batch_np = batch.data.to("cpu").numpy()
    loss = 0.0
    for i in range(len(torch.unique(batch))):
        gt_seg_i = gt_seg[batch==i]
        pred_seg_np_i = pred_seg_np[batch_np==i]
        gt_seg_np_i = gt_seg_np[batch_np==i]
        gt_seg_expand_i = torch.zeros((len(gt_seg_np_i), np.max(gt_seg_np_i)+1)).long().to(pred_seg.device)
        gt_seg_expand_np_i = np.zeros((len(gt_seg_np_i), np.max(gt_seg_np_i)+1), dtype=np.int)
        np.put_along_axis(gt_seg_expand_np_i, indices=gt_seg_np_i[:, None], values=1, axis=1)
        gt_seg_expand_i.scatter_(dim=1, index=gt_seg_i[:, None], src=torch.ones_like(gt_seg_i[:, None]))
        matching_id_i = hungarian_matching(pred_seg_np_i, gt_seg_expand_np_i)

        pred_seg_i_reorder = pred_seg[batch==i][:, matching_id_i[0]]
        gt_seg_i_reorder = gt_seg_expand_i[:, matching_id_i[1]]
        interset = torch.sum(pred_seg_i_reorder * gt_seg_i_reorder, dim=0)
        cost_i = 1 - torch.div(interset, pred_seg_i_reorder.sum(dim=0) + gt_seg_i_reorder.sum(dim=0) - interset + 1e-8)
        loss = loss + cost_i.mean()
    return loss / len(torch.unique(batch))


def infoNCE(vtx_feature, pts_feature, corr_v2p, corr_p2v, vtx_batch, pts_batch, corr_v2p_batch, corr_p2v_batch, tau):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss = 0.0
    for i in range(len(torch.unique(vtx_batch))):
        vtx_feature_i = vtx_feature[vtx_batch == i]
        pts_feature_i = pts_feature[pts_batch == i]
        # v2p
        corr_v2p_i = corr_v2p[corr_v2p_batch == i]
        if len(corr_v2p_i) == 0:
            loss += 0.0
            continue
        anchor = vtx_feature_i[corr_v2p_i[:, 0]]
        prod = torch.mm(anchor, pts_feature_i.T) / tau
        label_i = corr_v2p_i[:, 1]
        loss_i = cross_entropy_loss(prod, label_i)
        loss += loss_i.mean()
        
        # p2v
        corr_p2v_i = corr_p2v[corr_p2v_batch == i]
        if len(corr_p2v_i) == 0:
            loss += 0.0
            continue
        anchor = pts_feature_i[corr_p2v_i[:, 0]]
        prod = torch.mm(anchor, vtx_feature_i.T) / tau
        label_i = corr_p2v_i[:, 1]
        loss_i = cross_entropy_loss(prod, label_i)
        loss += loss_i.mean()
    return loss / len(torch.unique(vtx_batch))


def multi_pos_infoNCE(pred_feature, gt_skin, batch):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = 0.0
    for i in range(len(torch.unique(batch))):
        sample_ids = np.random.choice((batch == i).sum().item(), 512, replace=False)
        feature_i = pred_feature[batch == i][sample_ids]
        gt_skin_i = gt_skin[batch == i][sample_ids]

        gt_sim = (2 - torch.sum(torch.abs(gt_skin_i[None] - gt_skin_i[:, None]), axis=-1)) / 2.0
        gt_sim = (gt_sim > 0.9).float()

        pos_ids = torch.multinomial(gt_sim, 10, replacement=True)
        neg_ids = torch.multinomial(1 - gt_sim, 200, replacement=True)

        prod = torch.mm(feature_i, feature_i.T)
        prod_neg = torch.gather(prod, dim=1, index=neg_ids)
        loss_i = 0.0
        for j in range(10):
            prod_pos = torch.gather(prod, dim=1, index=pos_ids[:, j][:, None])
            loss_i += cross_entropy_loss(torch.cat((prod_pos, prod_neg), dim=1), torch.zeros(512).long().to(pred_feature.device))
        loss = loss + loss_i / 10
    return loss / len(torch.unique(batch))


def hingeLoss(pred_feature, gt_label, batch_sub):
    loss = 0.0
    for i in range(len(torch.unique(batch_sub))):
        sample_ids = np.random.choice((batch_sub == i).sum().item(), 256, replace=False)
        gt_label_i = gt_label[batch_sub == i][sample_ids]
        #gt_seg_expand_i = torch.nn.functional.one_hot(gt_label_i, num_classes=gt_label_i.max()+1)
        pred_sim = torch.matmul(pred_feature[batch_sub == i][sample_ids], pred_feature[batch_sub==i][sample_ids].transpose(0, 1))
        pred_dist = (1 - pred_sim) / 2.0
        #gt_sim = torch.matmul(gt_seg_expand_i.float(),  gt_seg_expand_i.transpose(0, 1).float())
        gt_sim = (2 - torch.sum(torch.abs(gt_label_i[None] - gt_label_i[:, None]), axis=-1)) / 2.0
        gt_sim = (gt_sim > 0.9).float()
        weight = 10 * gt_sim
        weight[weight == 0] = 1
        gt_sim[gt_sim == 0] = -1
        #loss += torch.nn.functional.hinge_embedding_loss(pred_dist, gt_sim, margin=0.2)
        loss_i = torch.nn.functional.hinge_embedding_loss(pred_dist, gt_sim, margin=0.2, reduction="none")
        loss_i = loss_i * weight
        loss = loss + (loss_i * weight).sum() / weight.sum()
    return loss / len(torch.unique(batch_sub))


def transLoss(adj_matrix, gt_seg, batch):
    """
    adj_matrix: BKxKxT
    gt_seg: BK
    batch: BK
    """
    loss = 0.0
    for i in range(len(torch.unique(batch))):
        gt_seg_i = gt_seg[batch==i]
        adj_matrix_i = adj_matrix[batch==i]
        gt_seg_expand_i = torch.nn.functional.one_hot(gt_seg_i, num_classes=gt_seg_i.max()+1)
        gt_sim = torch.matmul(gt_seg_expand_i.float(),  gt_seg_expand_i.transpose(0, 1).float())
        idx_same_part = torch.nonzero(gt_sim, as_tuple=False)
        dist_i = adj_matrix_i[idx_same_part[:, 0], idx_same_part[:, 1], :]
        loss += dist_i.mean()
    return loss / len(torch.unique(batch))


def multiLableBCE(feature_in, gt_seg, batch, tau=0.05):
    cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss = 0.0
    for i in range(len(torch.unique(batch))):
        feature_i = feature_in[batch==i]
        prod = torch.mm(feature_i, feature_i.T)  / tau

        gt_seg_i = gt_seg[batch==i]
        gt_seg_expand_i = torch.nn.functional.one_hot(gt_seg_i, num_classes=gt_seg_i.max()+1)
        gt_sim = torch.matmul(gt_seg_expand_i.float(),  gt_seg_expand_i.transpose(0, 1).float())
        
        loss += cross_entropy_loss(prod, gt_sim)
    loss = loss / len(torch.unique(batch))
    return loss


def cross_entropy_with_probs(input, target, weight=None, reduction="mean"):
    input_logsoftmax = F.log_softmax(input, dim=1)
    cum_losses = -target * input_logsoftmax
    if weight is not None:
        cum_losses = cum_losses * weight
    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.sum(dim=1).mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def chamfer_distance_with_average(p1, p2):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)
    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist_norm = torch.norm(dist, 2, dim=2)
    dist1 = torch.min(dist_norm, dim=1)[0]
    dist2 = torch.min(dist_norm, dim=0)[0]
    loss = 0.5 * ((torch.mean(dist1)) + (torch.mean(dist2)))
    return loss


def skin_difference_loss(pred_skin, gt_skin, pos, batch):
    ids_sub = fps(pos, batch=batch, ratio=0.25, random_start=True)
    pred_skin_sub = pred_skin[ids_sub]
    gt_skin_sub = gt_skin[ids_sub]
    batch_sub = batch[ids_sub]
    loss = 0.0
    for i in range(len(torch.unique(batch_sub))):
        pred_skin_i = pred_skin_sub[batch_sub == i]
        gt_skin_i = gt_skin_sub[batch_sub == i]
        pred_diffmat = torch.sum(torch.abs(pred_skin_i[:, None, :] - pred_skin_i[None, ...]), dim=-1)
        gt_diffmat = torch.sum(torch.abs(gt_skin_i[:, None, :] - gt_skin_i[None, ...]), dim=-1)
        pred_diffmat = pred_diffmat * (torch.abs(gt_diffmat) < 1e-6).float()
        loss += pred_diffmat.mean()
    return loss / len(torch.unique(batch))


def multi_positive_infonce_skinning(pred_feature, gt_skin, batch):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = 0.0
    for i in range(len(torch.unique(batch))):
        sample_ids = np.random.choice((batch == i).sum().item(), 512, replace=False)
        feature_i = pred_feature[batch == i][sample_ids]
        gt_skin_i = gt_skin[batch == i][sample_ids]

        gt_sim = (2 - torch.sum(torch.abs(gt_skin_i[None] - gt_skin_i[:, None]), dim=-1)) / 2.0
        gt_sim = (gt_sim > 0.9).float()

        pos_ids = torch.multinomial(gt_sim, 10, replacement=True)
        neg_ids = torch.multinomial(1 - gt_sim, 200, replacement=True)

        pred_sim = (2 - torch.sum(torch.abs(feature_i[:, None, :] - feature_i[None, ...]), dim=-1)) / 2.0
        pred_neg = torch.gather(pred_sim, dim=1, index=neg_ids)
        loss_i = 0.0
        for j in range(10):
            pred_pos = torch.gather(pred_sim, dim=1, index=pos_ids[:, j][:, None])
            loss_i += cross_entropy_loss(torch.cat((pred_pos, pred_neg), dim=1), torch.zeros(512).long().to(pred_feature.device))
        loss = loss + loss_i / 10
    return loss / len(torch.unique(batch))