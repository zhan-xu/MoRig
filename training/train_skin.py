import sys
sys.path.append("./")

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os, shutil, argparse, numpy as np
import models
from models.customized_losses import cross_entropy_with_probs, skin_difference_loss, multi_positive_infonce_skinning, multi_pos_infoNCE, log_ratio_loss
from datasets.dataset_rig import RigDataset
from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_rigging
from utils.rig_parser import Rig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def get_bone_names(skin_filename):
    with open(skin_filename, 'r') as fin:
        lines = fin.readlines()
    bone_names = []
    for li in lines:
        words = li.strip().split()
        if words[0] == 'bones':
            bone_names.append([words[1], words[2]])
    return bone_names

def post_filter(skin_weights, topology_edge, num_ring=1):
    skin_weights_new = np.zeros_like(skin_weights)
    for v in range(len(skin_weights)):
        adj_verts_multi_ring = []
        current_seeds = [v]
        for r in range(num_ring):
            adj_verts = []
            for seed in current_seeds:
                adj_edges = topology_edge[:, np.argwhere(topology_edge == seed)[:, 1]]
                adj_verts_seed = list(set(adj_edges.flatten().tolist()))
                adj_verts_seed.remove(seed)
                adj_verts += adj_verts_seed
            adj_verts_multi_ring += adj_verts
            current_seeds = adj_verts
        adj_verts_multi_ring = list(set(adj_verts_multi_ring))
        if v in adj_verts_multi_ring:
            adj_verts_multi_ring.remove(v)
        skin_weights_neighbor = [skin_weights[int(i), :][np.newaxis, :] for i in adj_verts_multi_ring]
        skin_weights_neighbor = np.concatenate(skin_weights_neighbor, axis=0)
        #max_bone_id = np.argmax(skin_weights[v, :])
        #if np.sum(skin_weights_neighbor[:, max_bone_id]) < 0.17 * len(skin_weights_neighbor):
        #    skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
        #else:
        #    skin_weights_new[v, :] = skin_weights[v, :]
        skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)

    #skin_weights_new[skin_weights_new.sum(axis=1) == 0, :] = skin_weights[skin_weights_new.sum(axis=1) == 0, :]
    return skin_weights_new

def main(args):
    global device
    lowest_loss = 1e20

    # create checkpoint dir and log dir
    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)

    # create model
    model = models.__dict__[args.arch](nearest_bone=args.nearest_bone, 
                                       use_motion=True,
                                       use_Dg=args.Dg, use_Lf=args.Lf, 
                                       motion_dim=args.motion_dim, 
                                       num_keyframes=args.num_keyframes, 
                                       aggr_method=args.aggr_method)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(RigDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True)
    val_loader = DataLoader(RigDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False)
    test_loader = DataLoader(RigDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False)
    if args.evaluate:
        print('\nEvaluation only')
        test_losses = test(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        for loss_name, loss_value in test_losses.items():
            print(f"test_{loss_name}: {loss_value:6f}. ", end="")
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.start_epoch, args.epochs):
        lr = scheduler.get_last_lr()
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr[0]))
        train_losses = train(train_loader, model, optimizer, args)
        val_losses = test(val_loader, model, args)
        test_losses = test(test_loader, model, args)
        scheduler.step()
        losses = [train_losses, val_losses, test_losses]
        for split_id, split_name in enumerate(["train", "val", "test"]):
            print(f"Epoch{epoch + 1}. ", end="")
            for loss_name, loss_value in losses[split_id].items():
                print(f"{split_name}_{loss_name}: {loss_value:6f}. ", end="")
                logger.add_scalar(f"{split_name}_{loss_name}", loss_value, epoch + 1)
            print("")
        # remember best acc and save checkpoint
        is_best = val_losses["total_loss"] < lowest_loss
        lowest_loss = min(val_losses["total_loss"], lowest_loss)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss,
                         'optimizer': optimizer.state_dict()}, is_best, checkpoint=args.checkpoint)

def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_skin_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_meter = AverageMeter()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if np.random.uniform() > 0.5:
            input_flow = data.gt_flow
        else:
            input_flow = data.pred_flow
        motion_all, motion_aggr, skin_pred = model(data, input_flow)

        if args.loss_cont == "logratio":
            loss_embedding = 0.0
            for t in range(motion_all.shape[1]):
                loss_embedding += log_ratio_loss(motion_all[:, t, :], data.gt_skin, data.batch)
            loss_embedding += log_ratio_loss(motion_aggr, data.gt_skin, data.batch)
            loss_embedding = 0.1 * loss_embedding
        else:
            loss_embedding = 0.0
            for t in range(motion_all.shape[1]):
                loss_embedding += multi_pos_infoNCE(motion_all[:, t, :], data.gt_skin, data.batch)
            loss_embedding += multi_pos_infoNCE(motion_aggr, data.gt_skin, data.batch)
            loss_embedding = 0.01 * loss_embedding
            
        # skinning prediction loss
        skin_gt = data.skin_label[:, 0:args.nearest_bone]
        loss_mask_batch = data.loss_mask.float()[:, 0:args.nearest_bone]
        skin_gt = skin_gt * loss_mask_batch
        skin_gt = skin_gt / (torch.sum(torch.abs(skin_gt), dim=1, keepdim=True) + 1e-8)
        vert_mask = (torch.abs(skin_gt.sum(dim=1) - 1.0) < 1e-8).float()  # mask out vertices whose skinning is missing from the picked K bones.
        loss_skin = cross_entropy_with_probs(skin_pred, skin_gt, reduction='none')
        loss_skin = (loss_skin * loss_mask_batch * vert_mask.unsqueeze(1)).sum() / (loss_mask_batch * vert_mask.unsqueeze(1)).sum()

        loss = loss_skin + loss_embedding 
        loss.backward()
        optimizer.step()
        loss_skin_meter.update(loss_skin.item())
        loss_motion_meter.update(loss_embedding.item())
        loss_meter.update(loss.item())
    return {"loss_skin": loss_skin_meter.avg, "loss_motion": loss_motion_meter.avg, 
            "total_loss": loss_meter.avg}

def test(test_loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()  # switch to test mode
    loss_skin_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_meter = AverageMeter()
    outdir = args.checkpoint.split('/')[-1]
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            input_flow = data.pred_flow
            motion_all, motion_aggr, skin_pred = model(data, input_flow)

            if args.loss_cont == "logratio":
                loss_embedding = 0.0
                for t in range(motion_all.shape[1]):
                    loss_embedding += log_ratio_loss(motion_all[:, t, :], data.gt_skin, data.batch)
                loss_embedding += log_ratio_loss(motion_aggr, data.gt_skin, data.batch)
                loss_embedding = loss_embedding * 0.01
            else:
                loss_embedding = 0.0
                for t in range(motion_all.shape[1]):
                    loss_embedding += multi_pos_infoNCE(motion_all[:, t, :], data.gt_skin, data.batch)
                loss_embedding += multi_pos_infoNCE(motion_aggr, data.gt_skin, data.batch)
                loss_embedding = 0.1 * loss_embedding
            
            # skinning prediction loss
            skin_gt = data.skin_label[:, 0:args.nearest_bone]
            loss_mask_batch = data.loss_mask.float()[:, 0:args.nearest_bone]
            skin_gt = skin_gt * loss_mask_batch
            skin_gt = skin_gt / (torch.sum(torch.abs(skin_gt), dim=1, keepdim=True) + 1e-8)
            vert_mask = (torch.abs(skin_gt.sum(dim=1) - 1.0) < 1e-8).float()  # mask out vertices whose skinning is missing from the picked K bones.
            loss_skin = cross_entropy_with_probs(skin_pred, skin_gt, reduction='none')
            loss_skin = (loss_skin * loss_mask_batch * vert_mask.unsqueeze(1)).sum() / (loss_mask_batch * vert_mask.unsqueeze(1)).sum()

            loss = loss_skin + loss_embedding
            loss_skin_meter.update(loss_skin.item())
            loss_motion_meter.update(loss_embedding.item())
            loss_meter.update(loss.item())

            if save_result:
                output_folder = 'results/{:s}/best_{:d}/'.format(outdir, best_epoch)
                if not os.path.exists(output_folder):
                    mkdir_p(output_folder)
                for i in range(len(torch.unique(data.batch))):
                    print('output result for model {:d}'.format(data.name[i].item()))
                    skin_pred_i = skin_pred[data.batch == i]
                    bone_names = get_bone_names(os.path.join(args.test_folder, "{:d}_skin.txt".format(data.name[i].item())))
                    tpl_e = np.loadtxt(os.path.join(args.test_folder, "{:d}_tpl_e.txt".format(data.name[i].item()))).T
                    loss_mask_sample = data.loss_mask.float()[data.batch == i, 0:args.nearest_bone]
                    skin_pred_i = torch.softmax(skin_pred_i, dim=1)
                    skin_pred_i = skin_pred_i * loss_mask_sample
                    skin_nn_i = data.skin_nn[data.batch == i, 0:args.nearest_bone]
                    skin_pred_asarray = np.zeros((len(skin_pred_i), len(bone_names)))
                    for v in range(len(skin_pred_i)):
                        for nn_id in range(len(skin_nn_i[v, :])):
                            if loss_mask_sample[v, nn_id] == 1:
                                skin_pred_asarray[v, skin_nn_i[v, nn_id]] = skin_pred_i[v, nn_id]
                    skin_pred_asarray = post_filter(skin_pred_asarray, tpl_e, num_ring=1)
                    skin_pred_asarray[skin_pred_asarray < np.max(skin_pred_asarray, axis=1, keepdims=True) * 0.5] = 0.0
                    skin_pred_asarray = skin_pred_asarray / (skin_pred_asarray.sum(axis=1, keepdims=True) + 1e-10)
                    with open(os.path.join(output_folder, "{:d}_bone_names.txt".format(data.name[i].item())), 'w') as fout:
                        for bone_name in bone_names:
                            fout.write("{:s} {:s}\n".format(bone_name[0], bone_name[1]))
                    np.save(os.path.join(output_folder, "{:d}_full_pred.npy".format(data.name[i].item())), skin_pred_asarray)
                    skel_filename = os.path.join(args.info_folder, "{:d}.txt".format(data.name[i].item()))
                    output_rigging(skel_filename, skin_pred_asarray, output_folder, data.name[i].item())       
                            
    return {"loss_skin": loss_skin_meter.avg, "loss_motion": loss_motion_meter.avg, 
            "total_loss": loss_meter.avg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='skinnet network')
    parser.add_argument('--arch', default='skinnet_motion')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    parser.add_argument('--train_batch', default=2, type=int, help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int, help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/train/',
                        type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/val/',
                        type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/test/', type=str, help='folder of testing data')
    parser.add_argument('--nearest_bone', type=int, default=5)
    parser.add_argument('--info_folder',default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/rig_info_remesh/', type=str, help='folder of rig information')
    parser.add_argument('--loss_cont', type=str, default="infonce")  # logratio, infonce
    parser.add_argument('--Dg', action='store_true', help='input inverset geodesic as addtional feature')
    parser.add_argument('--Lf', action='store_true', help='input isleaf indicator as addtional feature')
    parser.add_argument('--num_keyframes', default=5, type=int)
    parser.add_argument('--aggr_method', default="attn", type=str, choices=["max", "mean", "attn"])
    parser.add_argument('--motion_dim', default=32, type=int)
    print(parser.parse_args())
    main(parser.parse_args())
