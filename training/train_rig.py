import sys
sys.path.append("./")
import os, shutil, argparse, numpy as np
from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_point_cloud_ply

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset_rig import RigDataset
import models
from models.customized_losses import chamfer_distance_with_average, multi_pos_infoNCE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def pairwise_distances(x, y):
    #Input: x is a Nxd matrix
    #       y is an optional Mxd matirx
    #Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    #        if y is not given then use 'y=x'.
    #i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def meanshift_cluster(pts, bandwidth, weights, args):
    """
    meanshift written in pytorch
    :param pts: input points
    :param weights: weight per point during clustering
    :return: clustered points
    """
    pts_steps = []
    for i in range(args.meanshift_step):
        Y = pairwise_distances(pts, pts)
        K = torch.nn.functional.relu(bandwidth ** 2 - Y)
        if weights is not None:
            K = K * weights
        P = torch.nn.functional.normalize(K, p=1, dim=0, eps=1e-10)
        P = P.transpose(0, 1)
        pts = args.step_size * (torch.matmul(P, pts) - pts) + pts
        pts_steps.append(pts)
    return pts_steps


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
    if args.arch == "jointnet_motion":
        chn_output = 3
    elif args.arch == "masknet_motion":
        chn_output = 1
    else:
        raise NotImplementedError
    model = models.__dict__[args.arch](chn_output=chn_output, motion_dim=args.motion_dim, 
                                       num_keyframes=args.num_keyframes, aggr_method=args.aggr_method)
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
    train_loader = DataLoader(RigDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True, follow_batch=['joints'])
    val_loader = DataLoader(RigDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(RigDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
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
    loss_chamfer_meter = AverageMeter()
    loss_l1_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_bce_meter = AverageMeter()
    loss_meter = AverageMeter()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if args.arch == 'masknet_motion':
            if np.random.uniform() > 0.5:
                input_flow = data.gt_flow
            else:
                input_flow = data.pred_flow
            motion_all, motion_aggr, mask_pred = model(data, input_flow)
            loss_embedding = 0.0
            for t in range(motion_all.shape[1]):
                loss_embedding += multi_pos_infoNCE(motion_all[:, t, :], data.gt_skin, data.batch)
            loss_embedding += multi_pos_infoNCE(motion_aggr, data.gt_skin, data.batch)
            loss_mask = torch.nn.functional.binary_cross_entropy_with_logits(mask_pred, data.mask.float().unsqueeze(1), reduction='mean')
            loss = 0.1 * loss_embedding + loss_mask
            loss_motion_meter.update(0.1 * loss_embedding.item())
            loss_bce_meter.update(loss_mask.item())
       
        elif args.arch == 'jointnet_motion':
            if np.random.uniform() > 0.5:
                input_flow = data.gt_flow
            else:
                input_flow = data.pred_flow
            motion_all, motion_aggr, data_displacement = model(data, input_flow)
            data_displacement = torch.tanh(data_displacement)
            y_pred = data_displacement + data.pos
            
            loss_embedding = 0.0
            for t in range(motion_all.shape[1]):
                loss_embedding += multi_pos_infoNCE(motion_all[:, t, :], data.gt_skin, data.batch)
            loss_embedding += multi_pos_infoNCE(motion_aggr, data.gt_skin, data.batch)
            
            loss_chamfer = 0.0
            for i in range(len(torch.unique(data.joints_batch))):
                joint_gt = data.joints[data.joints_batch == i, :]
                y_pred_i = y_pred[data.batch == i, :]
                loss_chamfer += chamfer_distance_with_average(y_pred_i.unsqueeze(0), joint_gt.unsqueeze(0))
            loss_chamfer /= len(torch.unique(data.joints_batch))
            loss_l1 = torch.nn.functional.l1_loss(data_displacement, data.offsets)
            loss = 0.1 * loss_embedding + loss_chamfer + loss_l1
            loss_motion_meter.update(0.1 * loss_embedding.item())
            loss_chamfer_meter.update(loss_chamfer.item())
            loss_l1_meter.update(loss_l1.item())
            
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    return {"loss_chamfer": loss_chamfer_meter.avg, "loss_l1": loss_l1_meter.avg,
            "loss_motion": loss_motion_meter.avg,  "loss_bce": loss_bce_meter.avg,
            "total_loss": loss_meter.avg}


def test(test_loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()  # switch to test mode
    loss_chamfer_meter = AverageMeter()
    loss_l1_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_bce_meter = AverageMeter()
    loss_meter = AverageMeter()
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            if args.arch == 'masknet_motion':
                input_flow = data.pred_flow
                motion_all, motion_aggr, mask_pred = model(data, input_flow)
                loss_embedding = 0.0
                for t in range(motion_all.shape[1]):
                    loss_embedding += multi_pos_infoNCE(motion_all[:, t, :], data.gt_skin, data.batch)
                loss_embedding += multi_pos_infoNCE(motion_aggr, data.gt_skin, data.batch)
                loss_mask = torch.nn.functional.binary_cross_entropy_with_logits(mask_pred, data.mask.float().unsqueeze(1), reduction='mean')
                loss = 0.1 * loss_embedding + loss_mask
                loss_motion_meter.update(0.1 * loss_embedding.item())
                loss_bce_meter.update(loss_mask.item())
          
            elif args.arch == 'jointnet_motion':
                input_flow = data.pred_flow
                motion_all, motion_aggr, data_displacement = model(data, input_flow)
                data_displacement = torch.tanh(data_displacement)
                y_pred = data_displacement + data.pos
                
                loss_embedding = 0.0
                for t in range(motion_all.shape[1]):
                    loss_embedding += multi_pos_infoNCE(motion_all[:, t, :], data.gt_skin, data.batch)
                loss_embedding += multi_pos_infoNCE(motion_aggr, data.gt_skin, data.batch)
                
                loss_chamfer = 0.0
                for i in range(len(torch.unique(data.joints_batch))):
                    joint_gt = data.joints[data.joints_batch == i, :]
                    y_pred_i = y_pred[data.batch == i, :]
                    loss_chamfer += chamfer_distance_with_average(y_pred_i.unsqueeze(0), joint_gt.unsqueeze(0))
                loss_chamfer /= len(torch.unique(data.joints_batch))
                loss_l1 = torch.nn.functional.l1_loss(data_displacement, data.offsets)
                loss = 0.1 * loss_embedding + loss_chamfer + loss_l1
                loss_motion_meter.update(0.1 * loss_embedding.item())
                loss_chamfer_meter.update(loss_chamfer.item())
                loss_l1_meter.update(loss_l1.item())
                
            else:
                raise NotImplementedError
            
            loss_meter.update(loss.item())
            if save_result:
                output_folder = args.output_folder
                if not os.path.exists(output_folder):
                    mkdir_p(output_folder)
                if args.arch == 'masknet_motion':
                    mask_pred = torch.sigmoid(mask_pred)
                    for i in range(len(torch.unique(data.batch))):
                        mask_pred_sample = mask_pred[data.batch == i]
                        motion_embedding_sample = motion_aggr[data.batch == i]
                        np.save(os.path.join(output_folder, str(data.name[i].item()) + '_attn.npy'),
                                mask_pred_sample.data.to("cpu").numpy())
                        # np.save(os.path.join(output_folder, str(data.name[i].item()) + '_embedding.npy'),
                        #         motion_embedding_sample.data.to("cpu").numpy())
                else:
                    for i in range(len(torch.unique(data.batch))):
                        y_pred_sample = y_pred[data.batch == i, :]
                        output_point_cloud_ply(y_pred_sample, name=str(data.name[i].item()),
                                               output_folder=args.output_folder)
    return {"loss_chamfer": loss_chamfer_meter.avg, "loss_l1": loss_l1_meter.avg,
            "loss_motion": loss_motion_meter.avg, "loss_bce": loss_bce_meter.avg,
            "total_loss": loss_meter.avg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rigging network')
    parser.add_argument('--arch', default='jointnet_motion', choices=['jointnet_motion', 'masknet_motion'])
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    parser.add_argument('--train_batch', default=2, type=int, help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int,  help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/train/', type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/val/', type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/test/', type=str, help='folder of testing data')
    parser.add_argument('--num_keyframes', default=5, type=int)
    parser.add_argument('--aggr_method', default="attn", type=str, choices=["max", "mean", "attn"])
    parser.add_argument('--motion_dim', default=32, type=int)
    parser.add_argument('--output_folder', default='results/our_results', type=str)
    print(parser.parse_args())
    main(parser.parse_args())
