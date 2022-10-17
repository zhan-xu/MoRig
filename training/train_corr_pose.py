import sys
sys.path.append("./")
import os, glob, shutil, argparse, numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import save_checkpoint

import models
from datasets.dataset_pose import *
from models.customized_losses import infoNCE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_corr_meter = AverageMeter()
    loss_mask_meter = AverageMeter()
    loss_meter = AverageMeter()
    for data in train_loader:
        if args.sequential_frame:
            if args.dataset == "modelsresource":
                src_frame = np.random.choice(19)
                tar_frame_candidates = np.arange(max(0, src_frame-2), src_frame+2).tolist()
            elif args.dataset == "deformingthings":
                src_frame = np.random.choice(18)
                tar_frame_candidates = np.arange(max(0, src_frame-3), src_frame+3).tolist()
            else:
                raise NotImplementedError
        else:
            if args.dataset == "modelsresource":
                src_frame = 0
                tar_frame_candidates = np.arange(1, 6).tolist()
            elif args.dataset == "deformingthings":
                src_frame = np.random.choice(6)
                tar_frame_candidates = np.arange(6).tolist()
            else:
                raise NotImplementedError
        tar_frame_candidates = [tid for tid in tar_frame_candidates if tid != src_frame]
        tar_frame = np.random.choice(tar_frame_candidates)
        data.vtx = data.vtx_traj[:, 3*src_frame:3*src_frame+3]
        data.vtx_batch = data.vtx_traj_batch
        data.pts = data.pts_traj[:, 3*tar_frame:3*tar_frame+3]
        data.pts_batch = data.pts_traj_batch
        corr_v2p_id_tar = (data.corr_v2p_all[:,-1]==tar_frame)
        data.corr_v2p = data.corr_v2p_all[corr_v2p_id_tar, 0:-1]
        data.corr_v2p_batch = data.corr_v2p_all_batch[corr_v2p_id_tar]
        corr_p2v_id_tar = (data.corr_p2v_all[:,-1]==tar_frame)
        data.corr_p2v = data.corr_p2v_all[corr_p2v_id_tar, 0:-1]
        data.corr_p2v_batch = data.corr_p2v_all_batch[corr_p2v_id_tar]
        data.vismask = data.vismask_all[:, tar_frame]

        data = data.to(device)
        optimizer.zero_grad()
        vtx_feature, pts_feature, pred_vismask, temprature = model(data, args.train_vismask)
        
        loss_match = infoNCE(vtx_feature, pts_feature, data.corr_v2p, data.corr_p2v, data.vtx_batch, data.pts_batch, 
                             data.corr_v2p_batch, data.corr_p2v_batch, tau=temprature)
        if args.train_vismask:
            loss_mask = torch.nn.functional.binary_cross_entropy_with_logits(pred_vismask, data.vismask[:, None].float())
        else:
            loss_mask = torch.zeros(1).to(device)
        loss = loss_match + 5 * loss_mask
        loss.backward()
        optimizer.step()
        loss_corr_meter.update(loss_match.item(), n=len(data.name))
        loss_mask_meter.update(loss_mask.item(), n=len(data.name))
        loss_meter.update(loss.item(), n=len(data.name))
    return {"corr_loss": loss_corr_meter.avg, "vis_loss": loss_mask_meter.avg, "total_loss": loss_meter.avg}


def test(test_loader, model, args, save_result=False):
    global device
    model.eval()  # switch to test mode
    loss_corr_meter = AverageMeter()
    loss_mask_meter = AverageMeter()
    loss_meter = AverageMeter()
    for data in test_loader:
        if args.sequential_frame:
            src_frame, tar_frame = 10, 11
        else:
            src_frame, tar_frame = 0, 3
        data.vtx = data.vtx_traj[:, 3 * src_frame:3 * src_frame + 3]
        data.vtx_batch = data.vtx_traj_batch
        data.pts = data.pts_traj[:, 3 * tar_frame:3 * tar_frame + 3]
        data.pts_batch = data.pts_traj_batch
        corr_v2p_id_tar = (data.corr_v2p_all[:, -1] == tar_frame)
        data.corr_v2p = data.corr_v2p_all[corr_v2p_id_tar, 0:-1]
        data.corr_v2p_batch = data.corr_v2p_all_batch[corr_v2p_id_tar]
        corr_p2v_id_tar = (data.corr_p2v_all[:, -1] == tar_frame)
        data.corr_p2v = data.corr_p2v_all[corr_p2v_id_tar, 0:-1]
        data.corr_p2v_batch = data.corr_p2v_all_batch[corr_p2v_id_tar]
        data.vismask = data.vismask_all[:, tar_frame]

        data = data.to(device)
        with torch.no_grad():
            vtx_feature, pts_feature, pred_vismask, temprature = model(data, args.train_vismask)
            loss_match = infoNCE(vtx_feature, pts_feature, data.corr_v2p, data.corr_p2v, data.vtx_batch, data.pts_batch,
                                 data.corr_v2p_batch, data.corr_p2v_batch, tau=temprature)
            if args.train_vismask:
                loss_mask = torch.nn.functional.binary_cross_entropy_with_logits(pred_vismask, data.vismask[:, None].float())
            else:
                loss_mask = torch.zeros(1).to(loss_match.device)
        if save_result:
            for i in range(len(data.name)):
                model_name = data.name[i]
                print("processing: ", model_name)
                vtx_feature_np = vtx_feature[data.vtx_batch == i].to("cpu").numpy()
                pts_feature_np = pts_feature[data.pts_batch == i].to("cpu").numpy()
                corr_v2p_np = data.corr_v2p[data.corr_v2p_batch == i].to("cpu").numpy()
                similarity = np.matmul(vtx_feature_np,  pts_feature_np.T) / temprature.item()
                pairwise_nnind = np.argmax(similarity, axis=1)
                pts_i = data.pts[data.pts_batch == i].to("cpu").numpy()
                vtx_i = data.vtx[data.vtx_batch == i].to("cpu").numpy()
                # np.save(f"/mnt/DATA_LINUX/zhan/output/corrnet/{model_name}_gt_corr.npy", corr_v2p_np)
                # np.save(f"/mnt/DATA_LINUX/zhan/output/corrnet/{model_name}_nnind.npy", pairwise_nnind)
                # np.save(f"/mnt/DATA_LINUX/zhan/output/corrnet/{model_name}_pts.npy", pts_i)
                # np.save(f"/mnt/DATA_LINUX/zhan/output/corrnet/{model_name}_vtx.npy", vtx_i)
                if args.train_vismask:
                    pred_vismask_i = torch.sigmoid(pred_vismask[data.vtx_batch == i])
                    pred_vismask_np = pred_vismask_i.to("cpu").numpy().squeeze(axis=1)
                    gt_vismask_np = data.vismask[data.vtx_batch == i].to("cpu").numpy()
                    # np.save(f"/mnt/DATA_LINUX/zhan/output/corrnet/{model_name}_pred_vismask.npy", pred_vismask_np)
                    # np.save(f"/mnt/DATA_LINUX/zhan/output/corrnet/{model_name}_gt_vismask.npy", gt_vismask_np)
        loss = loss_match + 5*loss_mask
        loss_corr_meter.update(loss_match.item(), n=len(data.name))
        loss_mask_meter.update(loss_mask.item(), n=len(data.name))
        loss_meter.update(loss.item(), n=len(data.name))
    return {"corr_loss": loss_corr_meter.avg, "vis_loss": loss_mask_meter.avg, "total_loss": loss_meter.avg}


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
    model = models.__dict__[args.arch](input_feature=3, output_feature=args.output_feature, temprature=args.tau_nce)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.sequential_frame:
        if args.dataset == "deformingthings":
            train_loader = DataLoader(SeqDeformingThingsDataset(root=args.train_folder, transform=None),
                                      batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers, 
                                      follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            val_loader = DataLoader(SeqDeformingThingsDataset(root=args.val_folder, transform=None), 
                                    batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers, 
                                    follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            test_loader = DataLoader(SeqDeformingThingsDataset(root=args.test_folder, transform=None), 
                                     batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers, 
                                     follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
        elif args.dataset == "modelsresource":
            train_loader = DataLoader(SeqModelsResourcesDataset(root=args.train_folder, transform=None),
                                      batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers,
                                      follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            val_loader = DataLoader(SeqModelsResourcesDataset(root=args.val_folder, transform=None),
                                    batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
                                    follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            test_loader = DataLoader(SeqModelsResourcesDataset(root=args.test_folder, transform=None),
                                     batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
                                     follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
        else:
            raise NotImplementedError
    else:
        if args.dataset == "deformingthings":
            train_loader = DataLoader(DeformingThingsDataset(root=args.train_folder, transform=None), 
                                      batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers, 
                                      follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            val_loader = DataLoader(DeformingThingsDataset(root=args.val_folder, transform=None),
                                    batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
                                    follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            test_loader = DataLoader(DeformingThingsDataset(root=args.test_folder, transform=None),
                                    batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
                                    follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
        elif args.dataset == "modelsresource":
            train_loader = DataLoader(ModelsResourcesDataset(root=args.train_folder, transform=None),
                                      batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers,
                                      follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            val_loader = DataLoader(ModelsResourcesDataset(root=args.val_folder, transform=None),
                                    batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
                                    follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
            test_loader = DataLoader(ModelsResourcesDataset(root=args.test_folder, transform=None),
                                     batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
                                     follow_batch=['vtx_traj', 'pts_traj', 'corr_v2p_all', 'corr_p2v_all'])
        else:
            raise NotImplementedError
    if args.evaluate:
        print('\nEvaluation only')
        test_losses = test(test_loader, model, args, save_result=True)
        for loss_name, loss_value in test_losses.items():
            print(f"test_{loss_name}: {loss_value:6f}. ", end="")
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == args.vis_branch_start_epoch:
            args.train_vismask = True
            lowest_loss = 1e20
        lr = scheduler.get_last_lr()
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr[0]))
        train_losses = train(train_loader, model, optimizer, args)
        val_losses = test(val_loader, model, args)
        test_losses = test(test_loader, model, args)
        scheduler.step()
         # remember best acc and save checkpoint
        is_best = val_losses["total_loss"] < lowest_loss
        if is_best:
            best_epoch = epoch
        lowest_loss = min(val_losses["total_loss"], lowest_loss)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss,
                         'optimizer': optimizer.state_dict()},
                        is_best, checkpoint=args.checkpoint)
        losses = [train_losses, val_losses, test_losses]
        for split_id, split_name in enumerate(["train", "val", "test"]):
            print(f"Epoch{epoch + 1}. ", end="")
            for loss_name, loss_value in losses[split_id].items():
                print(f"{split_name}_{loss_name}: {loss_value:6f}. ", end="")
                logger.add_scalar(f"{split_name}_{loss_name}", loss_value, epoch + 1)
            print("")
    print("Best epoch:", best_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mesh depth corresponce')
    parser.add_argument('--arch', default='corrnet')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers to load data')

    parser.add_argument('--train_batch', default=2, type=int, help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int, help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

    parser.add_argument('--train_folder', default='/mnt/DATA_LINUX2/zhan/morig/DeformingThings4D/train/',
                        type=str, help='folder of training data')  # /mnt/neghvar
    parser.add_argument('--val_folder', default='/mnt/DATA_LINUX2/zhan/morig/DeformingThings4D/val/',
                        type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/mnt/DATA_LINUX2/zhan/morig/DeformingThings4D/test/',
                        type=str, help='folder of testing data')

    parser.add_argument('--output_feature', default=64, type=str, help='chn number of output feature')
    parser.add_argument('--tau_nce', default=0.07, type=float, help='temprature in hardest nce loss')
    parser.add_argument('--train_vismask', action='store_true', help='whether or not to train mask branch')
    parser.add_argument('--vis_branch_start_epoch', default=100, type=int)
    parser.add_argument('--sequential_frame', action='store_true')
    parser.add_argument('--dataset', default='modelsresource', choices=['deformingthings', 'modelsresource'])
    print(parser.parse_args())
    main(parser.parse_args())
