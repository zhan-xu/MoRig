import sys
sys.path.append("./")
import os, shutil, argparse, numpy as np, glob

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import save_checkpoint
from datasets.dataset_shape import ModelsResourcesShapeDataset
import models
from models.customized_losses import infoNCE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_flow_meter = AverageMeter()
    loss_match_meter = AverageMeter()
    loss_mask_meter = AverageMeter()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred_flow, vtx_feature, pts_feature, pred_vismask, tau_nce = model(data)
        loss_flow = torch.nn.functional.l1_loss(pred_flow, data.flow)
       
        if args.train_extractor:
            loss_match = infoNCE(vtx_feature, pts_feature, data.corr_v2p, data.corr_p2v, data.vtx_batch, data.pts_batch,
                                 data.corr_v2p_batch, data.corr_p2v_batch, tau=tau_nce)
            loss_mask = torch.nn.functional.binary_cross_entropy_with_logits(pred_vismask, data.vismask[:, None].float())
        else:
            loss_match = torch.zeros(1).to(loss_flow.device)
            loss_mask = torch.zeros(1).to(loss_flow.device)
        loss = loss_match + loss_mask + loss_flow
        loss.backward()
        optimizer.step()
        loss_flow_meter.update(loss_flow.item(), n=len(data.name))
        loss_match_meter.update(loss_match.item(), n=len(data.name))
        loss_mask_meter.update(loss_mask.item(), n=len(data.name))
    return {"flow_loss": loss_flow_meter.avg, "corr_loss": loss_match_meter.avg, "vis_loss": loss_mask_meter.avg}


def test(test_loader, model, args, save_result=False):
    global device
    model.eval()  # switch to test mode
    loss_flow_meter = AverageMeter()
    loss_match_meter = AverageMeter()
    loss_mask_meter = AverageMeter()
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred_flow, vtx_feature, pts_feature, pred_vismask, tau_nce = model(data)
            loss_flow = torch.nn.functional.mse_loss(pred_flow, data.flow)
            if args.train_extractor:
                loss_match = infoNCE(vtx_feature, pts_feature, data.corr_v2p, data.corr_p2v, data.vtx_batch, 
                                     data.pts_batch, data.corr_v2p_batch, data.corr_p2v_batch, tau=tau_nce)
                loss_mask = torch.nn.functional.binary_cross_entropy_with_logits(pred_vismask, data.vismask[:, None].float())
            else:
                loss_match = torch.zeros(1).to(loss_flow.device)
                loss_mask = torch.zeros(1).to(loss_flow.device)
        loss_flow_meter.update(loss_flow.item(), n=len(data.name))
        loss_match_meter.update(loss_match.item(), n=len(data.name))
        loss_mask_meter.update(loss_mask.item(), n=len(data.name))
        if save_result:
            for i in range(len(data.name)):
                model_name = data.name[i]
                print("processing: ", model_name)
                pred_flow_np = pred_flow[data.vtx_batch==i].to("cpu").numpy()
                gt_flow_np = data.flow[data.vtx_batch==i].to("cpu").numpy()
                pts_i = data.pts[data.pts_batch==i].to("cpu").numpy()
                vtx_i = data.vtx[data.vtx_batch==i].to("cpu").numpy()
                vtx_shift_i = vtx_i + pred_flow_np
                np.save("/mnt/neghvar/mnt/DATA_LINUX/zhan/output/mr_flownetG/{:s}_src_vtx.npy".format(model_name), vtx_i)
                np.save("/mnt/neghvar/mnt/DATA_LINUX/zhan/output/mr_flownetG/{:s}_shift_vtx.npy".format(model_name), vtx_shift_i)
                np.save("/mnt/neghvar/mnt/DATA_LINUX/zhan/output/mr_flownetG/{:s}_tar_pts.npy".format(model_name), pts_i)
    return {"flow_loss": loss_flow_meter.avg, "corr_loss": loss_match_meter.avg, "vis_loss": loss_mask_meter.avg}

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
    model = models.__dict__[args.arch](tau_nce=args.tau_nce, num_interp=args.num_interp)
    model.to(device)

    model.corr_extractor.load_state_dict(torch.load(args.init_extractor)['state_dict'])
    if not args.train_extractor:
        for name, param in model.corr_extractor.named_parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{"params": model.corr_extractor.parameters(), "lr": 1e-4},
                                      {'params': model.voting.parameters()},
                                      {'params': model.completing.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    if args.resume: # optionally resume from a checkpoint
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            #args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(ModelsResourcesShapeDataset(root=args.train_folder, transform=None),
                              batch_size=args.train_batch, shuffle=True,
                              follow_batch=['vtx', 'pts', 'corr_v2p', 'corr_p2v'], num_workers=args.num_workers)
    val_loader = DataLoader(ModelsResourcesShapeDataset(root=args.val_folder, transform=None),
                            batch_size=args.test_batch, shuffle=False,
                            follow_batch=['vtx', 'pts', 'corr_v2p', 'corr_p2v'], num_workers=args.num_workers)
    test_loader = DataLoader(ModelsResourcesShapeDataset(root=args.test_folder, transform=None),
                             batch_size=args.test_batch, shuffle=False,
                             follow_batch=['vtx', 'pts', 'corr_v2p', 'corr_p2v'], num_workers=args.num_workers)
    if args.evaluate:
        print('\nEvaluation only')
        test_losses = test(test_loader, model, args, save_result=True)
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
         # remember best acc and save checkpoint
        is_best = (val_losses["flow_loss"] + val_losses["corr_loss"] + val_losses["vis_loss"]) < lowest_loss
        lowest_loss = min((val_losses["flow_loss"] + val_losses["corr_loss"] + val_losses["vis_loss"]), lowest_loss)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scan mesh corresponce')
    parser.add_argument('--arch', default='deformnet')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 100], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers to load data')

    parser.add_argument('--train_batch', default=2, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--logdir', default='logs/test', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/train_deform/',
                        type=str, help='folder of training data') #/mnt/neghvar
    parser.add_argument('--val_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/val_deform/',
                        type=str, help='folder of validation data') #/mnt/neghvar
    parser.add_argument('--test_folder', default='/mnt/DATA_LINUX2/zhan/morig/ModelsResources/test_deform/',
                        type=str, help='folder of testing data') #/mnt/neghvar
    parser.add_argument('--init_extractor', default='checkpoints/corr_s/model_best.pth.tar',
                        type=str, help='folder of testing data')
    parser.add_argument('--num_frames', default=5, type=int, help='random sample points from the next num_frame frames')
    parser.add_argument('--tau_nce', default=0.07, type=float, help='temprature in hardest nce loss')
    parser.add_argument('--train_extractor', action='store_true')
    parser.add_argument('--num_interp', default=5, type=int)
    print(parser.parse_args())
    main(parser.parse_args())
