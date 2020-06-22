import os
import os.path as osp
import sys
import pickle
import argparse
from tabulate import tabulate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from lib.nets.Vrd_Model_GA import Vrd_Graph_GA
from lib.data_layers.vrd_data_graph_weight import VrdDataLayer_weight
from lib.data_layers.vg_data_graph_weight import VgnomeDataLayer_weight
from lib.graph_model_version import  train_net_gat, test_pre_gat
from lib.utils.metric import save_checkpoint
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# add path
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
add_path(osp.join(this_dir, '..'))


def save_pickl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)



def main(args):
    if args.ds_name == 'vrd':
        train_data_layer = VrdDataLayer_weight('vrd', 'train', model_type=args.model_type)
    if args.ds_name == 'vg':
        train_data_layer = VgnomeDataLayer_weight('vg', 'train', model_type=args.model_type)

    args.num_relations = train_data_layer._num_relations
    args.num_classes = train_data_layer._num_classes
    net_rel = Vrd_Graph_GA(args, trained=True)
    if torch.cuda.is_available() and args.device == "gpu":
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    net_rel.to(device=device)

    # tensorboard save file
    logs_pth = osp.join("../logs", args.ds_name, 'session_' + args.session)
    if not osp.exists(logs_pth):
        os.makedirs(logs_pth)
    writer = SummaryWriter(logs_pth)


    opt_params_rel = [p for p in net_rel.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(opt_params_rel, lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    criterion_rel = nn.MultiLabelMarginLoss().to(device=device)
    ops = {'optimizer': optimizer, 'criterion': criterion_rel}

    if args.resume:
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            net_rel.load_state_dict(checkpoint['state_dict_rel'])
            optimizer.load_state_dict(checkpoint['optimizer_rel'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    res_file = '../experiment/session_%s/%s_graph_layer.txt' % (args.session, args.ds_name)
    if not osp.exists('../experiment/session_%s' % args.session):
        os.makedirs('../experiment/session_%s' % args.session)
    save_dir = '../data/pretrained_model/%s' % args.ds_name
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    headers = ["Epoch", "Pre R@50", "ZS@50", "R@100", "ZS@100"]
    res = []
    loss_obj_dict = {}
    loss_rel_dict = {}
    for epoch in range(args.start_epoch, args.epochs):
        rec_obj_loss, rec_rel_loss = train_net_gat(train_data_layer, net_rel, epoch, ops, device, args, writer)
        state = {'epoch': epoch, 'state_dict_rel': net_rel.state_dict(),
                 'optimizer_rel': optimizer.state_dict(), }
        save_weight_pth = '%s/epoch_%d_session_%s_%s_graph_rel.pth' % (save_dir, epoch + 1, args.session, args.ds_name)
        save_checkpoint(save_weight_pth, state)

        res.append((epoch,) + test_pre_gat(net_rel, device, args, writer, epoch))
        with open(res_file, 'w') as f:
            f.write(tabulate(res, headers))

        if lr_schedule is not None:
            lr_schedule.step()

    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VRD Training')
    parser.add_argument('--name', dest='name', help='experiment name', default=None, type=str)
    parser.add_argument('--ds_name', dest='ds_name', help='dataset name', default='vg', type=str)
    parser.add_argument('--model_type', dest='model_type', help='model type: RANK_IM, LOC', default='RANK_IM', type=str)
    parser.add_argument('--num_relations', dest='num_relations', help='dataset relationships total number', default=70,
                        type=int)
    parser.add_argument('--num_classes', dest='num_classes', help='dataset object category total number', default=100,
                        type=int)
    parser.add_argument('--no_obj_prior', dest='use_obj_prior', action='store_false')
    parser.set_defaults(use_obj_prior=True)
    parser.add_argument('--session', default='0', type=str, metavar='S', help='training session')
    parser.add_argument('--device', default='gpu', type=str, metavar='D', help='run device (ie. gpu or cpu)')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_gamma', dest='lr_gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    # vrd [3, ]  epochs: 5
    # vg [8, 11], epochs: 12
    parser.add_argument('--lr_steps', dest='lr_steps', default=[3,], type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', dest='weight_decay', default=5e-4, type=float, metavar='W',
                        help='weightdecay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--proposal', default='', metavar="PROPOSAL", help="detector boxes proposals")
    parser.add_argument('--resume_optimizer', default='', type=str, metavar='Optimizer_PATH',
                        help='path to latest optimzer checkpoint (default: none')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)
    main(args)
