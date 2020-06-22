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
from torch.utils.tensorboard import SummaryWriter
from lib.nets.Vrd_Model_GA import Vrd_Graph_GA
from lib.data_layers.vrd_data_graph_weight import VrdDataLayer_weight
from lib.graph_model_version import test_rel_gat, test_pre_gat
from lib.data_layers.vg_data_graph_weight import VgnomeDataLayer_weight

# add path
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
add_path(osp.join(this_dir, '..'))


def main(args):
    # load data
    if args.ds_name == 'vrd':
        test_data_layer = VrdDataLayer_weight('vrd', 'test', model_type=args.model_type)
    if args.ds_name == 'vg':
        test_data_layer = VgnomeDataLayer_weight('vg', 'test', model_type=args.model_type)

    args.num_relations = test_data_layer._num_relations
    args.num_classes = test_data_layer._num_classes
    # load net change net
    net_rel = Vrd_Graph_GA(args, trained=False)

    if torch.cuda.is_available() and args.device == "gpu":
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    device = torch.device('cuda:0')
    net_rel.to(device=device)

    logs_pth = osp.join("../logs", args.ds_name, 'session_' + args.session)
    if not osp.exists(logs_pth):
        os.makedirs(logs_pth)
    writer = SummaryWriter(logs_pth)

    res_file = '../experiment/test_results_session_%s/%s_graph_det.txt' % (args.session, args.ds_name)
    res_pre_file = '../experiment/test_results_session_%s/%s_graph_pre.txt' % (args.session, args.ds_name)
    if not osp.exists('../experiment/test_results_session_%s' % args.session):
        os.makedirs('../experiment/test_results_session_%s' % args.session)

    if osp.isfile(args.resume):
        print("=> loading model '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        net_rel.load_state_dict(checkpoint['state_dict_rel'])
        args.epochs = checkpoint['epoch']
        pre_headers = ["Epoch", "Pre R@50", "ZS@50", "R@100", "ZS@100"] 
        pre_res = []
        for topk in [1, 70]:
            print(topk)
            pre_res.append((args.epochs,) + test_pre_gat(net_rel, device, args, writer, topk=topk))
            with open(res_pre_file, 'w') as f:
                f.write(tabulate(res_pre, pre_headers))
        
        headers = ["Epoch", "Rel R@50", "ZS@50", "R@100", "ZS@100"]
        res = []
        k = [1, 10, 70]
        for topk in k:
            print(topk)
            res.append((args.epochs,) +  test_rel_gat(net_rel, device, topk, args))
            print(tabulate(res, headers))
            with open(res_file, 'w') as f:
                f.write(tabulate(res, headers))
        
            
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VRD Training')
    parser.add_argument('--name', dest='name', help='experiment name', default=None, type=str)
    parser.add_argument('--ds_name', dest='ds_name', help='dataset name', default='vg', type=str)
    parser.add_argument('--model_type', dest='model_type', help='model type: RANK_IM, Faster-RCNN', default='RANK_IM', type=str)
    parser.add_argument('--no_obj_prior', dest='use_obj_prior', action='store_false')
    parser.set_defaults(use_obj_prior=True)
    parser.add_argument('--num_relations', dest='num_relations', help='dataset relationships total number', default=70,
                        type=int)
    parser.add_argument('--num_classes', dest='num_classes', help='dataset object category total number', default=100,
                        type=int)
    parser.add_argument('--session', default='0', type=str, metavar='S', help='training session')
    parser.add_argument('--device', default='gpu', type=str, metavar='D', help='run device (ie. gpu or cpu)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--proposal', default='../data/vrd/proposal.pkl', metavar="PROPOSAL",
                        help="detector boxes proposals")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)
    print('Evaluating...')
    main(args)
