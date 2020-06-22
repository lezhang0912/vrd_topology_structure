import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models import resnet
from torchvision.ops.misc import FrozenBatchNorm2d
from collections import OrderedDict
import sys

sys.path.insert(0, '../')
import os.path as osp
from lib.network import Conv2d, FC
import lib.network as network
from lib.graph_component import GINConv_corr
from torchvision.models import vgg16
from torchvision.ops.roi_pool import RoIPool
from torch_geometric.nn import GatedGraphConv, GINConv


class Vrd_Graph_GA(nn.Module):
    def __init__(self, args, trained=True, bn=False, backbone_type='res101'):
        super(Vrd_Graph_GA_v4, self).__init__()
        self.n_obj = args.num_classes
        self.n_rel = args.num_relations
        global res101
        if trained and backbone_type == 'res101':
            res101 = resnet.__dict__['resnet101'](pretrained=False, norm_layer=FrozenBatchNorm2d)
            weight_path = "../models/resnet101-5d3b4d8f.pth"
            state = torch.load(weight_path)
            res101.load_state_dict(state)
            layers_res = OrderedDict()
            for k, v in res101.named_children():
                if k in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
                    layers_res[k] = v
            backbone = nn.Sequential(layers_res)
            # 2048
            self.features = backbone
        else:
            res101 = resnet.__dict__['resnet101'](pretrained=False, norm_layer=FrozenBatchNorm2d)
            layers_res = OrderedDict()
            for k, v in res101.named_children():
                if k in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
                    layers_res[k] = v
            backbone = nn.Sequential(layers_res)
            # 2048
            self.features = backbone
        network.set_trainable(self.features, requires_grad=False)
        self.roi_pool = RoIPool((14, 14), 1.0 / 16)
        self.inter_layer = res101.layer4
        network.set_trainable(self.inter_layer, requires_grad=False)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc6 = nn.Linear(2048, 256)
        self.fc_obj = nn.Linear(2048, self.n_obj)
        self.gat_conv_rel1 = GatedGraphConv(out_channels=256, num_layers=2)
        self.rel_1 = nn.Linear(768, 256)
        self.rel_2 = nn.Linear(256, self.n_rel)
        self.fc_lov = nn.Linear(8, 256)
        self.fc_sub_obj = nn.Linear(2 * 300, 256)

        self.initialize_param()

    def forward(self, im_data, boxes, rel_boxes, spaFea, ix1, ix2, classes_embed, unknown, \
                edge_index_rel, edge_weight, args):
        im_data = im_data.permute(0, 3, 1, 2)
        x = self.features(im_data)
        x_so = self.roi_pool(x, boxes)
        x_so = self.inter_layer(x_so)
        x_so = self.pool(x_so)
        x_so = torch.flatten(x_so, 1)
        obj_score = self.fc_obj(x_so)

        x_so_g2 = self.fc6(x_so)
        x_so_g2 = F.relu(x_so_g2)
        x_so_g2 = F.dropout(x_so_g2, training=self.training)
        x_u = self.roi_pool(x, rel_boxes)
        x_u = self.inter_layer(x_u)
        x_u = self.pool(x_u)
        x_u = torch.flatten(x_u, 1)
        x_r = self.fc6(x_u)
        x_r = F.relu(x_r)
        x_r = F.dropout(x_r, training=self.training)
        x_spa = self.fc_lov(spaFea)
        x_spa = F.relu(x_spa)
        x_sub_sem = torch.index_select(classes_embed, 0, ix1)
        x_obj_sem = torch.index_select(classes_embed, 0, ix2)
        x_sub_obj = torch.cat((x_sub_sem, x_obj_sem), 1)
        x_sub_obj = self.fc_sub_obj(x_sub_obj)
        x_sub_obj = F.relu(x_sub_obj)
        # 视觉特征图
        x_c_g = torch.cat((x_so_g2, x_r), 0)
        x_g2 = self.gat_conv_rel1(x_c_g, edge_index_rel, edge_weight)
        rel_node_1 = x_g2[x_so.size(0):]
        rel_node = torch.cat((rel_node_1, x_sub_obj, x_spa), 1)
        rel_score = self.rel_1(rel_node)
        rel_score = F.relu(rel_score)
        rel_score = F.dropout(rel_score, training=self.training)
        rel_score = self.rel_2(rel_score)
        return rel_score, obj_score

    def initialize_param(self):
        nn.init.kaiming_normal_(self.fc6.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_obj.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rel_1.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rel_2.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_lov.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_sub_obj.weight.data, mode='fan_out', nonlinearity='relu')
        self.gat_conv_rel1.reset_parameters()


if __name__ == '__main__':
    pass
