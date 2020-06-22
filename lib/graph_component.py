import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import remove_self_loops
import numpy as np

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
            # 默认使用uniform初始化

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


# 添加图卷积模块
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out

class GCNConv_PRelu(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_PRelu, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        x = self.act(x)
        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        out = self.act(aggr_out)
        return out
 # edge_index =  torch.arange(x_so.size(0), dtype=torch.long, device=device)


class GCNConv_Relu(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Relu, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        out = F.relu(aggr_out)

        return out
 # edge_index =  torch.arange(x_so.size(0), dtype=torch.long, device=device)

class GINConv_corr(MessagePassing):
    # 添加相关系数的特征融合
    def __init__(self, nn, eps=0, train_eps=False):
        super(GINConv_corr, self).__init__('add')
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.nn)
        # 自己在代码中指定初始化方式
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight):
        "add correction factor"
        x = x.unsequeeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        updata_x_j = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
        out = self.nn((1 + self.eps) * x + updata_x_j)
        return out

    def message(self, x_j, edge_weight):
        # edge_index是一个张量，shape [E, 2], edge_weight [E, 1]
        # x_j shape: [F, f]
        return edge_weight.view(-1, 1)* x_j

    def __repr__(self):
        # 该类的名字
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


def adjoint_obj(boxes):
    #  numpy array
    nodes = boxes.shape[0]
    index = np.linspace(start=0,stop=nodes-1,num=nodes,dtype=np.int)
    edge_index = np.zeros((2,nodes*(nodes-1)), dtype=np.int)
    # [2, edge_nums]
    for i in index:
        begin = i * (nodes-1)
        end = (i+1) * (nodes-1)
        edge_index[0, begin:end] = i
        edge_index[0, begin:end] = np.delete(index,i)

    return edge_index

def adjoint_rel(boxes,ix1,ix2):
    # array
    nodes = boxes.shape[0]
    sub_nodes = ix1.size
    edge_nums = sub_nodes * 2
    edge_index = np.zeros((2, edge_nums), dtype=np.int)
    # sub_nodes 编号
    order_sub = np.linspace(start=0, stop=sub_nodes - 1, num=sub_nodes, dtype=np.int)
    sub_index = ix1
    subdict = dict(zip(order_sub.tolist(), sub_index.tolist()))
    order_obj = np.linspace(start=sub_nodes, stop=sub_nodes * 2 - 1, num=sub_nodes, dtype=np.int)
    obj_index = ix2
    objdict = dict(zip(order_obj.tolist(), obj_index.tolist()))
    pre_index = np.linspace(start=nodes, stop=nodes + sub_nodes - 1, num=sub_nodes, dtype=np.int)
    for index, i in enumerate(pre_index):
        # source to target
        edge_index[0, index] = subdict[index]
        edge_index[1, index] = i
        edge_index[0, index + sub_nodes] = objdict[index + sub_nodes]
        edge_index[1, index + sub_nodes] = i
    return edge_index

def cal_correction(co_prior, su_index, ob_index):
    # print("co_prior shape", co_prior.shape)
    sbp = np.sum(co_prior[su_index, ob_index, :])
    if sbp > 0.0:
        su_cor = np.power(sbp, 0.5) / (np.sum(co_prior[su_index, :, :]) + np.sum(co_prior[:, su_index, :]))
        ob_cor = np.power(sbp, 0.5) / (np.sum(co_prior[ob_index, :, :]) + np.sum(co_prior[:, ob_index, :]))
        # print("sub, obj, pred", sbp)
    else:
        su_cor = 0.0002
        ob_cor = 0.0002

    return su_cor, ob_cor


def adjoint_rel_weight(classes, ix1, ix2, co_prior, rel_classes=None):
    """
    # array add edge-weight 0615 so_prior shape [100, 100, 70]
    # rel_classes是list,元素类型也是list, 但是边的相关性可能对应两个值，我们取其中最大的那个来表示
    # 如果每个元素列表最大的相关性是0，那么我们就把它表示为0.5
    :param classes: list type or 1 dim array
    :param ix1: array
    :param ix2: array
    :param so_prior: array
    :param rel_classes: list
    :return:
    """

    nodes = len(classes)
    sub_nodes = ix1.size
    edge_nums = sub_nodes * 2
    edge_index = np.zeros((2, edge_nums), dtype=np.int)
    edge_weight = np.zeros((1, edge_nums), dtype=np.float)
    # sub_nodes 编号
    order_sub = np.linspace(start=0, stop=sub_nodes - 1, num=sub_nodes, dtype=np.int)

    sub_index = ix1
    subdict = dict(zip(order_sub.tolist(), sub_index.tolist()))

    order_obj = np.linspace(start=sub_nodes, stop=sub_nodes * 2 - 1, num=sub_nodes, dtype=np.int)
    obj_index = ix2
    objdict = dict(zip(order_obj.tolist(), obj_index.tolist()))

    pre_index = np.linspace(start=nodes, stop=nodes + sub_nodes - 1, num=sub_nodes, dtype=np.int)

    for index, i in enumerate(pre_index):
        # source to target
        edge_index[0, index] = subdict[index]
        edge_index[1, index] = i
        edge_index[0, index + sub_nodes] = objdict[index + sub_nodes]
        edge_index[1, index + sub_nodes] = i
        # add edge-weight
        sub_axis = classes[ix1[index]]
        obj_axis = classes[ix2[index]]
        # pred_axis = max(rel_classes)
        # edge_weight[0, index] = so_prior[sub_axis, obj_axis, pred_axis]
        # print(sub_axis, obj_axis)
        edge_weight[0, index], edge_weight[0, index + sub_nodes] = cal_correction(co_prior, np.int(sub_axis), np.int(obj_axis))
        # edge_weight[0, index + sub_nodes] = so_prior[sub_axis, obj_axis, pred_axis]

    # print("type edge_index", type(edge_index))
    # print("type edge_weight", type(edge_weight))
    return edge_index, edge_weight





