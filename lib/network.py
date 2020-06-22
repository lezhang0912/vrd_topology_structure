import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def load_pretrained_RO_npy(faster_rcnn_model, fname):
    params = np.load(fname).item()
    # vgg16
    vgg16_dict = faster_rcnn_model.state_dict()
    for name, val in vgg16_dict.items():
        if name.find('bn.') >= 0 or not 'conv' in name or 'lo' in name:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])

        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)

        val.copy_(param)

    # fc6 fc7
    frcnn_dict = faster_rcnn_model.state_dict()
    pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'fc6_obj.fc': 'fc6', 'fc7_obj.fc': 'fc7'}
    for k, v in pairs.items():
        key = '{}.weight'.format(k)
        param = torch.from_numpy(params[v]['weights']).permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(params[v]['biases'])
        frcnn_dict[key].copy_(param)

def load_pretrained_npy(faster_rcnn_model, fname):
    params = np.load(fname).item()
    # vgg16
    vgg16_dict = faster_rcnn_model.state_dict()
    for name, val in vgg16_dict.items():
        if name.find('bn.') >= 0 or not 'conv' in name or 'lo' in name:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])

        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)

        val.copy_(param)

    # fc6 fc7
    frcnn_dict = faster_rcnn_model.state_dict()
    pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7'}
    # pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'fc7_so.fc': 'fc7'}
    for k, v in pairs.items():
        key = '{}.weight'.format(k)
        param = torch.from_numpy(params[v]['weights']).permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(params[v]['biases'])
        frcnn_dict[key].copy_(param)

def load_vgg16_weight(net, weight_path):

    net_dict = net.state_dict()
    pretrained_dict = torch.load(weight_path)

    net_dict["conv1.0.conv.weight"] = pretrained_dict["features.0.weight"]
    net_dict["conv1.0.conv.bias"] = pretrained_dict["features.0.bias"]
    net_dict["conv1.1.conv.weight"] = pretrained_dict["features.2.weight"]
    net_dict["conv1.1.conv.bias"] = pretrained_dict["features.2.bias"]

    net_dict["conv2.0.conv.weight"] = pretrained_dict["features.5.weight"]
    net_dict["conv2.0.conv.bias"] = pretrained_dict["features.5.bias"]
    net_dict["conv2.1.conv.weight"] = pretrained_dict["features.7.weight"]
    net_dict["conv2.1.conv.bias"] = pretrained_dict["features.7.bias"]

    net_dict["conv3.0.conv.weight"] = pretrained_dict["features.10.weight"]
    net_dict["conv3.0.conv.bias"] = pretrained_dict["features.10.bias"]
    net_dict["conv3.1.conv.weight"] = pretrained_dict["features.12.weight"]
    net_dict["conv3.1.conv.bias"] = pretrained_dict["features.12.bias"]
    net_dict["conv3.2.conv.weight"] = pretrained_dict["features.14.weight"]
    net_dict["conv3.2.conv.bias"] = pretrained_dict["features.14.bias"]

    net_dict["conv4.0.conv.weight"] = pretrained_dict["features.17.weight"]
    net_dict["conv4.0.conv.bias"] = pretrained_dict["features.17.bias"]
    net_dict["conv4.1.conv.weight"] =  pretrained_dict["features.19.weight"]
    net_dict["conv4.1.conv.bias"] = pretrained_dict["features.19.bias"]
    net_dict["conv4.2.conv.weight"] = pretrained_dict["features.21.weight"]
    net_dict["conv4.2.conv.bias"] = pretrained_dict["features.21.bias"]
    net_dict["conv5.0.conv.weight"] = pretrained_dict["features.24.weight"]
    net_dict["conv5.0.conv.bias"] = pretrained_dict["features.24.bias"]
    net_dict["conv5.1.conv.weight"] = pretrained_dict["features.26.weight"]
    net_dict["conv5.1.conv.bias"] = pretrained_dict["features.26.bias"]
    net_dict["conv5.2.conv.weight"] = pretrained_dict["features.28.weight"]
    net_dict["conv5.2.conv.bias"] = pretrained_dict["features.28.weight"]
    # net_dict["fc6.fc.weight"] = pretrained_dict["classifier.0.weight"]
    # net_dict["fc6.fc.bias"] = pretrained_dict["classifier.0.bias"]
    # net_dict["fc7.fc.weight"] = pretrained_dict["classifier.3.weight"]
    # net_dict["fc7.fc.bias"] = pretrained_dict["classifier.3.bias"]
    print("successfully load vgg16 weight!\n")
    # ["fc8.fc.weight"] =
    # ["fc8.fc.bias"] =





def pretrain_with_det(net, det_model_path):
    det_model = torch.load(det_model_path)    
    for k in det_model.keys():
        if('rpn' in k or 'bbox' in k):
            del det_model[k]

    target_keys = []
    for k in net.state_dict().keys():
        if('conv' in k):
            target_keys.append(k)    

    for ix, k in enumerate(det_model.keys()):
        if('features' in k):
            det_model[target_keys[ix]] = det_model[k]
            del det_model[k]

    det_model['fc6.fc.weight'] = det_model['vgg.classifier.0.weight']
    det_model['fc6.fc.bias'] = det_model['vgg.classifier.0.bias']
    det_model['fc7.fc.weight'] = det_model['vgg.classifier.3.weight']
    det_model['fc7.fc.bias'] = det_model['vgg.classifier.3.bias']
    det_model['fc_obj.fc.weight'] = det_model['cls_score_net.weight']
    det_model['fc_obj.fc.bias'] = det_model['cls_score_net.bias']

    for k in det_model.keys():
        if('vgg' in k):
            del det_model[k]
    del det_model['cls_score_net.weight']
    del det_model['cls_score_net.bias']

    model_dict = net.state_dict()
    model_dict.update(det_model)
    net.load_state_dict(model_dict)

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    # v = Variable(torch.from_numpy(x).type(dtype))
    v = torch.from_numpy(x).type(dtype)
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            # weights_normal_init(m, dev)
            nn.init.normal_(m.weight.data, 0.0, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0.0, dev)
                nn.init.normal_(m.weight.data, 0.0, dev)
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0.0, dev)
                nn.init.normal_(m.weight.data, 0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)
