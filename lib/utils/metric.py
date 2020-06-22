import torch
import shutil


def save_checkpoint(filename, state, is_best=False):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')
	print("save weight completed")


# only trained weight
def specify_weight(net_rel, weight_dict):
	net_rel.fc6.weight.data = weight_dict['fc6.weight']
	net_rel.fc6.bias.data = weight_dict['fc6.bias']
	net_rel.fc7.weight.data = weight_dict['fc7.weight']
	net_rel.fc7.bias.data = weight_dict['fc7.bias']
	net_rel.fc8.weight.data = weight_dict['fc8.weight']
	net_rel.fc8.bias.data = weight_dict['fc8.bias']
	net_rel.gat_conv_rel1.weight.data = weight_dict['gat_conv_rel1.weight']
	net_rel.gat_conv_rel1.rnn.weight_ih.data = weight_dict['gat_conv_rel1.weight_ih']
	net_rel.gat_conv_rel1.rnn.weight_hh.data = weight_dict['gat_conv_rel1.weight_hh']
	net_rel.gat_conv_rel1.rnn.bias_ih.data = weight_dict['gat_conv_rel1.bisa_ih']
	net_rel.gat_conv_rel1.rnn.bias_hh.data = weight_dict['gat_conv_rel1.bisa_hh']


# specify weight
#         upgrade_weight = {
#             'fc6.weight': net_rel.fc6.weight.data,
#             'fc6.bias': net_rel.fc6.bias.data,
#             'fc7.weight': net_rel.fc7.weight.data,
#             'fc7.bias': net_rel.fc7.bias.data,
#             'fc8.weight': net_rel.fc8.weight.data,
#             'fc8.bias': net_rel.fc8.bias.data,
#             'gat_conv_rel1.weight': net_rel.gat_conv_rel1.weight.data,
#             'gat_conv_rel1.weight_ih': net_rel.gat_conv_rel1.rnn.weight_ih.data,
#             'gat_conv_rel1.weight_hh': net_rel.gat_conv_rel1.rnn.weight_hh.data,
#             'gat_conv_rel1.bisa_ih': net_rel.gat_conv_rel1.rnn.bias_ih.data,
#             'gat_conv_rel1.bisa_hh': net_rel.gat_conv_rel1.rnn.bias_hh.data,
#         }

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
	"""

	:param optimizer:
	:param warmup_iters:  1000
	:param warmup_factor: 1.0/1000
	:return:
	"""
	
	def f(x):
		if x >= warmup_iters:
			return 1
		alpha = float(x) / warmup_iters
		return warmup_factor * (1 - alpha) + alpha
	
	return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
