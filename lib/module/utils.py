import torch
import torch.nn.functional as F
import torch.nn as nn
import shutil
import pickle


def save_checkpoint(filename, state, is_best=False):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	
	# 计算每次的目标函数值
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


class MultiLabelAdaptiveMarginLoss(nn.Module):
	
	def __init__(self):
		super(MultiLabelAdaptiveMarginLoss, self).__init__()
	
	def forward(self, input_data, target, adaptive_margin, size_average=False):
		"""自定义损失函数,自适应边际损失函数"""
		# 自定义max函数 torch.max(), 自适应阈值为 1+P(p|s, o)-P(p'!s,o)
		# 网络输出为score， paper中的loss
		assert input_data.ndimension() == 2
		assert input_data.ndimension() == adaptive_margin.ndimension()
		assert adaptive_margin.ndimension() == input_data.ndimension()
		loss_data = 0
		for mini_index in range(input_data.size()[0]):
			mask = target[mini_index] > -1
			mini_target = torch.masked_select(target[mini_index], mask)
			assert mini_target.ndimension() == 1
			# 一维的
			for i in range(mini_target.size()[0]):
				# index_self = input_data[mini_index] != input_data[mini_index, mini_target[i]]
				index = torch.range(start=0, end=input_data.size()[1] - 1, dtype=torch.int64).cuda()
				index_mask = (index != mini_target[i])
				except_index = torch.masked_select(index, index_mask)
				except_self = torch.take(input_data[mini_index], except_index)
				need_margin = torch.take(adaptive_margin[mini_index], except_index)
				
				different = 1 + (adaptive_margin[mini_index, mini_target[i]] - need_margin) - (
						input_data[mini_index, mini_target[i]] - except_self)
				loss_data += torch.sum(F.relu(different))
		
		loss_data = loss_data / input_data.size()[1]
		if size_average:
			return loss_data / input_data.size()[0]
		else:
			return loss_data


class MultiLabelAdaptiveSoftmaxMarginLoss(nn.Module):
	
	def __init__(self):
		super(MultiLabelAdaptiveSoftmaxMarginLoss, self).__init__()
	
	def forward(self, input_data, target, adaptive_margin, size_average=False):
		# """我们理想中的损失函数"""
		# softmax 归一化, 可能存在梯度消失
		input_data = torch.exp(input_data)
		assert input_data.ndimension() == 2
		assert input_data.ndimension() == adaptive_margin.ndimension()
		assert adaptive_margin.ndimension() == input_data.ndimension()
		loss_data = 0
		print(input_data.size())
		print(adaptive_margin.size())
		print(target.size())
		print(input_data.data)
		print("\n")
		print(target.data)
		# input_data size:(5, 70)
		for mini_index in range(input_data.size()[0]):
			mask = target[mini_index] > -1
			mini_target = torch.masked_select(target[mini_index], mask)
			assert mini_target.ndimension() == 1
			
			# 一维的
			for i in range(mini_target.size()[0]):
				# index_self = input_data[mini_index] != input_data[mini_index, mini_target[i]]
				index = torch.range(start=0, end=input_data.size()[1] - 1, dtype=torch.int64).cuda()
				index_mask = (index != mini_target[i])
				except_index = torch.masked_select(index, index_mask)
				except_self = torch.take(input_data[mini_index], except_index)
				need_margin = torch.take(adaptive_margin[mini_index], except_index)
				
				different = (adaptive_margin[mini_index, mini_target[i]] - need_margin) - \
							(input_data[mini_index, mini_target[i]] - except_self)
				loss_data += torch.sum(F.relu(different))
		
		loss_data = loss_data / input_data.size()[1]
		if size_average:
			return loss_data / input_data.size()[0]
		else:
			return loss_data


class MultiLabelConstantMarginLoss(nn.Module):
	
	def __init__(self):
		super(MultiLabelConstantMarginLoss, self).__init__()
	
	def forward(self, input_data, target, size_average=False):
		"""margin=1, no prior，普通的hinge loss函数, pytorch已经有实现"""
		assert input_data.ndimension() == 2
		
		loss_data = 0
		for mini_index in range(input_data.size()[0]):
			mask = target[mini_index] > -1
			mini_target = torch.masked_select(target[mini_index], mask)
			assert mini_target.ndimension() == 1
			# 一维的
			for i in range(mini_target.size()[0]):
				# index_self = input_data[mini_index] != input_data[mini_index, mini_target[i]]
				# except_self = torch.masked_select(input_data[mini_index], index_self)
				index = torch.range(start=0, end=input_data.size()[1] - 1, dtype=torch.int64).cuda()
				index_mask = index != mini_target[i]
				except_index = torch.masked_select(index, index_mask)
				except_self = torch.take(input_data[mini_index], except_index)
				# fun max(0, 1-(X[Y[j]]- X[i]))
				different = 1 - (input_data[mini_index, mini_target[i]] - except_self)
				loss_data += torch.sum(F.relu(different))
		
		loss_data = loss_data / input_data.size()[1]
		if size_average:
			return loss_data / input_data.size()[0]
		else:
			return loss_data


class LESPLoss(nn.Module):
	
	def __init__(self):
		super(LESPLoss, self).__init__()
	
	def forward(self, input_data, target, size_average=False):
		"""log-sum-exp pairwise function
		Improving Pairwise Ranking for Multi-label Image Classification 2017CVPR
		"""
		loss_data = 0
		for mini_index in range(input_data.size()[0]):
			mask = target[mini_index] > -1
			mini_target = torch.masked_select(target[mini_index], mask)
			assert mini_target.ndimension() == 1
			# 一维的
			for i in range(mini_target.size()[0]):
				# index_self = input_data[mini_index] != input_data[mini_index, mini_target[i]]
				index = torch.range(start=0, end=input_data.size()[1] - 1, dtype=torch.int64).cuda()
				index_mask = index != mini_target[i]
				except_index = torch.masked_select(index, index_mask)
				except_self = torch.take(input_data[mini_index], except_index)
				
				different = (input_data[mini_index, mini_target[i]] - except_self)
				
				loss_data += torch.sum(torch.exp(different))
			one = torch.Tensor(1).cuda()
			loss = torch.log(one + loss_data)
		loss_1 = loss / input_data.size()[1]
		if size_average:
			return loss_1 / input_data.size()[0]
		else:
			return loss_1


def FastMLMLLoss():
	"""
	Fast Multi-Instance Multi-Label Learning zhou zhihua
	"""
	pass


class MultiLabelSoftMarginLoss_Our(nn.Module):
	
	def __init__(self):
		super(MultiLabelSoftMarginLoss_Our, self).__init__()
	
	def forward(self, input_data, target):
		"""MultiLabelSoftMargin，最大交叉熵分类函数"""
		# one-hot
		# 加
		hot = target >= 0
		# target_hot.dtype(torch.LongTensor)
		target_hot = hot.to(torch.float32)
		# print(target_hot)
		# print()
		# input_data.type(torch.LongTensor)
		loss_func = nn.MultiLabelSoftMarginLoss(size_average=False)
		loss_data = loss_func(input_data, target_hot)
		return loss_data


# 测试, 损失函数验证通过

def save_pickle(number, number_path):
	pkl_number = pickle.dumps(number)
	with open(number_path, "wb") as f:
		f.write(pkl_number)


def read_pickle(number_path):
	with open(number_path, "rb") as f:
		return pickle.load(f)


def adjust_learning_rate(lr, optimizer):
	"""Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
	decay_lr = lr * 0.1
	for param_group in optimizer.param_groups:
		param_group['lr'] = decay_lr
