import numpy as np
import os.path as osp
import scipy.io as sio
import scipy
import cv2
import pickle
import sys
import os
import math
from lib.utils.agumentation import flip_image

sys.path.insert(0, '../')
from lib.blob import prep_im_for_blob, prep_im_stand


class VgnomeDataLayer_weight(object):

	def __init__(self, ds_name, stage, model_type=None, proposals_path=None):

		self.stage = stage
		self.model_type = model_type
		self.this_dir = osp.dirname(__file__)
		self._classes = [x.strip() for x in open('../data/%s/obj.txt' % ds_name).readlines()]
		self._relations = [x.strip() for x in open('../data/%s/rel.txt' % ds_name).readlines()]
		self._num_classes = len(self._classes)
		self._num_relations = len(self._relations)
		self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
		self._relations_to_ind = dict(zip(self._relations, range(self._num_relations)))
		self._cur = 0
		self.cache_path = '../data/cache/vg/train.pkl'
		self.img_dir = '../data/vg/images/VG_100K'
		if os.path.exists(self.cache_path):
			with open(self.cache_path, 'rb') as f:
				anno = pickle.load(f)
		else:
			with open('../data/%s/%s.pkl' % (ds_name, stage), 'rb') as fid:
				anno = pickle.load(fid, encoding='latin1')
		if (self.stage == 'train'):
			self._anno = [x for x in anno if x is not None and len(x['classes']) > 1]
		else:
			self.proposals_path = proposals_path
			if (proposals_path != None) and model_type == 'RANK_IM':
				with open(proposals_path, 'rb') as fid:
					proposals = pickle.load(fid, encoding='latin1')
					self._boxes = proposals['boxes']
					self._pred_cls = proposals['cls']
					self._pred_confs = proposals['confs']
			if (proposals_path != None) and model_type == 'Faster-RCNN':
				with open(proposals_path, 'rb') as fid:
					self.rcnn_proposals = pickle.load(fid)
			self._anno = anno
		self._num_instance = len(self._anno)
		self._batch_size = 1
		with open('../data/%s/so_prior.pkl' % ds_name, 'rb') as fid:
			self._so_prior = pickle.load(fid, encoding='latin1')
		# 添加词嵌入向量
		with open('../data/%s/word_embed.pkl' % ds_name, 'rb') as fem:
			self._word_embed = pickle.load(fem, encoding='latin1')
		# 添加unknown词向量
		with open('../data/%s/unknown_embed.pkl' % ds_name, 'rb') as fun:
			self._unknown = pickle.load(fun, encoding='latin1')

	def forward(self):
		if (self.stage == 'train'):
			return self.forward_train_rank_im()
		else:
			if (self.proposals_path is None):
				return self.forward_test()
			else:
				if (self.model_type == 'Faster-RCNN'):
					return self.forward_det_rcnn()

				elif (self.model_type == 'LOC'):
					return self.forward_det_loc()
				else:
					return self.forward_det()


	def forward_train_rank_im(self):

		anno_img = self._anno[self._cur]
		im_path = osp.join(self.img_dir, anno_img['img_path'].split('/')[-1])

		im = cv2.imread(im_path)
		while (im is None):
			print("path or image has wrong")
			print(im_path)
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			anno_img = self._anno[self._cur]
			im_path = osp.join(self.img_dir, anno_img['img_path'].split('/')[-1])
			im = cv2.imread(im_path)

		ih = im.shape[0]
		iw = im.shape[1]
		image_blob, im_scale = prep_im_stand(im)
		blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
		blob[0] = image_blob
		boxes = np.zeros((anno_img['boxes'].shape[0], 5))
		boxes[:, 1:5] = anno_img['boxes'] * im_scale
		classes = np.array(anno_img['classes'])
		ix1 = np.array(anno_img['ix1'])
		ix2 = np.array(anno_img['ix2'])
		rel_classes = anno_img['rel_classes']

		n_rel_inst = len(rel_classes)
		rel_boxes = np.zeros((n_rel_inst, 5))
		rel_labels = -1 * np.ones((1, n_rel_inst * self._num_relations))

		our_label = -1 * np.ones((n_rel_inst, self._num_relations))

		SpatialFea = np.zeros((n_rel_inst, 8))
		Obj_Fea = np.zeros((anno_img['boxes'].shape[0], 1, 32, 32))
		for index in range(anno_img['boxes'].shape[0]):
			BBox = anno_img['boxes'][index]
			Obj_Fea[index][0] = self._getDualMask(ih, iw, BBox)
		UnionFea = np.zeros((n_rel_inst, 1, 32, 32))

		rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
		pos_idx = 0
		label_embeded = np.zeros((len(anno_img['classes']), 300))
		class_list = anno_img['classes']

		if isinstance(class_list, list):
			for index, item in enumerate(class_list):
				label_embeded[index, :] = self._word_embed[item, :]
		else:
			print("class_list is not list type")
		# 添加词向量
		for ii in range(len(rel_classes)):
			sBBox = anno_img['boxes'][ix1[ii]]
			oBBox = anno_img['boxes'][ix2[ii]]
			rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
			rel_boxes[ii, 1:5] = np.array(rBBox) * im_scale
			SpatialFea[ii] = self._getRelativeLoc(sBBox, oBBox)
			rel_so_prior[ii] = self._so_prior[classes[ix1[ii]], classes[ix2[ii]]]
			for r in rel_classes[ii]:
				rel_labels[0, pos_idx] = ii * self._num_relations + r
				pos_idx += 1

		for kk, value in enumerate(rel_classes):
			if isinstance(value, list):
				pos = 0
				for item in value:
					our_label[kk, pos] = item
					pos += 1
			else:
				our_label[kk, 0] = value

		image_blob = image_blob.astype(np.float32, copy=False)
		boxes = boxes.astype(np.float32, copy=False)
		classes = classes.astype(np.float32, copy=False)
		self._cur += 1
		# self.no_img += 1
		if (self._cur >= len(self._anno)):
			self._cur = 0

		unknown = self._unknown["unknown"]
		return blob, boxes, rel_boxes, Obj_Fea, UnionFea, SpatialFea, classes, unknown, ix1, ix2, \
			   label_embeded, rel_labels, rel_so_prior, rel_classes


	def forward_test(self):
		"""Get blobs and copy them into this layer's top blob vector."""
		anno_img = self._anno[self._cur]
		if (anno_img is None):
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			return None
		im_path = osp.join(self.img_dir, anno_img['img_path'].split('/')[-1])

		im = cv2.imread(im_path)

		if (im is None):
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			return None

		ih = im.shape[0]
		iw = im.shape[1]
		# PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
		image_blob, im_scale = prep_im_stand(im)
		# image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
		blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
		blob[0] = image_blob
		# Reshape net's input blobs
		boxes = np.zeros((anno_img['boxes'].shape[0], 5))
		boxes[:, 1:5] = anno_img['boxes'] * im_scale
		classes = np.array(anno_img['classes'])
		ix1 = np.array(anno_img['ix1'])
		ix2 = np.array(anno_img['ix2'])
		rel_classes = anno_img['rel_classes']

		n_rel_inst = len(rel_classes)
		rel_boxes = np.zeros((n_rel_inst, 5))
		SpatialFea = np.zeros((n_rel_inst, 8))

		Obj_Fea = np.zeros((anno_img['boxes'].shape[0], 1, 32, 32))
		UnionFea = np.zeros((n_rel_inst, 1, 32, 32))

		label_embeded = np.zeros((len(anno_img['classes']), 300))
		class_list = anno_img['classes']
		for index, item in enumerate(class_list):
			label_embeded[index, :] = self._word_embed[item, :]

		# SpatialFea = np.zeros((n_rel_inst, 8))
		for ii in range(n_rel_inst):
			sBBox = anno_img['boxes'][ix1[ii]]
			oBBox = anno_img['boxes'][ix2[ii]]
			rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
			soMask = [self._getDualMask(ih, iw, sBBox), \
					  self._getDualMask(ih, iw, oBBox)]
			rel_boxes[ii, 1:5] = np.array(rBBox) * im_scale
			# SpatialFea[ii] = soMask
			SpatialFea[ii] = self._getRelativeLoc(sBBox, oBBox)

		image_blob = image_blob.astype(np.float32, copy=False)
		boxes = boxes.astype(np.float32, copy=False)
		classes = classes.astype(np.float32, copy=False)
		self._cur += 1
		if (self._cur >= len(self._anno)):
			self._cur = 0

		unknown = self._unknown["unknown"]
		return blob, boxes, rel_boxes, Obj_Fea, UnionFea, SpatialFea, classes, unknown, ix1, \
			   ix2, label_embeded, anno_img['boxes'], rel_classes

	def forward_det(self):
		"""
		detect rel 3 tuple
		:return:
		"""
		anno_img = self._anno[self._cur]

		boxes_img = self._boxes[self._cur]
		pred_cls_img = self._pred_cls[self._cur]
		pred_confs_img = self._pred_confs[self._cur]
		if (boxes_img.shape[0] < 2):
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			return None
		im_path = osp.join(self.img_dir, anno_img['img_path'].split('/')[-1])
		im = cv2.imread(im_path)

		# 图片和路径名字对不上
		if (im is None):
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			return None
		ih = im.shape[0]
		iw = im.shape[1]
		# PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
		# image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
		image_blob, im_scale = prep_im_stand(im)
		blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
		blob[0] = image_blob
		# Reshape net's input blobs
		boxes = np.zeros((boxes_img.shape[0], 5))
		boxes[:, 1:5] = boxes_img * im_scale
		classes = pred_cls_img.reshape(len(pred_cls_img))
		# classes 里面是预测框的类别,是数组类型,shape is (9, 1)===>(9,)
		# 添加词向量
		label_embeded = np.zeros((len(classes), 300))
		for index, item in enumerate(classes):
			label_embeded[index] = self._word_embed[item]
		# 添加词向量
		ix1 = []
		ix2 = []
		n_rel_inst = len(pred_cls_img) * (len(pred_cls_img) - 1)
		rel_boxes = np.zeros((n_rel_inst, 5))
		# SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
		SpatialFea = np.zeros((n_rel_inst, 8))
		rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
		i_rel_inst = 0
		# 两两相互变为主宾语
		for s_idx in range(len(pred_cls_img)):
			for o_idx in range(len(pred_cls_img)):
				if (s_idx == o_idx):
					continue
				ix1.append(s_idx)
				ix2.append(o_idx)
				sBBox = boxes_img[s_idx]
				oBBox = boxes_img[o_idx]
				rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
				soMask = [self._getDualMask(ih, iw, sBBox), \
						  self._getDualMask(ih, iw, oBBox)]
				rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
				# SpatialFea[i_rel_inst] = soMask
				SpatialFea[i_rel_inst] = self._getRelativeLoc(sBBox, oBBox)
				rel_so_prior[i_rel_inst] = self._so_prior[classes[s_idx], classes[o_idx]]
				i_rel_inst += 1
		# image_blob = image_blob.astype(np.float32, copy=False)
		boxes = boxes.astype(np.float32, copy=False)
		classes = classes.astype(np.float32, copy=False)
		ix1 = np.array(ix1)
		ix2 = np.array(ix2)
		self._cur += 1
		if (self._cur >= len(self._anno)):
			self._cur = 0
		unknown = self._unknown["unknown"]
		return blob, boxes, rel_boxes, SpatialFea, classes, unknown, \
			   ix1, ix2, label_embeded, boxes_img, pred_confs_img, rel_so_prior

	def forward_det_loc(self):

		anno_img = self._anno[self._cur]
		boxes_img = self._boxes[self._cur]
		pred_cls_img = self._pred_cls[self._cur]
		pred_confs_img = self._pred_confs[self._cur]
		if (boxes_img.shape[0] < 2):
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			return None
		im_path = osp.join(self.img_dir, anno_img['img_path'].split('/')[-1])
		im = cv2.imread(im_path)

		# 图片和路径名字对不上
		if (im is None):
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			return None
		ih = im.shape[0]
		iw = im.shape[1]
		# PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
		# image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
		image_blob, im_scale = prep_im_stand(im)
		blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
		blob[0] = image_blob
		# Reshape net's input blobs
		boxes = np.zeros((boxes_img.shape[0], 5))
		boxes[:, 1:5] = boxes_img * im_scale
		classes = pred_cls_img
		# 添加词向量
		label_embeded = np.zeros((len(classes), 300))
		for index, item in classes:
			label_embeded[index] = self._word_embed[item]
		# 添加词向量
		ix1 = []
		ix2 = []
		n_rel_inst = len(pred_cls_img) * (len(pred_cls_img) - 1)
		rel_boxes = np.zeros((n_rel_inst, 5))
		SpatialFea = np.zeros((n_rel_inst, 8))
		rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
		i_rel_inst = 0
		for s_idx in range(len(pred_cls_img)):
			for o_idx in range(len(pred_cls_img)):
				if (s_idx == o_idx):
					continue
				ix1.append(s_idx)
				ix2.append(o_idx)
				sBBox = boxes_img[s_idx]
				oBBox = boxes_img[o_idx]
				rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
				rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
				SpatialFea[i_rel_inst] = self._getRelativeLoc(sBBox, oBBox)
				rel_so_prior[i_rel_inst] = self._so_prior[classes[s_idx], classes[o_idx]]
				i_rel_inst += 1
		image_blob = image_blob.astype(np.float32, copy=False)
		boxes = boxes.astype(np.float32, copy=False)
		classes = classes.astype(np.float32, copy=False)
		ix1 = np.array(ix1)
		ix2 = np.array(ix2)
		self._cur += 1
		if (self._cur >= len(self._anno)):
			self._cur = 0

		unknown = self._unknown["unknown"]
		return blob, boxes, rel_boxes, SpatialFea, classes, unknown, \
			   ix1, ix2, boxes_img, pred_confs_img, rel_so_prior

	def forward_det_rcnn(self):
		# 1224 这是我们基于x101模型做的
		anno_img = self._anno[self._cur]
		if (anno_img is None):
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			return None
		im_path = osp.join(self.img_dir, anno_img['img_path'].split('/')[-1])
		im = cv2.imread(im_path)
		proposal = self.rcnn_proposals
		id_img = list(proposal.keys())
		img_id = im_path.split('/')[-1]
		if img_id in id_img:

			boxes_img = np.array(proposal[img_id]['boxes'], dtype=np.uint16)
			boxes_img = boxes_img.reshape(-1, 4)
			# label from 1 to 100 thus -1
			pred_cls_img = np.array(proposal[img_id]['labels'], dtype=np.uint8) - 1
			pred_cls_img = pred_cls_img.reshape(-1, 1)
			pred_confs_img = np.array(proposal[img_id]['scores'], dtype=np.float)
			pred_confs_img = pred_confs_img.reshape(-1, 1)
		else:
			self._cur += 1
			if (self._cur >= len(self._anno)):
				self._cur = 0
			print("current img is: ", self._cur)
			return None

		ih = im.shape[0]
		iw = im.shape[1]

		image_blob, im_scale = prep_im_stand(im)
		blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
		blob[0] = image_blob
		# Reshape net's input blobs
		boxes = np.zeros((boxes_img.shape[0], 5))
		boxes[:, 1:5] = boxes_img * im_scale
		classes = pred_cls_img.reshape(len(pred_cls_img))
		# classes 里面是预测框的类别,是数组类型,shape is (9, 1)===>(9,)
		# 添加词向量
		label_embeded = np.zeros((len(classes), 300))
		for index, item in enumerate(classes):
			label_embeded[index] = self._word_embed[item]
		# 添加词向量
		ix1 = []
		ix2 = []
		n_rel_inst = len(pred_cls_img) * (len(pred_cls_img) - 1)
		rel_boxes = np.zeros((n_rel_inst, 5))
		# SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
		SpatialFea = np.zeros((n_rel_inst, 8))
		rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
		i_rel_inst = 0
		# 两两相互变为主宾语
		for s_idx in range(len(pred_cls_img)):
			for o_idx in range(len(pred_cls_img)):
				if (s_idx == o_idx):
					continue
				ix1.append(s_idx)
				ix2.append(o_idx)
				sBBox = boxes_img[s_idx]
				oBBox = boxes_img[o_idx]
				rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
				# soMask = [self._getDualMask(ih, iw, sBBox), \
				#           self._getDualMask(ih, iw, oBBox)]
				rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
				# SpatialFea[i_rel_inst] = soMask
				SpatialFea[i_rel_inst] = self._getRelativeLoc(sBBox, oBBox)
				rel_so_prior[i_rel_inst] = self._so_prior[classes[s_idx], classes[o_idx]]
				i_rel_inst += 1
		# image_blob = image_blob.astype(np.float32, copy=False)
		boxes = boxes.astype(np.float32, copy=False)
		classes = classes.astype(np.float32, copy=False)
		ix1 = np.array(ix1)
		ix2 = np.array(ix2)
		self._cur += 1
		if (self._cur >= len(self._anno)):
			self._cur = 0
		unknown = self._unknown["unknown"]

		return blob, boxes, rel_boxes, SpatialFea, classes, unknown, \
			   ix1, ix2, label_embeded, boxes_img, pred_confs_img, rel_so_prior

	def _getUnionBBox(self, aBB, bBB, ih, iw, margin=10):
		return [max(0, min(aBB[0], bBB[0]) - margin), \
				max(0, min(aBB[1], bBB[1]) - margin), \
				min(iw, max(aBB[2], bBB[2]) + margin), \
				min(ih, max(aBB[3], bBB[3]) + margin)]

	def _getDualMask(self, ih, iw, bb):
		rh = 32.0 / ih
		rw = 32.0 / iw
		x1 = max(0, int(math.floor(bb[0] * rw)))
		x2 = min(32, int(math.ceil(bb[2] * rw)))
		y1 = max(0, int(math.floor(bb[1] * rh)))
		y2 = min(32, int(math.ceil(bb[3] * rh)))
		mask = np.zeros((32, 32))
		mask[y1: y2, x1: x2] = 1
		assert (mask.sum() == (y2 - y1) * (x2 - x1))
		return mask

	def _getRelMask(self, aBB, bBB):
		result = np.zeros_like(aBB)
		result = (aBB + bBB) - np.multiply(aBB, bBB)
		return result

	def _getRelativeLoc(self, aBB, bBB):
		sx1, sy1, sx2, sy2 = aBB.astype(np.float32)
		ox1, oy1, ox2, oy2 = bBB.astype(np.float32)
		sw, sh, ow, oh = sx2 - sx1, sy2 - sy1, ox2 - ox1, oy2 - oy1
		xy = np.array([(sx1 - ox1) / ow, (sy1 - oy1) / oh, (ox1 - sx1) / sw, (oy1 - sy1) / sh])
		wh = np.log(np.array([sw / ow, sh / oh, ow / sw, oh / sh]))
		return np.hstack((xy, wh))


if __name__ == '__main__':
	vrd = VgnomeDataLayer_weight(ds_name='vrd', stage='test', model_type='Faster-RCNN', \
								 proposals_path="../data/vrd/proposals_fpn.pkl")
	data = vrd.forward_det_rcnn()
	print(data[0])
