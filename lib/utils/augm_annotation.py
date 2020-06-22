import albumentations as albu
import cv2
import pickle
import os
from lib.utils.visualization import draw_annotations
from lib.utils.agumentation import flip_image
import numpy as np
from tqdm import trange
import random


def augment_data(anno_pth):
	with open(anno_pth, 'rb') as f:
		anno_data = pickle.load(f, encoding='latin1')
	data = []
	for i in trange(len(anno_data)):
		anno = anno_data[i]
		if anno is not None:
			anno['flipped'] = False
			data.append(anno)
	
	for i in trange(len(anno_data)):
		anno = anno_data[i]
		if anno is not None:
			anno['flipped'] = True
			data.append(anno)
	random.seed(1)
	random.shuffle(data)
	
	with open("../../data/cache/vrd/train.pkl", 'wb') as f:
		pickle.dump(data, f)


if __name__ == "__main__":
	data_pth = "../../data/cache/vrd/train.pkl"
	with open(data_pth, 'rb') as f:
		data = pickle.load(f)
	
	print(len(data))
