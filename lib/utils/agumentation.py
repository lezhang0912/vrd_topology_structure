import albumentations as albu
import cv2
import pickle
import os
from lib.utils.visualization import draw_annotations
import numpy as np


# 增加个水平翻转
def flip_image(min_area=0., min_visibility=0.):
	list_transforms = []
	
	albu.HorizontalFlip()
	list_transforms.extend([albu.HorizontalFlip(p=1.0)])
	
	return albu.Compose(list_transforms, bbox_params=albu.BboxParams(format='pascal_voc', min_area=min_area, \
																	 min_visibility=min_visibility,
																	 label_fields=['category_id']))


if __name__ == "__main__":
	data_pth = "../data/vrd/test.pkl"
	
	with open(data_pth, 'rb') as f:
		anno_data = pickle.load(f, encoding='latin1')
	anno = anno_data[0]
	img_pth = os.path.join("../data/vrd/test_images", anno['img_path'].split('/')[-1])
	img = cv2.imread(img_pth)
	annotations = {'bboxes': anno['boxes'], 'labels': anno['classes']}
	draw_annotations(img, annotations)
	cv2.namedWindow('images', cv2.WINDOW_KEEPRATIO)
	cv2.imshow('images', img)
	cv2.waitKey(1000)
	
	data = {"image": img, "bboxes": anno['boxes'], "category_id": anno['classes']}
	transform = flip_image()
	augmented = transform(**data)
	
	anno_filp = {'bboxes': np.array(augmented['bboxes']), 'labels': anno['classes']}
	
	draw_annotations(image=augmented['image'], annotations=anno_filp)
	img2 = augmented['image']
	cv2.namedWindow('filpped_images', cv2.WINDOW_KEEPRATIO)
	cv2.imshow('filpped_images', img2)
	cv2.waitKey(0)
# test pass
