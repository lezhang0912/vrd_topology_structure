![image](https://github.com/lezhang0912/vrd_topology_structure/blob/master/img/VRD_topoloy%20structure%20copy.PNG)

# **Visual relationship detection with region topology structure**

This repository is about the code implementation of paper-Visual relationship detection with region topology structure. Here, we are very grateful for the public code provided by Liang et al. [vrd-dsr](https://github.com/GriffinLiang/vrd-dsr) The work of this paper is closely related to this.

------

## Environment

​	ubuntu 16.04

	pytorch 1.2.0 +

	python 3.6.5

	torch_geometric 1.1+

	albumentations 0.3+

	tabulate 0.8.0+

## Build directory

```
mkdir data/cache # caching file

mkdir data/pretrained_model # model checkpoint

mkdir data/pretrained_model/vg

mkdir data/pretrained_model/vrd

mkdir data/vrd # vrd dataset file

mkdir data/vg # vg dataset file

mkdir experiment # print result of each session

mkdir logs # logs file

mkdir models # resnet_101 pretrained weight file
```

## Prepare dataset

*vrd dataset*

Download from [vrd dataset](https://drive.google.com/file/d/158EyLESdU-et6iHu1-NK4dwVouHJKBNa/view?usp=sharing)

*unzip file* 

`cd data/vrd/`

`cp -r your/vrd/download/path/*  ./`

*vg dataset*

Download from [vg dataset](https://drive.google.com/file/d/1FL3bSW7owthjpKdv2uileOSkKtiO_XN9/view?usp=sharing)

*unzip file* 

`cd data/vg/`

`cp -r  your/vg/download/path/*  ./`

Please download vg images  from [vg images](https://visualgenome.org/api/v0/api_home.html)

merge images to a file

next build soft link

`ln -s your/vg/images/path  ./images`

## Load pre_trained weight 

*vrd weight*

Download from [vrd weight](https://drive.google.com/file/d/1sUzKO27mTvwgAbuk1Do7oXRHnDeUjaps/view?usp=sharing)

`cd data/pretrained_model/vrd/`

`cp epoch_4_session_4_vrd_graph_rel.pth  ./`

*vg weight*

Download from [vg weight](https://drive.google.com/file/d/1OtHN4jzxp0fJWo20aPnDsAD17KG7bhl_/view?usp=sharing)

`cd data/pretrained_model/vg/`

`cp epoch_7_session_5_vg_graph_rel.pth  ./`

## Start training from scratch

First, you should load resnet_101 pretrained weight on imagenet.

Download [resnet_101 pretrained_weight](https://drive.google.com/file/d/1Wa5zpvdOdnZaMwsbQUxh0A1H9wGIKM8I/view?usp=sharing)

`cp your/resnet_101_wegint_path/resnet101-5d3b4d8f.pth  models/`

In the training phase, after training an epoch, we will conduct a Pre Det task evaluation.

`cd tools`

######  vrd dataset

`python train_graph.py --dataset vrd --name VRD_RANK --session 4 --device gpu --epochs 5`

**note:lr_step = [3, ]**

######  vg dataset

`python train_graph.py --dataset vg --name VRD_RANK --session 5 --device gpu --epochs 12`

**note: lr_step = [8,  11]**

## Evaluate model

`cd tools`

for example: vrd dataset

``python  test_graph.py --ds_name vrd --device gpu--model_type  Faster-RCNN --proposal  ../data/faster-rcnn-detection/x101_proposals.pkl  --resume ../data/pretrained_model/vrd/epoch_4_session_4_vrd_graph_rel.pth``

## Citation

If you use this code, please cite the following paper(s):

```en
@article{vrd_graph,
	title={Visual relationship detection with region topology structure},
	author={Le Zhang,Ying Wang,HaiShun Chen,Jie Li,ZhenXi Zhang},
	year={2020}
}
```


