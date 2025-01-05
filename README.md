<p align="center">
  <h1 align="center">ESMStereo: Enhanced ShuffleMixer Disparity Upsampling
for Real-Time and Accurate Stereo Matching</h1>
  <p align="center">
    Mahmoud Tahmasebi* (mahmoud.tahmasebi@research.atu.ie), Saif Huq, Kevin Meehan, Marion McAfee
  </p>
  <h3 align="center"><a href="TBD">Paper</a>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/M2219/ESMStereo/blob/main/imgs/Graphical_abstract.png" alt="Logo" width="80%">
  </a>
</p>


## Performance of ESMStereo-L on KITTI raw dataset (37 FPS for the resolution of 380 x 1248 on RTX 4070 S)
<p align="center">
  <img width="600" height="300" src="./imgs/ESMStereo_L.gif" data-zoomable>
</p>

## Performance of ESMStereo-S on KITTI raw dataset (105 FPS for the resolution of 380 x 1248 on RTX 4070 S)
<p align="center">
  <img width="600" height="300" src="./imgs/ESMStereo_S.gif" data-zoomable>
</p>


# SOTA results.
<p align="center">
<table>
<tr><th>The results on SceneFlow </th><th>Performance on AGX Orin 64GB</th></tr>
<tr><td>

| Method | EPE <br> px|Runtime <br> (ms)|
|:-:|:-:|:-:|
| SADSNet-M-N7  | 1.16 | 8.5 |
| SADSNet-L-N7 | 0.90 | 13 |
| LightStereo-S | 0.73  | 17 |
| ADCPNet | 1.48 | 20 |
| IINet | 0.54 | 26 |
| Fast-ACVNet+ | 0.59 | 27 |
| RTSMNet-c8 | 0.71 | 28 |
| CGIStereo | 0.64 | 29 |
| FADNet++ | 0.76 | 33 |
| RT-IGEV++ | 0.55  | 42 |
| ------ | ------ | ------ |
| **ESMStereo-S-gwc**| 1.10  | 8.6 |
| **ESMStereo-M-gwc**| 0.77 | 14 |
| **ESMStereo-L-gwc**| **0.53** | 26 |

</td><td>

| Architecture |Performance <br> (FPS)|
|:-:|:-:|
| **ESMStereo-S-gwc**| 91 |
| **ESMStereo-M-gwc**| 29 |
| **ESMStereo-L-gwc**| 8.4 |

</td></tr> </table>
</p>

# How to use

## Environment
* NVIDIA RTX 4070 S
* Python 3.10
* Pytorch 2.5.1+cu118

## Install

```
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==1.0.11
```

## Data Preparation
* [SceneFlow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)

The structure of /datasets directory 

```shell
/datasets/
|-- ETH3D
|   |-- two_view_training
|   `-- two_view_training_gt
|-- Middlebury
|   |-- Eval3_GT
|   |-- MiddEval3-GT0-H
|   |-- MiddEval3-GT0-Q
|   |-- MiddEval3-data-H
|   |-- MiddEval3-data-Q
|   |-- testH
|   `-- trainingH
|-- SceneFlow
|   `-- flyingthings3d
|-- kitti_2012
|   |-- testing
|   `-- training
|-- kitti_2015
|   |-- testing
|   `-- training
|-- kittiraw
|   |-- 2011_09_26
`-- vkitti
    |-- vkitti_depth
    `-- vkitti_rgb
```

## Train

Use the following commands to train ESMStereo on SceneFlow.
First training,
```
python3 train_sceneflow.py --logdir checkpoints/Large  --cv gwc --cv_scale 4
python3 train_sceneflow.py --logdir checkpoints/Medium --cv gwc --cv_scale 8
python3 train_sceneflow.py --logdir checkpoints/Small  --cv gwc --cv_scale 16 --backbone mobilenetv2_100
```

Use the following commands to finetune ESMStereo on KITTI using the pretrained model on SceneFlow,
```
python3 train_kitti.py --logdir Large  --loadckpt checkpoint/esmstereo_L_gwc.ckpt --cv gwc --cv_scale 4
python3 train_kitti.py --logdir Medium --loadckpt checkpoint/esmstereo_M_gwc.ckpt --cv gwc --cv_scale 8
python3 train_kitti.py --logdir Small --loadckpt checkpoint/esmstereo_S_gwc.ckpt --cv gwc --cv_scale 16 --backbone mobilenetv2_100

```

### Pretrained Model
Download the trained weights folder and extract it in the root directory.
* [ESMStereo](place the link)

Generate disparity images of KITTI test set,
```
python save_disp.py
```
Generate performance tagged frames of KITTI raw dataset for making a video,
```
python save_vid.py
```

Use ``` --performance ``` to evaluate the performance on a targeted GPU,

```
python3 train_sceneflow.py --logdir checkpoints/Large  --cv gwc --cv_scale 4 --performance

```
Use ``` test_kitti.py ```, ``` test_mid.py ``` and ``` test_eth3d.py ``` for generalization results on KITTI, Middleburry, and ETH3D


# Citation

```

```
# Acknowledgements

Thanks to open source works: [CoEx](https://github.com/antabangun/coex), [ACVNet](https://github.com/gangweiX/Fast-ACVNet), [PSMNet](https://github.com/JiaRenChang/PSMNet?tab=readme-ov-file), [OpenStereo](https://github.com/XiandaGuo/OpenStereo/tree/v2).
