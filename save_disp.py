from __future__ import print_function, division

import os
import time

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import skimage
import cv2

from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='ESMStereo')
parser.add_argument('--model', default='ESMStereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--backbone', default='efficientnet_b2', help='select a model structure', choices=["mobilenetv2_100", "efficientnet_b2"])
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath_12', default='/datasets/kitti_2012/', help='data path')
parser.add_argument('--datapath_15', default='/datasets/kitti_2015/', help='data path')
parser.add_argument('--testlist',default='./filenames/kitti15_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./checkpoint/esmstereo_L_gwc.ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--cv_scale', type=int, default=4, help='cost volume scale factor', choices=[16, 8, 4])

parser.add_argument('--cv', type=str, default='gwc', choices=[
          'norm_correlation',
          'gwc',
], help='selecting a cost volumes')

args = parser.parse_args()

gwc = False
norm_correlation = False
if args.cv == 'norm_correlation':
    norm_correlation = True
elif args.cv == 'gwc':
    gwc = True

StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

model = __models__[args.model](args.maxdisp, gwc, norm_correlation, args.backbone, args.cv_scale)
model = nn.DataParallel(model)
model.cuda()

print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model_dict = model.state_dict()
pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)

save_dir = './test'

def test():
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        torch.cuda.synchronize()
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)

            fn = os.path.join(save_dir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)

            if False:
                cv2.imwrite(
                    fn,
                    cv2.applyColorMap(
                        cv2.convertScaleAbs(disp_est_uint, alpha=0.01), cv2.COLORMAP_JET
                    ),
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
                )

@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda(), train_status=False)
    return disp_ests[-1]

if __name__ == '__main__':
    test()
