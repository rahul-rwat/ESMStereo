from __future__ import print_function, division

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import cv2

from PIL import Image
from torch.utils.data import DataLoader

from datasets import __datasets__
from models import __models__
from utils import *
from datasets.data_io import read_all_lines

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="ESMStereo")
parser.add_argument("--model", default="ESMStereo", help="select a model structure", choices=__models__.keys(),)
parser.add_argument('--backbone', default='efficientnet_b2', help='select a model structure', choices=["mobilenetv2_100", "efficientnet_b2"])
parser.add_argument("--maxdisp", type=int, default=192, help="maximum disparity")
parser.add_argument("--dataset", default="kitti", help="dataset name", choices=__datasets__.keys())
parser.add_argument("--datapath_raw", default="/datasets/kittiraw/2011_09_26/2011_09_26_drive_0009_sync", help="data path")
parser.add_argument("--testlist", default="./filenames/kitti_raw.txt", help="testing list")
parser.add_argument("--loadckpt", default="./checkpoint/esmstereo_S_gwc.ckpt", help="load the weights from a specific checkpoint",)
parser.add_argument('--cv_scale', type=int, default=16, help='cost volume scale factor', choices=[16, 8, 4])
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
test_dataset = StereoDataset(
    args.datapath_raw, args.datapath_raw, args.testlist, training=False
)
TestImgLoader = DataLoader(
    test_dataset, 1, shuffle=False, num_workers=4, drop_last=False
)

model = __models__[args.model](args.maxdisp, gwc, norm_correlation, args.backbone, args.cv_scale)

model = nn.DataParallel(model)
model.cuda()

print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model_dict = model.state_dict()
pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)

save_dir = "./test"

def load_path(list_filename):
    lines = read_all_lines(list_filename)
    splits = [line.split() for line in lines]
    left_images = [x[0] for x in splits]
    right_images = [x[1] for x in splits]

    return left_images, right_images


def test():
    os.makedirs(save_dir, exist_ok=True)
    fps_list = np.array([])
    im_left_list = []
    im_right_list = []
    left_img_host = []
    samples = []
    for batch_idx, sample in enumerate(TestImgLoader):

        left_filenames, right_filenames = load_path(args.testlist)

        left_name = left_filenames[batch_idx].split("/")[1]
        left_img = np.array(
            Image.open(os.path.join(args.datapath_raw, left_filenames[batch_idx]))
        )

        left_img_host.append(left_img)
        im_left_list.append(sample["left"].cuda())
        im_right_list.append(sample["right"].cuda())
        samples.append(sample)

    disps, fps_list = test_sample(im_left_list, im_right_list)

    for disp_gen, fps, sample, left_img in zip(disps, fps_list, samples, left_img_host):

        disp_est_np = tensor2numpy(disp_gen)

        print("fps = {:3f}".format(fps))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(
            disp_est_np, top_pad_np, right_pad_np, left_filenames
        ):
            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est, dtype=np.float32)

            fn = os.path.join(save_dir, fn.split("/")[-1])

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            disp_np = cv2.applyColorMap(
                cv2.convertScaleAbs(disp_est_uint, alpha=0.01), cv2.COLORMAP_JET
            )

            print("saving to", fn, disp_est.shape)
            out_img = np.concatenate((left_img, disp_np), 0)
            cv2.putText(
                out_img,
                "%.1f fps" % (fps),
                (10, left_img.shape[0] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(fn, out_img)


@make_nograd_func
def test_sample(im_left, im_right):

    model.eval()
    fps = []
    disps = []
    for rep in range(len(im_left)):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        disp_ests = model(im_left[rep], im_right[rep], train_status=False)
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        fps.append(1000 / runtime)
        disps.append(disp_ests[0])

    return disps, fps

if __name__ == "__main__":
    test()
