import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image, ImageTk
import tkinter as tk
from torchvision import transforms
from models import __models__
from utils import tensor2numpy, make_nograd_func

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def configure_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
    profile = pipeline.start(config)

    ir_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 1))
    ir_intrinsics = ir_profile.get_intrinsics()

    try:
        ir1_to_ir2 = profile.get_stream(rs.stream.infrared, 1).get_extrinsics_to(
            profile.get_stream(rs.stream.infrared, 2))
        baseline = abs(ir1_to_ir2.translation[0])
    except:
        baseline = 0.05

    focal_length = ir_intrinsics.fx
    return pipeline, baseline, focal_length

def preprocess_image(img):
    img = Image.fromarray(img).convert('RGB')
    w, h = img.size
    m = 32
    wi, hi = (w // m + 1) * m, (h // m + 1) * m
    img = img.crop((w - wi, h - hi, w, h))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).cuda()

@make_nograd_func
def compute_disparity(model, left_tensor, right_tensor):
    model.eval()
    disp_ests = model(left_tensor, right_tensor, train_status=False)
    return disp_ests[-1]

def disparity_to_depth_map(disparity_map, baseline, focal_length, max_depth=5.0):
    disparity_map = np.where(disparity_map <= 0, 1e-6, disparity_map)
    depth_map = (baseline * focal_length) / disparity_map
    depth_map = np.clip(depth_map, 0, max_depth)
    return depth_map

def create_depth_colormap(depth_map, max_depth=5.0):
    depth_normalized = np.clip(depth_map / max_depth, 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_JET)
    return depth_colored

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ESMStereo', choices=__models__.keys())
    parser.add_argument('--backbone', default='efficientnet_b2', choices=["mobilenetv2_100", "efficientnet_b2"])
    parser.add_argument('--maxdisp', type=int, default=192)
    parser.add_argument('--loadckpt', default='/home/enord/ESMStereo/checkpoint_ESMStereo/esmstereo_L_nc.ckpt')
    parser.add_argument('--cv_scale', type=int, default=4, choices=[16, 8, 4])
    parser.add_argument('--cv', default='norm_correlation', choices=['norm_correlation', 'gwc'])
    parser.add_argument('--max_depth', type=float, default=5.0)
    args = parser.parse_args()

    gwc = args.cv == 'gwc'
    norm_correlation = args.cv == 'norm_correlation'

    model = __models__[args.model](args.maxdisp, gwc, norm_correlation, args.backbone, args.cv_scale)
    model = nn.DataParallel(model).cuda().eval()

    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    pipeline, baseline, focal_length = configure_realsense()

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Depth Map Viewer")
    label = tk.Label(root)
    label.pack()

    # Exit with 'q' key
    root.bind("<q>", lambda e: root.quit())

    try:
        while True:
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)

            if not left_frame or not right_frame:
                continue

            left_img = np.asanyarray(left_frame.get_data())
            right_img = np.asanyarray(right_frame.get_data())

            left_tensor = preprocess_image(left_img)
            right_tensor = preprocess_image(right_img)

            disp_tensor = compute_disparity(model, left_tensor, right_tensor)
            disp_np = tensor2numpy(disp_tensor)[0]

            depth_map = disparity_to_depth_map(disp_np, baseline, focal_length, max_depth=args.max_depth)
            depth_colored = create_depth_colormap(depth_map, max_depth=args.max_depth)
            depth_colored_resized = cv2.resize(depth_colored, (1280, 720))
            depth_rgb = cv2.cvtColor(depth_colored_resized, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(depth_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            label.config(image=img_tk)
            label.image = img_tk

            root.update_idletasks()
            root.update()

    except tk.TclError:
        print("GUI closed.")

    pipeline.stop()
    root.destroy()

if __name__ == '__main__':
    main()
