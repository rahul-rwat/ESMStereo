from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import cv2
import math
import gc
import time
import timm

from typing import List, Optional, Tuple

from .submodule import *
from .shufflemixer import FMBlock

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

class SubModule(nn.Module):
    def __init__(self) -> None:
        super(SubModule, self).__init__()

    def weight_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Feature(SubModule):
    def __init__(self, backbone: str) -> None:
        super(Feature, self).__init__()
        self.backbone = backbone
        pretrained =  True
        if backbone == "efficientnet_b2":
            model = timm.create_model('efficientnet_b2', pretrained=pretrained, features_only=True)
            layers = [1, 2, 3, 5, 6]
            self.chans = [16, 24, 48, 120, 208]
            self.conv_stem = model.conv_stem
            self.bn1 = model.bn1
            self.act1 = nn.ReLU6()


        elif backbone == "mobilenetv2_100":
            model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
            layers = [1, 2, 3, 5, 6]
            self.chans = [16, 24, 32, 96, 160]
            self.conv_stem = model.conv_stem
            self.bn1 = model.bn1
            self.act1 = nn.ReLU6()

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x: torch.Tensor) -> List[ torch.Tensor]:

        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        return [x2, x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self, chans: List[int], vol_size: int) -> None:
        super(FeatUp, self).__init__()
        self.v = vol_size
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

        if self.v == 16:
            self.conv16 = BasicConv(chans[3]*2, chans[2]*2, kernel_size=3, stride=1, padding=1)

        if self.v in [8, 4]:
            self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)

        if self.v == 8:
            self.conv8 = BasicConv(chans[2]*2, chans[2]*2, kernel_size=3, stride=1, padding=1)

        if self.v == 4:
            self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
            self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL: List[ torch.Tensor], featR: Optional[List[ torch.Tensor]]=None) -> Tuple[List[ torch.Tensor], List[ torch.Tensor]]:
        x2, x4, x8, x16, x32 = featL
        y2, y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)


        if self.v == 16:
            x16 = self.conv16(x16)
            y16 = self.conv16(y16)

        if self.v in [8, 4]:
            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

        if self.v == 8:
            x8 = self.conv8(x8)
            y8 = self.conv8(y8)

        if self.v == 4:
            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)
            x4 = self.conv4(x4)
            y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]



class aggregation(nn.Module):
    def __init__(self, in_channels: int, add_channel: int) -> None:
        super(aggregation, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels+add_channel, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels+add_channel, in_channels+add_channel, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=(1, 1, 1), stride=1, dilation=1))

        self.conv2 = nn.Sequential(BasicConv(in_channels+add_channel, in_channels+add_channel*2, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=(1, 1, 1), stride=2, dilation=1),
                                   BasicConv(in_channels+add_channel*2, in_channels+add_channel*2, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=(1, 1, 1), stride=1, dilation=1))

        self.conv3 = nn.Sequential(BasicConv(in_channels+add_channel*2, in_channels+add_channel*4, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels+add_channel*4, in_channels+add_channel*4, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels+add_channel*4, in_channels+add_channel*2, deconv=True, is_3d=True, bn=True,
                                  gelu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels+add_channel*2, in_channels+add_channel, deconv=True, is_3d=True, bn=True,
                                  gelu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels+add_channel, 1, deconv=True, is_3d=True, bn=False,
                                  gelu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))


        self.agg_0 = nn.Sequential(BasicConv(2*in_channels+add_channel*4, in_channels+add_channel*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels+add_channel*2, in_channels+add_channel*2, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(2*in_channels+add_channel*2, in_channels+add_channel, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels+add_channel, in_channels+add_channel, is_3d=True, kernel_size=3, padding=1, stride=1))


    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up[:, :, 0:conv2.shape[2], 0:conv2.shape[3], 0:conv2.shape[4]], conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up[:, :, 0:conv1.shape[2], 0:conv1.shape[3], 0:conv1.shape[4]], conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv = self.conv1_up(conv1)

        return conv


class up_refinement(nn.Module):
    def __init__(self, C: int, cf1: int, cf2: int) -> None:
        super(up_refinement, self).__init__()


        self.conv1 = nn.Sequential(BasicConv(1, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(C, C, deconv=True, is_3d=False, bn=True,
                                  gelu=True, kernel_size=4, padding=1, stride=2)

        self.conv2_up = BasicConv(C, C, deconv=True, is_3d=False, bn=True,
                                  gelu=True, kernel_size=4, padding=1, stride=2)

        self.conv1_up = BasicConv(C, 1, deconv=True, is_3d=False, bn=False,
                                  gelu=False, kernel_size=4, padding=1, stride=2)


        self.agg_0 = nn.Sequential(BasicConv(2*C+cf1, C, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(2*C+cf2, C, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1))

    def forward(self, disp: torch.Tensor, left_f1x: torch.Tensor, left_f2x: torch.Tensor) ->  torch.Tensor:

        conv1 = self.conv1(disp)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up[:, 0:conv2.shape[1], 0:conv2.shape[2], 0:conv2.shape[3]], conv2, left_f1x), dim=1)
        conv2 = self.agg_0(conv2)

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1, left_f2x), dim=1)

        conv1 = self.agg_1(conv1)
        conv = self.conv1_up(conv1)

        return conv


class upsample4(nn.Module):
    def __init__(self) -> None:
        super(upsample4, self).__init__()

        C2x = 32
        cf1 = 96
        cf2 = 48

        self.dm2x = nn.Sequential(BasicConv(1, C2x, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_2x = nn.Sequential(BasicConv(C2x+cf2, C2x, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(C2x, C2x, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(C2x), nn.GELU()
                                    )

        n_feats = 16
        self.to_feat = nn.Conv2d(C2x, n_feats, 3, 1, 1, bias=False)
        n_blocks = 2
        self.blocks = nn.Sequential(*[FMBlock(n_feats, 7, 2) for _ in range(n_blocks)])

        self.upsampling2 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True))

        self.tail2x = nn.Conv2d(n_feats, 1, 3, 1, 1)

        self.ref2x = up_refinement(C2x, cf1, cf2)

        C4x = 32
        cf1 = 48
        cf2 = 32

        self.dm4x = nn.Sequential(BasicConv(1, C4x, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_4x = nn.Sequential(BasicConv(C4x+cf2, C4x, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(C4x, C4x//2, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(C4x//2), nn.GELU())


        self.upsampling4 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True))

        self.tail4x = nn.Conv2d(n_feats, 1, 3, 1, 1)
        self.ref4x = up_refinement(C4x, cf1, cf2)

    def forward(self, left_f1x: torch.Tensor, left_f2x: torch.Tensor, left_f4x: torch.Tensor, init_disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        disp_features2x = self.dm2x(init_disp)
        cat_features2x = self.spx_2x(torch.cat((disp_features2x, left_f2x), dim=1))

        x = self.to_feat(cat_features2x)
        x = self.blocks(x)
        x2 = self.upsampling2(x)
        x2 = self.tail2x(x2)
        x2 = self.ref2x(x2, left_f1x, left_f2x)

        upsampled2x_disp = F.interpolate(init_disp, scale_factor=2, mode='bilinear', align_corners=False) + x2

        disp_features4x = self.dm4x(upsampled2x_disp)
        cat_features4x = self.spx_4x(torch.cat((disp_features4x, left_f4x), dim=1))

        x4 = self.upsampling4(cat_features4x)
        x4 = self.tail4x(x4)
        x4 = self.ref4x(x4, left_f2x, left_f4x)

        upsampled4x_disp = F.interpolate(upsampled2x_disp, scale_factor=2, mode='bilinear', align_corners=False) + x4

        return upsampled4x_disp, upsampled2x_disp

class upsample8(nn.Module):
    def __init__(self):
        super(upsample8, self).__init__()

        C2x = C4x = C8x = 16
        n_feats = 8

        cf1 = 240
        cf2 = 96

        self.dm2x = nn.Sequential(BasicConv(1, C2x, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_2x = nn.Sequential(BasicConv(C2x+cf2, C2x, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(C2x, C2x, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(C2x), nn.GELU()
                                    )

        self.to_feat = nn.Conv2d(C2x, n_feats, 3, 1, 1, bias=False)
        n_blocks = 2
        self.blocks = nn.Sequential(*[FMBlock(n_feats, 7, 2) for _ in range(n_blocks)])

        self.upsampling2 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True))

        self.tail2x = nn.Conv2d(n_feats, 1, 3, 1, 1)

        self.ref2x = up_refinement(C2x, cf1, cf2)

        cf1 = 96
        cf2 = 24

        self.dm4x = nn.Sequential(BasicConv(1, C4x, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_4x = nn.Sequential(BasicConv(C4x+cf2, C4x, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(C4x, C4x//2, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(C4x//2), nn.GELU())


        self.upsampling4 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True))

        self.tail4x = nn.Conv2d(n_feats, 1, 3, 1, 1)
        self.ref4x = up_refinement(C4x, cf1, cf2)

        cf1 = 24
        cf2 = 32

        self.dm8x = nn.Sequential(BasicConv(1, C8x, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C8x, C8x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C8x, C8x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C8x, C8x, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_8x = nn.Sequential(BasicConv(C8x+cf2, C8x, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(C8x, C8x//2, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(C8x//2), nn.GELU())


        self.upsampling8 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True))

        self.tail8x = nn.Conv2d(n_feats, 1, 3, 1, 1)
        self.ref8x = up_refinement(C8x, cf1, cf2)


    def forward(self, left_f2x, left_f4x, left_f8x, stem_f2, init_disp):

        disp_features2x = self.dm2x(init_disp)
        cat_features2x = self.spx_2x(torch.cat((disp_features2x, left_f4x), dim=1))

        x = self.to_feat(cat_features2x)
        x = self.blocks(x)
        x2 = self.upsampling2(x)
        x2 = self.tail2x(x2)
        x2 = self.ref2x(x2, left_f2x, left_f4x)

        upsampled2x_disp = F.interpolate(init_disp, scale_factor=2, mode='bilinear', align_corners=False) + x2

        disp_features4x = self.dm4x(upsampled2x_disp)
        cat_features4x = self.spx_4x(torch.cat((disp_features4x, left_f8x), dim=1))

        x4 = self.upsampling4(cat_features4x)
        x4 = self.tail4x(x4)
        x4 = self.ref4x(x4, left_f4x, left_f8x)

        upsampled4x_disp = F.interpolate(upsampled2x_disp, scale_factor=2, mode='bilinear', align_corners=False) + x4

        disp_features8x = self.dm8x(upsampled4x_disp)
        cat_features8x = self.spx_8x(torch.cat((disp_features8x, stem_f2), dim=1))

        x8 = self.upsampling8(cat_features8x)
        x8 = self.tail8x(x8)
        x8 = self.ref8x(x8, left_f8x, stem_f2)

        upsampled8x_disp = F.interpolate(upsampled4x_disp, scale_factor=2, mode='bilinear', align_corners=False) + x8


        return upsampled8x_disp, upsampled4x_disp, upsampled2x_disp,

class upsample16(nn.Module):
    def __init__(self) -> None:
        super(upsample16, self).__init__()

        C2x = 16
        cf1 = 32
        cf2 = 32

        self.dm2x = nn.Sequential(BasicConv(1, C2x, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C2x, C2x, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_2x = nn.Sequential(BasicConv(C2x+cf2, C2x, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(C2x, C2x, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(C2x), nn.GELU()
                                    )

        n_feats = 8
        self.to_feat = nn.Conv2d(C2x, n_feats, 3, 1, 1, bias=False)
        n_blocks = 2
        self.blocks = nn.Sequential(*[FMBlock(n_feats, 7, 2) for _ in range(n_blocks)])

        self.upsampling2 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 16, 1, 1, 0),
                nn.PixelShuffle(4),
                nn.SiLU(inplace=True))

        self.tail2x = nn.Conv2d(n_feats, 1, 3, 1, 1)

        self.ref2x = up_refinement(C2x, cf1, cf2)

        C4x = 16
        cf1 = 24
        cf2 = 24

        self.dm4x = nn.Sequential(BasicConv(1, C4x, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C4x, C4x, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_4x = nn.Sequential(BasicConv(C4x+cf2, C4x, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(C4x, C4x//2, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(C4x//2), nn.GELU())


        self.upsampling4 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 16, 1, 1, 0),
                nn.PixelShuffle(4),
                nn.SiLU(inplace=True))

        self.tail4x = nn.Conv2d(n_feats, 1, 3, 1, 1)
        self.ref4x = up_refinement(C4x, cf1, cf2)

    def forward(self, left_f1x: torch.Tensor, left_f2x: torch.Tensor, left_f4x: torch.Tensor, left_f8x: torch.Tensor, init_disp: torch.Tensor) -> Tuple[ torch.Tensor,  torch.Tensor]:

        disp_features2x = self.dm2x(init_disp)

        cat_features2x = self.spx_2x(torch.cat((disp_features2x, left_f2x), dim=1))

        x = self.to_feat(cat_features2x)
        x = self.blocks(x)
        x2 = self.upsampling2(x)
        x2 = self.tail2x(x2)

        x2 = self.ref2x(x2, left_f2x, left_f1x)

        upsampled2x_disp = F.interpolate(init_disp, scale_factor=4, mode='bilinear', align_corners=False) + x2

        disp_features4x = self.dm4x(upsampled2x_disp)

        cat_features4x = self.spx_4x(torch.cat((disp_features4x, left_f4x), dim=1))

        x4 = self.upsampling4(cat_features4x)
        x4 = self.tail4x(x4)
        x4 = self.ref4x(x4, left_f4x, left_f8x)

        upsampled4x_disp = F.interpolate(upsampled2x_disp, scale_factor=4, mode='bilinear', align_corners=False) + x4

        return upsampled4x_disp, upsampled2x_disp

class conf_upsample(nn.Module):
    def __init__(self, C: int, fc: int) -> None:
        super(conf_upsample, self).__init__()


        self.conv1 = BasicConv(1, C, is_3d=False, bn=True, gelu=True, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3, padding=1, stride=2, dilation=1)
        self.conv1_up = BasicConv(C, 1, deconv=True, is_3d=False, bn=True, gelu=True, kernel_size=4, padding=1, stride=2)

        self.cm = nn.Sequential(BasicConv(1, C, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.conf_spx_4 = nn.Sequential(BasicConv(C+fc, C, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(C, C, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(C), nn.ReLU()
                                   )

        self.conf_spx = nn.ConvTranspose2d(C, 9, kernel_size=4, stride=4, padding=0)

    def forward(self, left_f1x: torch.Tensor, init_conf: torch.Tensor) -> torch.Tensor:

        conf_features = self.cm(init_conf)
        cat_features = self.conf_spx_4(torch.cat((conf_features, left_f1x), dim=1))
        cat_features = self.conf_spx(cat_features)
        sfm = F.softmax(cat_features, 1)

        b, c, h, w = init_conf.shape
        conf_unfold = F.unfold(init_conf, 3, 1, 1).reshape(b, -1, h, w)
        conf_unfold = F.interpolate(conf_unfold, (h * 4, w * 4), mode='nearest').reshape(b, 9, h * 4, w * 4)

        conf1 = (conf_unfold * sfm).sum(1).unsqueeze(1)
        conf = self.conv1(conf1)
        conf = self.conv2(conf)
        conf = self.conv1_up(conf)

        return conf + conf1


class LAFNet_ESM(nn.Module):
    def __init__(self, C):
        super(LAFNet_ESM, self).__init__()
        self.C = C

        # cost feature extractor
        self.softmax = nn.Softmax(dim=1)
        self.cost_conv1 = nn.Conv2d(7, C, kernel_size=3, padding=1)
        self.cost_bn1 = nn.BatchNorm2d(C)
        self.cost_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.cost_bn2 = nn.BatchNorm2d(C)
        self.cost_conv3 = nn.Conv2d(C, C, kernel_size=1, padding=0)
        self.cost_bn3 = nn.BatchNorm2d(C)

        # disparity feature extractor
        self.disp_conv1 = nn.Conv2d( 1, C, kernel_size=3, padding=1)
        self.disp_bn1 = nn.BatchNorm2d(C)
        self.disp_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.disp_bn2 = nn.BatchNorm2d(C)
        self.disp_conv3 = nn.Conv2d(C, C, kernel_size=1, padding=0)
        self.disp_bn3 = nn.BatchNorm2d(C)

        # image feature extractor
        self.imag_conv1 = nn.Conv2d(64, C, kernel_size=3, padding=1)
        self.imag_bn1 = nn.BatchNorm2d(C)
        self.imag_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.imag_bn2 = nn.BatchNorm2d(C)
        self.imag_conv3 = nn.Conv2d(C, C, kernel_size=1, padding=0)
        self.imag_bn3 = nn.BatchNorm2d(C)

        # cost attention extractor
        self.cost_att_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.cost_att_bn1 = nn.BatchNorm2d(C)
        self.cost_att_conv2 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.cost_att_bn2 = nn.BatchNorm2d(1)

        # disparity attention extractor
        self.disp_att_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.disp_att_bn1 = nn.BatchNorm2d(C)
        self.disp_att_conv2 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.disp_att_bn2 = nn.BatchNorm2d(1)

        # image attention extractor
        self.imag_att_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.imag_att_bn1 = nn.BatchNorm2d(C)
        self.imag_att_conv2 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.imag_att_bn2 = nn.BatchNorm2d(1)

        self.softmax_att = nn.Softmax(dim=1)

        # scale extractor
        self.scale_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.scale_bn1 = nn.BatchNorm2d(C)
        self.scale_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.scale_bn2 = nn.BatchNorm2d(C)
        self.scale_conv3 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.scale_bn3 = nn.BatchNorm2d(1)

        # embedding
        self.embed_conv1 = nn.Conv2d(3 * C, C, kernel_size=3, padding=1)
        self.embed_bn1 = nn.BatchNorm2d(C)
        self.embed_conv2 = nn.Conv2d( C, C, kernel_size=3, padding=0, stride=3)
        self.embed_bn2 = nn.BatchNorm2d(C)

        # predictor
        self.fusion_conv1 = nn.Conv2d(C + 1, C, kernel_size=3, padding=1)
        self.fusion_bn1_iter1 = nn.BatchNorm2d(C)
        self.fusion_bn1_iter2 = nn.BatchNorm2d(C)
        self.fusion_bn1_iter3 = nn.BatchNorm2d(C)
        self.fusion_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.fusion_bn2_iter1 = nn.BatchNorm2d(C)
        self.fusion_bn2_iter2 = nn.BatchNorm2d(C)
        self.fusion_bn2_iter3 = nn.BatchNorm2d(C)
        self.fusion_conv3 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.fusion_bn3_iter1 = nn.BatchNorm2d(1)
        self.fusion_bn3_iter2 = nn.BatchNorm2d(1)
        self.fusion_bn3_iter3 = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

        self.conf_up4 = conf_upsample(C, 96)
        self.conf_up1 = conf_upsample(C, 24)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.scale_bn3.weight, 0)
        nn.init.constant_(self.scale_bn3.bias, 0)


    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, cost, disp, imag, left_f1x, left_f2x, device):
        x = self.softmax(-self.L2normalize(cost) * 100)
        x = torch.topk(x, k=7, dim=1).values

        x = F.relu(self.cost_bn1(self.cost_conv1(x)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        cost_x = F.relu(self.cost_bn3(self.cost_conv3(x)))

        x = F.relu(self.disp_bn1(self.disp_conv1(disp)))
        x = F.relu(self.disp_bn2(self.disp_conv2(x)))
        disp_x = F.relu(self.disp_bn3(self.disp_conv3(x)))

        x = F.relu(self.imag_bn1(self.imag_conv1(imag)))
        x = F.relu(self.imag_bn2(self.imag_conv2(x)))
        imag_x = F.relu(self.imag_bn3(self.imag_conv3(x)))

        x = F.relu(self.cost_att_bn1(self.cost_att_conv1(cost_x)))
        cost_att_x = self.cost_att_bn2(self.cost_att_conv2(x))

        x = F.relu(self.disp_att_bn1(self.disp_att_conv1(disp_x)))
        disp_att_x = self.disp_att_bn2(self.disp_att_conv2(x))

        x = F.relu(self.imag_att_bn1(self.imag_att_conv1(imag_x)))
        imag_att_x = self.imag_att_bn2(self.imag_att_conv2(x))

        att_x = self.softmax_att(torch.cat((cost_att_x, disp_att_x, imag_att_x), 1))

        cost_att_x = att_x[:,0,:,:].unsqueeze(1)
        disp_att_x = att_x[:,1,:,:].unsqueeze(1)
        imag_att_x = att_x[:,2,:,:].unsqueeze(1)

        cost_x = cost_x * (cost_att_x.repeat(1, self.C, 1, 1))
        disp_x = disp_x * (disp_att_x.repeat(1, self.C, 1, 1))
        imag_x = imag_x * (imag_att_x.repeat(1, self.C, 1, 1))

        x = torch.cat((cost_x, disp_x, imag_x), 1)
        feat = F.relu(self.embed_bn1(self.embed_conv1(x)))

        x = F.relu(self.scale_bn1(self.scale_conv1(feat)))
        x = F.relu(self.scale_bn2(self.scale_conv2(x)))
        scale = 2 * self.sigmoid(self.scale_bn3(self.scale_conv3(x)))

        b, c, h, w = disp.size()

        grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        grid_h = torch.tensor(grid_h, dtype=torch.float, requires_grad=False).to(device).repeat(b, 1, 1, 1)
        grid_w = torch.tensor(grid_w, dtype=torch.float, requires_grad=False).to(device).repeat(b, 1, 1, 1)
        grid = torch.cat((grid_w, grid_h), 1).transpose(1, 2).transpose(2, 3)

        scale_t = scale.transpose(1, 2).transpose(2, 3)

        grid_enlarge = torch.zeros([b, 3 * h, 3 * w, 2]).to(device)

        step_x = 2 / (h - 1)
        step_y = 2 / (w - 1)

        grid_enlarge[:,0::3,0::3,:] = grid + torch.cat(((-1) * step_y * scale_t , (-1) * scale_t), 3)
        grid_enlarge[:,0::3,1::3,:] = grid + torch.cat((( 0) * step_y * scale_t , (-1) * scale_t), 3)
        grid_enlarge[:,0::3,2::3,:] = grid + torch.cat(((+1) * step_y * scale_t , (-1) * scale_t), 3)
        grid_enlarge[:,1::3,0::3,:] = grid + torch.cat(((-1) * step_y * scale_t , ( 0) * scale_t), 3)
        grid_enlarge[:,1::3,1::3,:] = grid + torch.cat((( 0) * step_y * scale_t , ( 0) * scale_t), 3)
        grid_enlarge[:,1::3,2::3,:] = grid + torch.cat(((+1) * step_y * scale_t , ( 0) * scale_t), 3)
        grid_enlarge[:,2::3,0::3,:] = grid + torch.cat(((-1) * step_y * scale_t , (+1) * scale_t), 3)
        grid_enlarge[:,2::3,1::3,:] = grid + torch.cat((( 0) * step_y * scale_t , (+1) * scale_t), 3)
        grid_enlarge[:,2::3,2::3,:] = grid + torch.cat(((+1) * step_y * scale_t , (+1) * scale_t), 3)

        feat_enlarge = F.grid_sample(feat, grid_enlarge, align_corners=True)

        feat = F.relu(self.embed_bn2(self.embed_conv2(feat_enlarge)))

        out = torch.zeros([b, c, h, w], dtype=torch.float).to(device) + 0.5

        x = torch.cat((feat, out), 1)

        x = F.relu(self.fusion_bn1_iter1(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter1(self.fusion_conv2(x)))
        out = F.relu(self.fusion_bn3_iter1(self.fusion_conv3(x)))

        x = torch.cat((feat, out), 1)

        x = F.relu(self.fusion_bn1_iter2(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter2(self.fusion_conv2(x)))
        out = F.relu(self.fusion_bn3_iter2(self.fusion_conv3(x)))

        x = torch.cat((feat, out), 1)

        x = F.relu(self.fusion_bn1_iter3(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter3(self.fusion_conv2(x)))
        init_conf = F.relu(self.fusion_bn3_iter3(self.fusion_conv3(x)))

        out4 = self.conf_up4(left_f1x, init_conf)
        out1 = self.conf_up1(left_f2x, out4)

        return self.sigmoid(out1)

class ESMStereo_confidence(nn.Module):
    def __init__(self, maxdisp: int,  gwc: bool=False, norm_correlation: bool=True, backbone: str="efficientnet_b2",  cv_scale: int=4, device=torch.device("cuda")) -> None:
        super(ESMStereo_confidence, self).__init__()
        self.device = device
        self.maxdisp = maxdisp
        self.vol_size = cv_scale
        self.gwc = gwc
        self.norm_correlation = norm_correlation
        self.backbone = backbone

        self.feature = Feature(self.backbone)

        if self.vol_size in [4, 8]:
            self.feature_up = FeatUp(self.feature.chans, self.vol_size)
        else:
            pass

        if self.vol_size == 4:
            self.stem_2 = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
                )

            self.stem_4 = nn.Sequential(
                BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                nn.BatchNorm2d(48), nn.ReLU()
                )

        if self.vol_size == 8:
            self.stem_2 = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
                )

            self.stem_4 = nn.Sequential(
                BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                nn.BatchNorm2d(48), nn.ReLU()
                )

            self.stem_8 = nn.Sequential(
                BasicConv(48, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU()
                )

        if self.vol_size == 16:
            self.stem_2 = nn.Sequential(
                BasicConv(3, 16, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU()
                )

            self.stem_4 = nn.Sequential(
                BasicConv(16, 24, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(24, 24, 3, 1, 1, bias=False),
                nn.BatchNorm2d(24), nn.ReLU()
                )

            self.stem_8 = nn.Sequential(
                BasicConv(24, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
                )

            self.stem_16 = nn.Sequential(
                BasicConv(32, 40, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(40, 40, 3, 1, 1, bias=False),
                nn.BatchNorm2d(40), nn.ReLU()
                )

        if self.vol_size == 4:
            self.conv = BasicConv(96, 64, kernel_size=3, padding=1, stride=1)
            self.desc = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)

        elif self.vol_size == 8:
            self.conv = BasicConv(160, 64, kernel_size=3, padding=1, stride=1)
            self.desc = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)

        elif self.vol_size == 16:
            self.conv = BasicConv(136, 64, kernel_size=3, padding=1, stride=1)
            self.desc = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
            self.conv_f2 = BasicConv(96, 32, kernel_size=3, padding=1, stride=1)
            self.conv_f0 = BasicConv(16, 24, kernel_size=3, padding=1, stride=1)
        else:
            pirnt("Choose the cost volume resolution: 4, 8, 16")


        reduction_multiplier = 8
        if self.norm_correlation:
            print("Cost volumes: norm correlation")
            if self.vol_size == 16:
                self.semantic = nn.Sequential(
                    BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(32, 8, 3, 1, 1, bias=False),
                    )
            self.corr_stem = BasicConv(1, reduction_multiplier, deconv=False, is_3d=True, bn=True, gelu=True, kernel_size=3, padding=1, stride=1)

        if self.gwc:
            print("Cost volumes: gwc ")
            if self.vol_size == 16:
                self.semantic = nn.Sequential(
                    BasicConv(96, 64, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                    )
            self.num_groups = 32
            self.group_stem = BasicConv(self.num_groups, reduction_multiplier, deconv=False, is_3d=True, bn=True, gelu=True, kernel_size=3, padding=1, stride=1)


        self.agg = BasicConv(reduction_multiplier, reduction_multiplier, deconv=False, is_3d=True, bn=True, gelu=True, kernel_size=3, padding=1, stride=1)

        if self.vol_size == 4:
            add_channel = 16
            self.upsample_module = upsample4()

        elif self.vol_size == 8:
            add_channel = 8
            self.upsample_module = upsample8()

        elif self.vol_size == 16:
            add_channel = 4
            self.upsample_module = upsample16()
            self.confidence_net = LAFNet_ESM(16)


        self.aggregation_out = aggregation(reduction_multiplier, add_channel)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> List[ torch.Tensor]:

        features_left = self.feature(left)
        features_right = self.feature(right)

        if self.vol_size in [4, 8]:
            features_left, features_right = self.feature_up(features_left, features_right)
        else:
            pass

        if self.vol_size == 4:
            stem_2x = self.stem_2(left)
            stem_2y = self.stem_2(right)

            stem_4x = self.stem_4(stem_2x)
            stem_4y = self.stem_4(stem_2y)

            match_left = torch.cat((features_left[0], stem_4x), 1)
            match_right = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(match_left))
            match_right = self.desc(self.conv(match_right))

        if self.vol_size == 8:
            stem_2x = self.stem_2(left)
            stem_2y = self.stem_2(right)

            stem_4x = self.stem_4(stem_2x)
            stem_4y = self.stem_4(stem_2y)

            stem_8x = self.stem_8(stem_4x)
            stem_8y = self.stem_8(stem_4y)

            match_left = torch.cat((features_left[1], stem_8x), 1)
            match_right = torch.cat((features_right[1], stem_8y), 1)

            match_left = self.desc(self.conv(match_left))
            match_right = self.desc(self.conv(match_right))

        if self.vol_size == 16:

            stem_2x = self.stem_2(left)
            stem_2y = self.stem_2(right)

            stem_4x = self.stem_4(stem_2x)
            stem_4y = self.stem_4(stem_2y)

            stem_8x = self.stem_8(stem_4x)
            stem_8y = self.stem_8(stem_4y)

            stem_16x = self.stem_16(stem_8x)
            stem_16y = self.stem_16(stem_8y)

            match_left = torch.cat((features_left[3], stem_16x), 1)
            match_right = torch.cat((features_right[3], stem_16y), 1)

            match_left = self.desc(self.conv(match_left))
            match_right = self.desc(self.conv(match_right))

            att = self.semantic(features_left[3]).unsqueeze(2)


        if self.norm_correlation:
            volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp // self.vol_size)
            if self.vol_size == 16:
                volume = self.corr_stem(volume) * att
            else:
                volume = self.corr_stem(volume)

        if self.gwc:
            volume = build_gwc_volume(match_left, match_right, self.maxdisp // self.vol_size, self.num_groups)

            if self.vol_size == 16:
                volume = self.group_stem(volume * att)
            else:
                volume = self.group_stem(volume)

        volume = self.agg(volume)
        cost = self.aggregation_out(volume)

        if self.vol_size == 4:
            disp_samples = torch.arange(0, self.maxdisp // self.vol_size, dtype=cost.dtype, device=cost.device)
            disp_samples = disp_samples.view(1, self.maxdisp // self.vol_size, 1, 1).repeat(cost.shape[0], 1, cost.shape[3], cost.shape[4])
            init_pred = regression_topk(cost.squeeze(1), disp_samples, 2)
            disp_1, disp_2  = self.upsample_module(features_left[1], features_left[0], stem_2x, init_pred)

        if self.vol_size == 8:
            init_pred = disparity_regression(cost.squeeze(1), self.maxdisp // self.vol_size).unsqueeze(1)
            disp_1, disp_2, disp_4  = self.upsample_module(features_left[2], features_left[1], features_left[0], stem_2x, init_pred)

        if self.vol_size == 16:

            init_pred = disparity_regression(cost.squeeze(1), self.maxdisp // self.vol_size).unsqueeze(1)
            f2 = self.conv_f2(features_left[3])
            f0 = self.conv_f0(features_left[0])
            disp_1, disp_2  = self.upsample_module(features_left[2], f2, features_left[1], f0, init_pred)
            conf_out = self.confidence_net(cost.squeeze(1), init_pred, match_left, features_left[3], features_left[1],self.device)

        return disp_1.squeeze(1)*4, conf_out.squeeze(1)


