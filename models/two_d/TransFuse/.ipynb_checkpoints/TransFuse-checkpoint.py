import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision.models import resnet50
from .DeiT import deit_small_patch16_224 as deit
from .DeiT import deit_base_patch16_224 as deit_base
from .DeiT import deit_base_patch16_384 as deit_base_384
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ChannelPool(nn.Module):
    def forward(self, x):
        # [B,C,H,W] -> [B,2,H,W]  (max, mean)
        return torch.cat((torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g (transformer branch uses SE-like)
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l (cnn branch)
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W   = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp  = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g    = self.compress(g)
        g    = self.spatial(g)
        g    = self.sigmoid(g) * g_in

        # channel attention for transformer branch
        x_in = x
        x    = x.mean((2, 3), keepdim=True)
        x    = self.fc1(x)
        x    = self.relu(x)
        x    = self.fc2(x)
        x    = self.sigmoid(x) * x_in

        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class TransFuse_S(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_S, self).__init__()
        print('正在初始化TransFuse_S')
        self.resnet = resnet34()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-333f7ec4.pth', map_location='cpu'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = deit(pretrained=pretrained)

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate/2)

        self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        B, C, H, W = imgs.shape
        ph, pw = H // 16, W // 16

        # bottom-up path (ViT)
        x_b = self.transformer(imgs)                # [B, N, D], N=ph*pw, D=384
        x_b = torch.transpose(x_b, 1, 2).contiguous().view(B, -1, ph, pw)  # [B, 384, ph, pw]
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)     # [B,128, 2*ph, 2*pw]
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)   # [B, 64, 4*ph, 4*pw]
        x_b_2 = self.drop(x_b_2)  # transformer pred supervise here

        # top-down path (ResNet)
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)

        x_u   = self.resnet.layer3(x_u_1)
        x_u   = self.drop(x_u)

        # joint path
        x_c     = self.up_c(x_u, x_b)
        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1   = self.up_c_1_2(x_c, x_c_1_1)
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2   = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part (上采样回输入分辨率)
        map_x = F.interpolate(self.final_x(x_c), size=(H, W), mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), size=(H, W), mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), size=(H, W), mode='bilinear', align_corners=True)


        
        # return logit_x

        
        return map_x

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


class TransFuse_L(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_L, self).__init__()

        self.resnet = resnet50()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth', map_location='cpu'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = deit_base(pretrained=pretrained)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate/2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        B, C, H, W = imgs.shape
        ph, pw = H // 16, W // 16

        # bottom-up path
        x_b = self.transformer(imgs)                # [B, N=ph*pw, 768]
        x_b = torch.transpose(x_b, 1, 2).contiguous().view(B, -1, ph, pw)  # [B, 768, ph, pw]
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)   # [B,512, 2*ph, 2*pw]
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1) # [B,256, 4*ph, 4*pw]
        x_b_2 = self.drop(x_b_2)  # transformer pred supervise here

        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)

        x_u   = self.resnet.layer3(x_u_1)
        x_u   = self.drop(x_u)

        # joint path
        x_c     = self.up_c(x_u, x_b)
        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1   = self.up_c_1_2(x_c, x_c_1_1)
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2   = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), size=(H, W), mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), size=(H, W), mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), size=(H, W), mode='bilinear', align_corners=True)

        return map_x

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)
        

class TransFuse_L_384(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_L_384, self).__init__()

        self.resnet = resnet50()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth', map_location='cpu'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = deit_base_384(pretrained=pretrained)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate/2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        B, C, H, W = imgs.shape
        ph, pw = H // 16, W // 16

        # bottom-up path
        x_b = self.transformer(imgs)                # [B, ph*pw, 768]
        x_b = torch.transpose(x_b, 1, 2).contiguous().view(B, -1, ph, pw)  # [B,768, ph, pw]
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)   # [B,512, 2*ph, 2*pw]
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1) # [B,256, 4*ph, 4*pw]
        x_b_2 = self.drop(x_b_2)

        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)

        x_u   = self.resnet.layer3(x_u_1)
        x_u   = self.drop(x_u)

        # joint path
        x_c     = self.up_c(x_u, x_b)
        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1   = self.up_c_1_2(x_c, x_c_1_1)
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2   = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), size=(H, W), mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), size=(H, W), mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), size=(H, W), mode='bilinear', align_corners=True)

        return map_x

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = int(x2.size(2) - x1.size(2))
            diffX = int(x2.size(3) - x1.size(3))

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        self.need_skip = (inp_dim != out_dim)
        
    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        assert x.size(1) == self.inp_dim, f"{x.size(1)} vs {self.inp_dim}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
