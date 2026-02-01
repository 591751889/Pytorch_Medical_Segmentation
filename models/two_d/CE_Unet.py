import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def mean_channels(F):
    assert F.dim() == 4  # [B, C, H, W]
    return F.mean(dim=(2, 3), keepdim=True)

def stdv_channels(F):
    assert F.dim() == 4
    mean = mean_channels(F)
    var = ((F - mean) ** 2).mean(dim=(2, 3), keepdim=True)
    return torch.sqrt(var + 1e-6)


class CEAM(nn.Module):
    def __init__(self, channels, kernels=[3, 5]):
        super().__init__()
        self.c=channels
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False))
        self.softmax=nn.Softmax(dim=0)
    def forward(self,x):
        bs,c,h,w=x.size()
        avg_pool = x.mean(dim=(2, 3), keepdim=True)
        contrast = stdv_channels(x)
        y= contrast + avg_pool
        y=y.squeeze(-1).squeeze(-1).unsqueeze(1)
        conv_outs=[]
        for conv in self.convs:
            out=conv(y)
            out=torch.sigmoid(out)
            out=out.squeeze(1).unsqueeze(-1).unsqueeze(-1)
            conv_outs.append(out)
        feats_channels=torch.stack(conv_outs,0)
        fuse_attention=(attention_weights*feats_channels).sum(0)
        
        return x*fuse_attention
        
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mr1=CEAM(mid_channels,kernels=[3, 5])
        self.mr2=CEAM(out_channels,kernels=[3, 5])

    def forward(self, x):
        x=self.double_conv[0](x)
        x=self.double_conv[1](x)
        x=self.double_conv[2](x)
        x=self.mr1(x)
        x=self.double_conv[3](x)
        x=self.double_conv[4](x)
        x=self.double_conv[5](x)
        x=self.mr2(x)
        return x


class Down(nn.Module):
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class CE_Unet(nn.Module):
    def __init__(self, in_channels, classes,bilinear=False):
        super(CE_Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
