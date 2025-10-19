from monai.networks.nets import UNETR
import torch

net = UNETR(in_channels=1, out_channels=4, img_size=256, feature_size=32, norm_name='batch', spatial_dims=2)

if __name__ == '__main__':
    x = torch.randn(4, 1, 256, 256)
    y = net(x)
    print(y.shape)               # → torch.Size([4, 1, 256, 256])