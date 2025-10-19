import torch
from monai.networks.nets import SwinUNETR

# 2‑D 版：给 2 个尺寸 + 指明 spatial_dims=2
model = SwinUNETR(
    img_size=(256, 256),     # 仍然必填，长度与 spatial_dims 一致
    in_channels=3,
    out_channels=1,
    spatial_dims=2           # 声明 2‑D
)
if __name__ == '__main__':



    x = torch.randn(4, 3, 256, 256)
    y = model(x)
    print(y.shape)               # → torch.Size([4, 1, 256, 256])
