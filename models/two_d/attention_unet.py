import torch
from monai.networks.nets import AttentionUnet



if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionUnet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,  # ← 改回奇数
        dropout=0.0,
    ).to(device)
    model.eval()

    x = torch.randn(4, 3, 256, 256, device=device)  # 随机输入 (B=4, C=3, H=W=256)

    with torch.no_grad():
        y = model(x)

    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")  # 预期：(4, out_channels, 256, 256)
    print(f"output dtype: {y.dtype}, device: {y.device}")
    print(f"output stats -> min: {y.min().item():.4f}, max: {y.max().item():.4f}, mean: {y.mean().item():.4f}")
