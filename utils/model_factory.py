def build_model(
    model_name: str,
    device,
    in_channels: int = 3,
    out_channels: int = 1,
    img_size=(256, 256),
):
    """
    根据 model_name 创建对应的分割模型，并放到指定 device 上。

    Args:
        model_name (str): 模型名称
        device: torch.device
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        img_size (tuple): (H, W)，给 SwinUNETR / UNETR 等用

    Returns:
        torch.nn.Module: 已经 .to(device) 的模型
    """
    print(f"Building model: {model_name}")

    if model_name == 'Unet':
        from models.two_d.unet import Unet
        model = Unet(in_channels=in_channels, out_channels=out_channels)

    elif model_name == 'DeepLabV3':
        from models.two_d.deeplab import DeepLabV3
        model = DeepLabV3(in_channels=in_channels, out_channels=out_channels)

    elif model_name == 'miniseg':
        from models.two_d.miniseg import MiniSeg
        model = MiniSeg(in_channels=in_channels, out_channels=out_channels)

    elif model_name == 'segnet':
        from models.two_d.segnet import SegNet
        model = SegNet(in_channels=in_channels, out_channels=out_channels)

    elif model_name == 'unetpp':
        from models.two_d.unetpp import ResNet34UnetPlus
        model = ResNet34UnetPlus(in_channels=in_channels, out_channels=out_channels)

    elif model_name == 'SwinUNETR':
        from models.two_d.swin_unetr import SwinUNETR
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
        )

    elif model_name == 'AttentionUnet':
        from models.two_d.attention_unet import AttentionUnet
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.0,
        )

    elif model_name == 'UNETR':
        from models.two_d.UNETR import UNETR
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=2,
            qkv_bias=False,
            save_attn=False,
        )

    elif model_name == 'MISSFormer':
        from models.two_d.MISSFormer import MISSFormer
        model = MISSFormer(num_classes=out_channels, token_mlp_mode="mix_skip")

    elif model_name == 'UCTransNet':
        from models.two_d.UCTransNet import Cfg, UCTransNet
        cfg = Cfg()
        model = UCTransNet(
            cfg,
            n_channels=in_channels,
            n_classes=out_channels,
            img_size=img_size[0],  # 你现在是 256
            vis=False,
        )

    elif model_name == 'TransFuse':
        from models.two_d.TransFuse import TransFuse_S  # 你现在用的是 S 版
        model = TransFuse_S(num_classes=out_channels, pretrained=False)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # 统一放到 device 上
    model = model.to(device)
    return model
