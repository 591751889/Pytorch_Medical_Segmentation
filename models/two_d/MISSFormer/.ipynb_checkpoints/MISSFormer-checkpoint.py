import torch
import torch.nn as nn
from .segformer import *
from typing import Tuple
from einops import rearrange


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x,
            'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class SegU_decoder(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, n_class=9, norm_layer=nn.LayerNorm,
                 is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class BridgeLayer_4(nn.Module):
    """
    4 个尺度的桥接层。注意这里把更高层通道数“摊平”到序列维度，
    因而切片长度是 [1,2,5,8] * [64^2, 32^2, 16^2, 8^2] = [4096, 2048, 1280, 512]
    对应重排空间尺寸分别为 64/32/16/8。
    """
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims, dims * 4)
        self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape  # C 将被定为 64（stage1 的通道）
            # 这里的 reshape(B, -1, C) 会把更高层的通道数折叠到长度维度上
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # len = 64*64 = 4096
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # len = (32*32*128)/64 = 2048
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # len = (16*16*320)/64 = 1280
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # len = (8*8*512)/64  = 512

            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)   # 总长度 4096+2048+1280+512 = 7936
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        # 按长度切片回到 4 个尺度分段
        tem1 = tx[:, :4096, :].reshape(B, -1, C)           # 64x64,   C
        tem2 = tx[:, 4096:6144, :].reshape(B, -1, C * 2)   # 32x32, 2C（长度 2048）
        tem3 = tx[:, 6144:7424, :].reshape(B, -1, C * 5)   # 16x16, 5C（长度 1280）
        tem4 = tx[:, 7424:7936, :].reshape(B, -1, C * 8)   # 8x8,   8C（长度 512）

        m1f = self.mixffn1(tem1, 64, 64).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 32, 32).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 16, 16).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 8, 8).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        tx2 = tx1 + t1

        return tx2


class BridgeLayer_3(nn.Module):
    """
    3 个尺度的桥接层（不包含最浅层 64x64）。
    切片长度：2048(32x32, 2C) / 1280(16x16, 5C) / 512(8x8, 8C)
    """
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn2 = MixFFN(dims * 2, dims * 8)
        self.mixffn3 = MixFFN(dims * 5, dims * 20)
        self.mixffn4 = MixFFN(dims * 8, dims * 32)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            # 仅拼接后 3 个尺度
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # len = 2048
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # len = 1280
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # len = 512
            inputs = torch.cat([c2f, c3f, c4f], -2)         # 总长度 3840
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem2 = tx[:, :2048, :].reshape(B, -1, C * 2)       # 32x32, 2C
        tem3 = tx[:, 2048:3328, :].reshape(B, -1, C * 5)   # 16x16, 5C（2048+1280=3328）
        tem4 = tx[:, 3328:3840, :].reshape(B, -1, C * 8)   # 8x8,   8C（+512=3840）

        m2f = self.mixffn2(tem2, 32, 32).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 16, 16).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 8, 8).reshape(B, -1, C)

        t1 = torch.cat([m2f, m3f, m4f], -2)
        tx2 = tx1 + t1

        return tx2


class BridegeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape
        outs = []

        sk1 = bridge4[:, :4096, :].reshape(B, 64, 64, C).permute(0, 3, 1, 2)
        sk2 = bridge4[:, 4096:6144, :].reshape(B, 32, 32, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, 6144:7424, :].reshape(B, 16, 16, C * 5).permute(0, 3, 1, 2)
        sk4 = bridge4[:, 7424:7936, :].reshape(B, 8, 8, C * 8).permute(0, 3, 1, 2)

        outs.extend([sk1, sk2, sk3, sk4])
        return outs


class BridegeBlock_3(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_3(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        if (type(x) == list):
            outs.append(x[0])  # 保留最浅层特征（若需要）
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape

        sk2 = bridge4[:, :2048, :].reshape(B, 32, 32, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, 2048:3328, :].reshape(B, 16, 16, C * 5).permute(0, 3, 1, 2)
        sk4 = bridge4[:, 3328:3840, :].reshape(B, 8, 8, C * 8).permute(0, 3, 1, 2)

        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class MISSFormer(nn.Module):
    def __init__(self, num_classes=9, token_mlp_mode="mix_skip", encoder_pretrained=True):
        super().__init__()

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        d_base_feat_size = 8  # 256 输入下，最终步长为 32，256/32=8
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]

        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        # 关键：把 224 改成 256
        self.backbone = MiT(256, dims, layers, token_mlp_mode)

        self.reduction_ratios = [1, 2, 4, 8]
        self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], heads[3],
                                        reduction_ratios[3], token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2], heads[2],
                                        reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], heads[1],
                                        reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], heads[0],
                                        reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder = self.backbone(x)
        bridge = self.bridge(encoder)  # list

        b, c, _, _ = bridge[3].shape
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(bridge[3].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0, 2, 3, 1))

        return tmp_0


if __name__ == '__main__':
    # === 构建并测试 MISSFormer（当前实现：256×256） ===
    model = MISSFormer(num_classes=9, token_mlp_mode="mix_skip")
    x = torch.randn(4, 3, 256, 256)  # 注意：256×256
    y = model(x)

    print("输入:", x.shape)
    print("输出:", y.shape)  # 期望是 (4, 9, 256, 256)
