import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class Binary_Loss(nn.Module):
    def __init__(self):
        super(Binary_Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        loss = self.criterion(model_output, targets)
   
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1  # 调整为更小的平滑系数

    def forward(self, inputs, targets):
        num = targets.size(0)
        probs = torch.sigmoid(inputs)
        m1 = probs.view(num, -1)  # 预测概率展平 [B, N]
        m2 = targets.view(num, -1)  # 标签展平 [B, N]
        
        intersection = (m1 * m2).sum(1)  # 每个样本的交集 [B]
        dice_score = (2. * intersection + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)  # 修正分子
        
        loss = 1 - dice_score.mean()  # 平均所有样本的Dice Loss
        return loss
    
    
class CrossEntropyLossWrapper(nn.Module):
    """
    计算 Cross-Entropy Loss (CELoss) 的类

    参数:
    logits: torch.Tensor, 模型的输出，shape 为 [N, C, H, W]
    labels: torch.Tensor, 真实的类别标签，shape 为 [N, H, W]，每个元素为浮点数
    """
    def __init__(self):
        super(CrossEntropyLossWrapper, self).__init__()

    def forward(self, logits, labels):
        # 检查 logits 是否有 2 个通道
        if logits.size(1) != 2:
            raise ValueError(f"logits 通道数应为 2，当前为 {logits.size(1)}")

        # 检查输入的尺寸是否符合要求
        if logits.shape[0] != labels.shape[0] or logits.shape[2:] != labels.shape[1:]:
            print("Logits 形状:", logits.shape, "Labels 形状:", labels.shape)
            raise ValueError("logits 和 labels 的形状不匹配")

        # 将 labels 从 [N, 1, H, W] 转为 [N, H, W]
        labels = labels.squeeze(1)

        # 将 labels 二值化处理，以确保标签为二分类索引（0 或 1）
        labels_binary = (labels > 0.5).long()

        # 计算 Cross-Entropy Loss
        loss = F.cross_entropy(logits, labels_binary)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean', eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # 1. 原始 BCE loss
        bce_loss = self.bce(inputs, targets.float())

        # 2. 计算 pt，并做数值稳定处理
        pt = torch.sigmoid(inputs)
        pt = pt.clamp(self.eps, 1.0 - self.eps)

        # 3. 计算 focal weight
        pos_weight = self.alpha * (1 - pt) ** self.gamma
        neg_weight = (1 - self.alpha) * pt ** self.gamma
        focal_weight = pos_weight * targets + neg_weight * (1 - targets)

        # 4. 叠加权重
        focal_loss = focal_weight * bce_loss

        # 5. reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

#
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha  # 平衡正负样本的权重
#         self.gamma = gamma  # 调节难易样本的权重
#         self.reduction = reduction
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')
#
#     def forward(self, inputs, targets):
#         bce_loss = self.bce(inputs, targets)
#         pt = torch.sigmoid(inputs)
#         focal_weight = self.alpha * (1 - pt) ** self.gamma * targets + (1 - self.alpha) * pt ** self.gamma * (1 - targets)
#         focal_loss = focal_weight * bce_loss
#         return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
#         

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.6 ,focal_weight=0.4):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, inputs, targets):
        # 无需压缩 targets 的维度，保持四维传递
        dice_loss = self.dice_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return total_loss
class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.6, BCE_weight=0.4):
        super(BCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = BCE_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = Binary_Loss()

    def forward(self, inputs, targets):
        # 无需压缩 targets 的维度，保持四维传递
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total_loss

class BCEDiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.3, BCE_weight=0.3,Focal_weight=0.4):
        super(BCEDiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = BCE_weight
        self.focal_weight = Focal_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = Binary_Loss()
        self.focal_loss = FocalLoss()
    def forward(self, inputs, targets):
        # 无需压缩 targets 的维度，保持四维传递
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        focal_loss=self.focal_loss(inputs, targets)
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss+self.focal_weight * focal_loss
        return total_loss