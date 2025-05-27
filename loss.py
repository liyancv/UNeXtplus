# 这段代码定义了两个损失函数类: BCEDiceLoss 和 LovaszHingeLoss，这两个损失函数通常用于二分类或多分类任务中。
# 以下是每个损失函数的简要说明：

# BCEDiceLoss:
# 说明： Binary Cross Entropy Dice Loss，结合了二分类交叉熵损失和 Dice 损失。
# 实现： 在前向传播中，首先计算二分类交叉熵损失 (bce)，然后计算 Dice 损失。最终的损失是这两者的线性组合。
# 参数： 无需额外参数。

# LovaszHingeLoss:
# 说明： Lovasz Hinge Loss，用于处理类别不平衡和边界框预测任务。
# 实现： 在前向传播中，首先对输入和目标进行适当的处理（squeeze 操作），然后使用 lovasz_hinge 函数计算 Lovasz Hinge 损失。
# 依赖： 该损失函数使用了 LovaszSoftmax 包中的 lovasz_hinge 函数，因此在导入该模块时需要确保正确导入
# 这段代码中两个损失函数的设计是为了在训练中同时考虑分类和空间位置的信息，以便更好地适应不同任务和数据。
import torch
import torch.nn as nn
import torch.nn.functional as F
from lovasz_softmax import lovasz_softmax

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
