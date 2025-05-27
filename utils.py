# 本文件定义了一些实用工具和辅助函数，这些函数可以在深度学习模型的训练和评估过程中使用
import argparse
import torch.nn as nn
import torch
import time



def maybe_to_torch(d):
    """Converting data to Tensor type.

    Args:
        d (_type_): _description_

    Returns:
        _type_: Tensor
    """
    if isinstance(d, list):
        # HACK: 和nnUNet不一样
        d = [i if isinstance(i, torch.Tensor) else maybe_to_torch(i) for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    """Putting data on GPU.

    Args:
        data (_type_): _description_
        non_blocking (bool, optional): Controls whether data operations are asynchronous. Defaults to True.
        gpu_id (int, optional): gpu number. Defaults to 0.

    Returns:
        _type_: cuda type
    """
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

# 将字符串转换为布尔值的辅助函数。
# 用于解析命令行参数时，将字符串 'true' 转换为 True，将字符串 'false' 转换为 False。
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 计算模型的可训练参数数量
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 用于计算和存储平均值和当前值的类
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
