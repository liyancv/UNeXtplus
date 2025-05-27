import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.utils.checkpoint as checkpoint
from swin_transformer import SwinTransformerBlock


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,), stride=(stride,), bias=False)



def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class shiftMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.DWConv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(_init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.DWConv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.DWConv = nn.Conv2d(dim, dim, (3, 3), 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.DWConv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvDropoutNormNonLin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonLin=nn.GELU, nonLin_kwargs=None):
        """
        conv_op 卷积操作类
        norm_op 标准化操作类
        conv_kwargs=None 表示 conv_op 中的参数是可选的
        dropout_op dropout 操作类
        nonLin 非线性激活函数类
        """
        super(ConvDropoutNormNonLin, self).__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)

        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs['p'] > 0:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)

        self.lrelu = nonLin(**nonLin_kwargs) if nonLin_kwargs is not None else nonLin()

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
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
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TransConv(nn.Module):
    """
    这是一个基本的神经网络模块,它实现了一个具有深度监督的Swin Transformer模型,
    用于图像分类任务。它由多个层组成,包括卷积层、批量标准化层、dropout层、非线性层和Swin Transformer块层。
    该模块的输出可以用于下游任务，如目标检测和语义分割等。此外，如果需要，该模块还可以实现对中间层输出的深度监督
    """

    def __init__(self, input_features,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 kernel_size=(3, 3),
                 window_size=7,
                 conv_op=nn.Conv2d, norm_op=None, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonLin=None, nonLin_kwargs=None,
                 basic_block=ConvDropoutNormNonLin,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 is_encoder=True,
                 use_checkpoint=False  # use_checkpoint 是一个布尔变量，用于控制是否使用 PyTorch 的模型检查点机制，以减少显存的占用
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.is_encoder = is_encoder

        assert depth > 0, '深度必须大于0'
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': kernel_size, 'padding': 1}

        # 纯卷积块定义
        self.conv_blocks1 = basic_block(
            input_features, input_features, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonLin, nonLin_kwargs
        )
        self.conv_blocks2 = basic_block(
            input_features, dim, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonLin, nonLin_kwargs
        )

        # build blocks

        self.swin_blocks = SwinTransformerBlock(
            dim=input_features, input_resolution=input_resolution, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, drop=drop,
            shift_size=0 if (depth % 2 == 0) else window_size // 2,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, norm_layer=norm_layer,
            drop_path=drop_path[depth] if isinstance(drop_path, list) else drop_path
        )
        self.input_features = input_features  # 新增这行
        self.dim = dim  # 新增这行
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 在构造函数中定义 device

    def forward(self, x):
        s = x
        # print(x.shape)  # torch.Size([16, 16, 112, 112])
        x = self.conv_blocks1(x)
        # print(x.shape)  # torch.Size([16, 32, 112, 112])
        if self.use_checkpoint:
            s = checkpoint.checkpoint(self.swin_blocks, s)
        else:
            s = self.swin_blocks(s)
            # print(s.shape)  # torch.Size([16, 16, 112, 112])

        x = self.conv_blocks2(x + s)

        return x


class EfficientAttn(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = dim ** -0.5

        # 使用更小的 qkv 维度
        reduced_dim = dim // 2

        # 共享权重
        self.qkv = nn.Linear(dim, reduced_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(reduced_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.shift_size > 0:
            img_mask = torch.zeros((1, 112, 112, 1))  # 假设输入分辨率为 112x112
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, C, H, W = x.shape  # 假设 x 的形状为 (B, C, H, W)
        N = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)  # 转换形状为 (B, N, C)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads // 2)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_mask is not None:
            attn = attn + self.attn_mask  # 添加注意力掩码用于 SW-MSA
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C // 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # 转换回 (B, C, H, W)
        return x

# 经过优化的多头自注意力：OLSA（Optimised Long Self-Attention）
class OLSA(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, attn_drop=0., proj_drop=0.):
        super(OLSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # 多尺度注意力机制
        self.multi_scale_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop)
        
        # 瓶颈聚合
        self.bottleneck = nn.Linear(dim, dim // 2)
        self.proj = nn.Linear(dim // 2, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape  # 假设输入形状为 (B, C, H, W)
        N = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)  # 转换形状为 (B, N, C)

        # 多尺度注意力机制
        attn_output, _ = self.multi_scale_attention(x, x, x)
        
        # 瓶颈聚合
        bottleneck_output = self.bottleneck(attn_output)
        x = self.proj(bottleneck_output)
        x = self.proj_drop(x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # 转换回 (B, C, H, W)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = dim ** -0.5

        # 使用较小的qkv维度
        reduced_dim = dim // 2

        # 共享权重
        self.qkv = nn.Linear(dim, reduced_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(reduced_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.shift_size > 0:
            img_mask = torch.zeros((1, 112, 112, 1))  # assuming input resolution 112x112
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, C, H, W = x.shape  # Assuming x has shape (B, C, H, W)
        N = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)  # 转换形状为 (B, N, C)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads // 2)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_mask is not None:
            attn = attn + self.attn_mask  # add the attention mask for SW-MSA
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C // 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # 转换回 (B, C, H, W)
        return x


# 自适应特征融合模块
class AFF(nn.Module):
    def __init__(self, in_channels):
        super(AFF, self).__init__()
        self.local_att = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.global_att = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_l, x_g):
        attn_l = self.local_att(x_l)
        attn_g = self.global_att(x_g)
        attn = self.sigmoid(attn_l + attn_g)
        return x_l * attn + x_g * (1 - attn)
    
#修改后，采用深度可分离卷积的AFF函数
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AFFOptimized(nn.Module):
    def __init__(self, in_channels):
        super(AFFOptimized, self).__init__()
        self.local_att = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=1)
        self.global_att = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_l, x_g):
        attn_l = self.local_att(x_l)
        attn_g = self.global_att(x_g)
        attn = self.sigmoid(attn_l + attn_g)
        return x_l * attn + x_g * (1 - attn)


# 深度可分离卷积块
class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_op, conv_kwargs, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonLin=None):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise = conv_op(in_channels, in_channels, **conv_kwargs)
        self.pointwise = conv_op(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm_op = norm_op(out_channels, **norm_op_kwargs) if norm_op is not None else None
        self.dropout_op = dropout_op(**dropout_op_kwargs) if dropout_op is not None else None
        self.nonLin = nonLin() if nonLin is not None else None

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.norm_op is not None:
            x = self.norm_op(x)
        if self.dropout_op is not None:
            x = self.dropout_op(x)
        if self.nonLin is not None:
            x = self.nonLin(x)
        return x


class SPHTension(nn.Module):
    def __init__(self, input_features,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 kernel_size=(3, 3),
                 window_size=7,
                 conv_op=nn.Conv2d, norm_op=None, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonLin=None, nonLin_kwargs=None,
                 basic_block=ConvDropoutNormNonLin,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 is_encoder=True,
                 use_checkpoint=False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.is_encoder = is_encoder

        assert depth > 0, 'Depth must be greater than 0'
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': kernel_size, 'padding': 1}

        # 纯卷积块定义
        self.conv_blocks1 = DepthwiseSeparableConvBlock(
            input_features, input_features, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonLin
        )
        self.conv_blocks2 = basic_block(
            input_features, dim, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonLin, nonLin_kwargs
        )

        # build blocks
        self.sparse_attn_blocks = SparseAttention(dim=input_features, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.EfficientAttn_blocks = EfficientAttn(dim=input_features, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        
        self.swin_blocks = SwinTransformerBlock(
            dim=input_features, input_resolution=input_resolution, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, drop=drop,
            shift_size=0 if (depth % 2 == 0) else window_size // 2,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, norm_layer=norm_layer,
            drop_path=drop_path[depth] if isinstance(drop_path, list) else drop_path
        )
        

        # self.aff = AFF(input_features)
        self.aff = AFFOptimized(input_features)

    def forward(self, x):
        s = x
        x = self.conv_blocks1(x)
        if self.use_checkpoint:
            s = checkpoint.checkpoint(self.EfficientAttn_blocks, s)
        else:
            s = self.EfficientAttn_blocks(s)

        x = self.aff(x, s)
        x = self.conv_blocks2(x)

        return x


class ChannelReduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReduction, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)


# model unet, method = 0
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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
        print('UNet model create success!!')
        return logits
        
        

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)  # 数据归一化
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

# model unet++, method = 1
class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

        
# model transunet(number = 2) is in transunet.py
        

# model unext, method number = 3
class UNeXt(nn.Module):

    # UNext: Conv 3 + MLP 2 + shifted MLP
    # transUNext : conv + transConv 2 + MLP 2 + shifted MLP

    def __init__(self, num_classes=1, input_channels=3, img_size=224, embed_dims=None,
                 drop_rate=0.1, drop_path_rate=0.2, norm_layer=nn.LayerNorm, ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_resolution = [img_size, img_size]
        self.conv_op = nn.Conv2d

        if embed_dims is None:
            embed_dims = [128, 160, 256]

        self.encoder1 = nn.Conv2d(self.input_channels, 16, (3, 3), stride=(1, 1), padding=1)
        self.encoder2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1)
        self.encoder3 = nn.Conv2d(32, 128, (3, 3), stride=(1, 1), padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.decoderNorm3 = norm_layer(160)
        self.decoderNorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        )])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
        )])

        self.decoderBlock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        )])

        self.decoderBlock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
        )])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_channels=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_channels=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, (3, 3), stride=(1, 1), padding=1)
        self.decoder2 = nn.Conv2d(160, 128, (3, 3), stride=(1, 1), padding=1)
        self.decoder3 = nn.Conv2d(128, 32, (3, 3), stride=(1, 1), padding=1)
        self.decoder4 = nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=1)
        self.decoder5 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=(1, 1))

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        B = x.shape[0]
        # Encoder
        # Conv Stage

        # Stage 1

        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        # torch.Size([6, 16, 112, 112])
        t1 = out
        # Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        # torch.Size([6, 32, 56, 56])
        t2 = out
        # Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        # torch.Size([6, 128, 28, 28])
        t3 = out

        # Tokenized MLP Stage
        # Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # torch.Size([6, 256, 7, 7])
        # Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        # torch.Size([6, 160, 14, 14])
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.decoderBlock1):
            out = blk(out, H, W)

        # Stage 3

        out = self.decoderNorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.decoderBlock2):
            out = blk(out, H, W)

        out = self.decoderNorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)  
    
# model UNeXt++, model number = 4
class UNeXtPlus(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, img_size=224, embed_dims=None,
                 drop_rate=0.1, drop_path_rate=0.2, norm_layer=nn.LayerNorm, depths=None, method=1):
        super().__init__()

        self.method = method
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_resolution = [img_size, img_size]
        self.conv_op = nn.Conv2d
        norm_op = nn.InstanceNorm2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op = nn.Dropout2d
        dropout_op_kwargs = {'p': drop_rate, 'inplace': True}
        nonLin = nn.GELU
        nonLin_kwargs = None

        if embed_dims is None:
            embed_dims = [96, 128, 192]

        if depths is None:
            depths = [1, 1]

        # 使用深度可分离卷积
        self.encoder1_dw = nn.Conv2d(self.input_channels, self.input_channels, (3, 3), stride=(1, 1), padding=1, groups=self.input_channels)
        self.encoder1_pw = nn.Conv2d(self.input_channels, 16, (1, 1))

        self.encoder2_dw = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=1, groups=16)
        self.encoder2_pw = nn.Conv2d(16, 32, (1, 1))

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(embed_dims[0])

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.decoderNorm3 = norm_layer(128)
        self.decoderNorm4 = norm_layer(96)

        self.transConv2 = SPHTension(input_features=32, dim=96,
                                         input_resolution=[img_size // 4, img_size // 4],
                                         depth=depths[1], num_heads=4,
                                         dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                         norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                         nonLin=nonLin, nonLin_kwargs=nonLin_kwargs, is_encoder=True)

        self.transConv3 = SPHTension(input_features=96, dim=32,
                                         input_resolution=[img_size // 8, img_size // 8],
                                         depth=depths[1], num_heads=4,
                                         dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                         norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                         nonLin=nonLin, nonLin_kwargs=nonLin_kwargs, is_encoder=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        )])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
        )])

        self.decoderBlock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        )])

        self.decoderBlock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
        )])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_channels=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_channels=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(192, 128, (3, 3), stride=(1, 1), padding=1)
        self.decoder2 = nn.Conv2d(128, 96, (3, 3), stride=(1, 1), padding=1)

        self.depthwise_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise_conv3 = nn.Conv2d(32, embed_dims[0], kernel_size=1, stride=1)

        self.decoder4 = nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=1)
        self.decoder5 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=1)

        self.dbn1 = nn.BatchNorm2d(128)
        self.dbn2 = nn.BatchNorm2d(96)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=(1, 1))

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        B = x.shape[0]
        # 编码器
        # 卷积阶段

        # 第1阶段
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1_pw(self.encoder1_dw(x))), 2, 2))
        t1 = out
        # 第2阶段
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2_pw(self.encoder2_dw(out))), 2, 2))
        t2 = out
        # 第3阶段

        # out = F.relu(self.pointwise_conv3(self.depthwise_conv3(out)))
        # out = F.relu(F.max_pool2d(self.ebn3(out), 2, 2))
        # t3 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.transConv2(out)), 2, 2))

        # Tokenized MLP阶段
        # 第4阶段

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Bottleneck阶段

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 第4阶段

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.decoderBlock1):
            out = blk(out, H, W)

        # 第3阶段

        out = self.decoderNorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.decoderBlock2):
            out = blk(out, H, W)

        out = self.decoderNorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.transConv3(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
    
    