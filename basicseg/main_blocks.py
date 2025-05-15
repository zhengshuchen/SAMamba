import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x



class CSI(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type='bn',
                                activation=True)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.drop = nn.Dropout2d(0.3)
         self.mambacr = MAMBACR(filters, filters)
         self.final_conv = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(1, 1),
                                padding=(0, 0))

    def forward(self, x):
        x_skip = self.skip(x)
        x = self.mambacr(x_skip)
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.final_conv(x_skip + x)
        return x



class DPCF(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.ac = AdaptiveCombiner()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=in_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0))
         )
    def forward(self, x_low, x_high):
        image_size = x_high.size(2)
        if x_high != None:
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:
            x_low = F.interpolate(x_low, size=[image_size, image_size], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x0 = self.ac(x_low[0], x_high[0])
        x1 = self.ac(x_low[1], x_high[1])
        x2 = self.ac(x_low[2], x_high[2])
        x3 = self.ac(x_low[3], x_high[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)

        return x


class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super(AdaptiveCombiner, self).__init__()
        # 定义可学习参数d，形状与p和i相同，这里假设p和i的形状为(batch_size, channel, w, h)
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, p, i):
        batch_size, channel, w, h = p.shape
        # 将self.d扩展为与p和i相同的形状
        d = self.d.expand(batch_size, channel, w, h)
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups = 128
            )
        self.block2 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=128
            )
        self.block3 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=128
            )
        self.block4 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=128
            )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )

    def forward(self, x):
        # 创建一个空列表，用于存储拆分后的张量
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        # 按通道逐一拆分
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x


class MAMBACR(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm1 = nn.LayerNorm(input_dim//4)
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim//4, output_dim//4)
        self.skip_scale= nn.Parameter(torch.ones(1))
        self.cpe2 = nn.Conv2d(input_dim//4, input_dim//4, 3, padding=1, groups=input_dim//4)
        # self.out_s = conv_block(
        #     in_features=4,
        #     out_features=4,
        #     kernel_size=(1, 1),
        #     padding=(0, 0),
        # )
        self.out = conv_block(
            in_features=output_dim,
            out_features=output_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.mlp = Mlp(in_features=input_dim//4, hidden_features=int(input_dim//4 * 4))
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mlp(self.norm1(self.mamba(x1))) + self.skip_scale * x1
        x_mamba2 = self.mlp(self.norm1(self.mamba(x2))) + self.skip_scale * x2
        x_mamba3 = self.mlp(self.norm1(self.mamba(x3))) + self.skip_scale * x3
        x_mamba4 = self.mlp(self.norm1(self.mamba(x4))) + self.skip_scale * x4

        x_mamba1 = x_mamba1.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)
        x_mamba2 = x_mamba2.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)
        x_mamba3 = x_mamba3.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)
        x_mamba4 = x_mamba4.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)

        # 按通道逐一拆分
        # 创建一个空列表，用于存储拆分后的张量
        split_tensors = []
        for channel in range(x_mamba1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x_mamba1, x_mamba2, x_mamba3, x_mamba4]]
            concatenated_channel = torch.cat(channel_tensors, dim=1) # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        out = self.out(x)

        return out