
import torch
import torch.nn as nn
import time
from thop import profile
from thop import clever_format  # 用于格式化输出的 MACs 和参数数量
import torch.nn.functional as F
from basicseg.utils.registry import NET_REGISTRY

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

@NET_REGISTRY.register()
class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = in_c
        self.n_classes = in_c
        self.bilinear = bilinear

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_c)

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
        return logits

def main():
    x = torch.rand(2,3,512,512)
    net = UNet()
    import ptflops
    params,macs = ptflops.get_model_complexity_info(net,(3,512,512))
    print(params,macs)
    # y = net(x)
    # print(y.shape)

if __name__ == '__main__':
    # 定义输入张量
    input_tensor = torch.randn(1, 3, 1024, 1024)  # 假设输入为 (batch_size=1, 3通道, 512x512 图像)

    # 实例化模型 (确保你的 `SAMamba` 模型类定义正确)
    net = UNet()
    # print(net.encoder)

    # 检查当前设备并将模型移动到相应设备
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    input_tensor = input_tensor.to(device)

    # 使用 thop 计算 MACs 和参数量
    # flops, params = profile(net, inputs=(input_tensor,))
    # flops, params = clever_format([flops, params], "%.2f")
    #
    # # 打印计算成本和参数量
    # print(f"Computational cost (MACs): {flops}")
    # print(f"Number of parameters: {params}")

    # # 测试 100 张图片的推理时间
    total_time = 0
    num_images = 100

    # 确保模型处于评估模式
    net.eval()

    with torch.no_grad():  # 禁用梯度计算以提高推理速度
        for _ in range(num_images):
            torch.cuda.synchronize()  # 同步 GPU 和 CPU，确保时间精确
            start = time.time()
            result = net(input_tensor)
            torch.cuda.synchronize()
            end = time.time()

            infer_time = end - start
            total_time += infer_time

            # print(f'Single inference time: {infer_time:.6f} seconds')

    # 计算平均推理时间和 FPS
    average_time = total_time / num_images
    fps = 1 / average_time if average_time > 0 else float('inf')

    print(f'Average inference time for 100 images: {average_time:.6f} seconds')
    print(f'FPS: {fps:.2f}')