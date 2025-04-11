import torch
import torch.nn as nn
from thop import profile
from basicseg.main_blocks import CSI, DPCF
from basicseg.mona_with_select import MonaOp
from basicseg.networks.sam2.build_sam import build_sam2
from thop import clever_format  # 用于格式化输出的 MACs 和参数数量
from basicseg.utils.registry import NET_REGISTRY
import time


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.monaOp = MonaOp(dim)

    def forward(self, x):
        x =  self.monaOp(x)
        net = self.block(x)
        return net

    

@NET_REGISTRY.register()
class SAMamba(nn.Module):
    def __init__(self, checkpoint_path='/media/data2/zhengshuchen/code/SAMamba/sam2_configs/sam2_hiera_small.pt') -> None:
        super(SAMamba, self).__init__()
        model_cfg = "sam2_hiera_s.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.mbhf1 = CSI(96, 128)
        self.mbhf2 = CSI(192, 128)
        self.mbhf3 = CSI(384, 128)
        self.mbhf4 = CSI(768, 128)

        self.up1 = (DPCF(128, 128))
        self.up2 = (DPCF(128, 128))
        self.up3 = (DPCF(128, 128))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        #
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.mbhf1(x1), self.mbhf2(x2), self.mbhf3(x3), self.mbhf4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.head(self.deconv2(self.deconv1(x)))
        return out

if __name__ == '__main__':
    # 定义输入张量
    input_tensor = torch.randn(1, 3, 1024, 1024)  # 假设输入为 (batch_size=1, 3通道, 512x512 图像)

    # 实例化模型 (确保你的 `SAMamba` 模型类定义正确)
    net = SAMamba()
    # print(net.encoder)

    # 检查当前设备并将模型移动到相应设备
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    input_tensor = input_tensor.to(device)

    # 使用 thop 计算 MACs 和参数量
    flops, params = profile(net, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.2f")

    # 打印计算成本和参数量
    print(f"Computational cost (MACs): {flops}")
    print(f"Number of parameters: {params}")

    # # 测试 100 张图片的推理时间
    # total_time = 0
    # num_images = 100
    #
    # # 确保模型处于评估模式
    # net.eval()
    #
    # with torch.no_grad():  # 禁用梯度计算以提高推理速度
    #     for _ in range(num_images):
    #         torch.cuda.synchronize()  # 同步 GPU 和 CPU，确保时间精确
    #         start = time.time()
    #         result = net(input_tensor)
    #         torch.cuda.synchronize()
    #         end = time.time()
    #
    #         infer_time = end - start
    #         total_time += infer_time
    #
    #         # print(f'Single inference time: {infer_time:.6f} seconds')
    #
    # # 计算平均推理时间和 FPS
    # average_time = total_time / num_images
    # fps = 1 / average_time if average_time > 0 else float('inf')
    #
    # print(f'Average inference time for 100 images: {average_time:.6f} seconds')
    # print(f'FPS: {fps:.2f}')