import torch
import torch.nn as nn
import torch.nn.functional as F

from basicseg.main_blocks import MBHF, ACF
from .sam2.build_sam import build_sam2
from basicseg.utils.registry import NET_REGISTRY




class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

    

@NET_REGISTRY.register()
class SAMamba(nn.Module):
    def __init__(self, checkpoint_path='/media/data2/zhengshuchen/code/SAMamba/sam2_configs/sam2_hiera_tiny.pt') -> None:
        super(SAMamba, self).__init__()
        model_cfg = "sam2_hiera_t.yaml"
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
        self.mbhf1 = MBHF(96, 64)
        self.mbhf2 = MBHF(192, 64)
        self.mbhf3 = MBHF(384, 64)
        self.mbhf4 = MBHF(768, 64)

        self.up1 = (ACF(64, 64))
        self.up2 = (ACF(64, 64))
        self.up3 = (ACF(64, 64))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        #
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.mbhf1(x1), self.mbhf2(x2), self.mbhf3(x3), self.mbhf4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.head(self.deconv2(self.deconv1(x)))
        return out