"""
architecture/videomae_teacher.py

Wrapper để load VideoMAE ViT-Large fine-tuned làm teacher trong KD pipeline.
Weights được load bởi driver.py, không load trong __init__.

Yêu cầu:
    pip install timm einops
    git clone https://github.com/MCG-NJU/VideoMAE.git videomae_repo
"""

import torch
import torch.nn as nn
import sys
import os

VIDEOMAE_REPO = "/mnt/data/quangtungbk/Continuous-Action-Recognition-v2/models"
if VIDEOMAE_REPO not in sys.path:
    sys.path.insert(0, VIDEOMAE_REPO)


class VideoMAETeacher(nn.Module):
    """
    VideoMAE ViT-Large teacher wrapper.

    Input : [B, C, T, H, W]  — cùng format với MobileNet student
    Output: logits [B, num_classes]

    Weights được load bởi driver.py sau khi khởi tạo.
    """

    def __init__(self, config_file, architecture_config):
        super().__init__()

        teacher_cfg = config_file["teacher"]
        num_classes  = teacher_cfg["num_classes"]

        try:
            import modeling_finetune
        except ImportError:
            raise ImportError(
                "Không tìm thấy VideoMAE repo. Chạy:\n"
                "  git clone https://github.com/MCG-NJU/VideoMAE.git videomae_repo\n"
                "trong thư mục gốc project."
            )

        self.model = modeling_finetune.vit_large_patch16_224(
            pretrained=False,
            num_classes=num_classes,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            head_drop_rate=0.0,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        # weights chưa load ở đây — driver.py sẽ load sau

    def forward(self, x):
        """x: [B, C, T, H, W]"""
        return self.model(x)

    def train(self, mode=True):
        """Teacher luôn ở eval mode."""
        super().train(False)
        return self