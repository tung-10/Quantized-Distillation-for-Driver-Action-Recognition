import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# VideoMAE building blocks (khớp 100% với checkpoint gốc)
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """Tubelet embedding: Conv3D — khớp với checkpoint patch_embed.proj.*"""
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 in_chans=3, embed_dim=1024):
        super().__init__()
        self.patch_size   = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)               # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class Attention(nn.Module):
    """BEiTv2-style attention: q_bias + v_bias riêng (khớp checkpoint)"""
    def __init__(self, dim, num_heads=16, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias    = nn.Parameter(torch.zeros(dim))
        self.v_bias    = nn.Parameter(torch.zeros(dim))
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Ghép bias: [q_bias | 0 | v_bias]
        qkv_bias = torch.cat([self.q_bias,
                               torch.zeros_like(self.v_bias),
                               self.v_bias])
        qkv = F.linear(x, self.qkv.weight, qkv_bias)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads=num_heads,
                               attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# VideoMAELogit — wrapper chính cho framework Drive&Act
# ──────────────────────────────────────────────────────────────────────────────

class VideoMAELogit(nn.Module):
    """
    VideoMAE-Large wrapper cho framework Drive&Act.
    Keys khớp 100% với checkpoint gốc (không cần prefix remapping).
    Interface giống I3DLogit:
      - __init__(config_file, architecture_config)
      - forward(inp) → logits (B, num_classes), không qua softmax
    Input: (B, C, T, H, W)
    """

    # Preset cho ViT-Base và ViT-Large
    _PRESETS = {
        "vit_base":  dict(embed_dim=768,  depth=12, num_heads=12),
        "vit_large": dict(embed_dim=1024, depth=24, num_heads=16),
        "vit_huge":  dict(embed_dim=1280, depth=32, num_heads=16),
    }

    def __init__(self, config_file, architecture_config):
        super().__init__()

        num_classes  = config_file["train"]["num_classes"]
        dropout_prob = architecture_config.get("dropout_prob", 0.0)
        preset_name  = architecture_config.get("name", "vit_large")

        preset = self._PRESETS.get(preset_name, self._PRESETS["vit_large"])
        embed_dim  = preset["embed_dim"]
        depth      = preset["depth"]
        num_heads  = preset["num_heads"]

        # ── Backbone (keys khớp checkpoint) ───────────────────────────
        self.patch_embed = PatchEmbed3D(
            img_size=224, patch_size=16, tubelet_size=2,
            in_chans=3, embed_dim=embed_dim,
        )
        self.blocks  = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=4.)
            for _ in range(depth)
        ])
        self.fc_norm = nn.LayerNorm(embed_dim)   # key: fc_norm.weight/bias

        # ── Classifier head ───────────────────────────────────────────
        self.dropout = nn.Dropout(dropout_prob)
        self.head    = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, inp):
        """inp: (B, C, T, H, W) → logits: (B, num_classes)"""
        x = self.patch_embed(inp)          # (B, N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.fc_norm(x.mean(dim=1))   # mean pooling → (B, D)
        x = self.dropout(x)
        return self.head(x)                # (B, num_classes)

    def load_state_dict(self, state_dict, strict=True):
        # Bỏ key không thuộc model (logit_scale từ contrastive pretraining)
        filtered = {k: v for k, v in state_dict.items()
                    if k not in ('logit_scale',)}

        missing, unexpected = super().load_state_dict(filtered, strict=False)

        # Chỉ báo lỗi thực sự (bỏ qua cls_token, pos_embed nếu không có)
        ignorable = {'cls_token', 'pos_embed'}
        real_missing = [k for k in missing if k not in ignorable]
        if real_missing:
            print(f"[VideoMAELogit]  Missing: {real_missing[:5]}")
        if unexpected:
            print(f"[VideoMAELogit]  Unexpected: {unexpected[:5]}")

        n_blocks = sum(1 for k in filtered if k.startswith('blocks.') and k.endswith('.norm1.weight'))
        print(f"[VideoMAELogit] Loaded — {n_blocks} transformer blocks, "
              f"patch_embed 3D")

class _HeadProxy:
    """
    Giả lập interface list[-1] để driver.py có thể gọi:
        architecture.classifier[-1].in_features
        architecture.classifier[-1] = nn.Linear(...)
    mà thực chất đọc/ghi vào self._model.head.
    """
    def __init__(self, model):
        self._model = model
    def __getitem__(self, idx):
        return self._model.head
    def __setitem__(self, idx, value):
        self._model.head = value
# ──────────────────────────────────────────────────────────────────────────────
# VideoMAETeacher — wrapper teacher cho student-teacher distillation
# ──────────────────────────────────────────────────────────────────────────────
class VideoMAETeacher(nn.Module):
    """
    VideoMAE Teacher wrapper cho student-teacher distillation.
    Interface: __init__(config_file, architecture_config) / forward(inp) -> logits
    """
    def __init__(self, config_file, architecture_config):
        super().__init__()
        self.model = VideoMAELogit(config_file, architecture_config)
    @property
    def classifier(self):
        """Proxy vào self.model.head, giả lập subscript list[-1]."""
        return _HeadProxy(self.model)
    def forward(self, inp):
        """inp: (B, C, T, H, W) -> logits: (B, num_classes)"""
        return self.model(inp)