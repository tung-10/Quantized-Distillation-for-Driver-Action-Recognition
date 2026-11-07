import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet mean/std (RGB order) dung de re-normalize input tu pipeline [-1,1] BGR
# Shape (1, C, 1, 1, 1) de broadcast voi (B, C, T, H, W)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# VideoMAE building blocks (khop 100% voi checkpoint goc)
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """Tubelet embedding: Conv3D — khop voi checkpoint patch_embed.proj.*"""
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
    """BEiTv2-style attention: q_bias + v_bias rieng (khop checkpoint)"""
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
        # Ghep bias: [q_bias | 0 | v_bias]
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
# VideoMAELogit — wrapper chinh cho framework Drive&Act
# ──────────────────────────────────────────────────────────────────────────────

class VideoMAELogit(nn.Module):
    """
    VideoMAE-Large wrapper cho framework Drive&Act.
    Keys khop 100% voi checkpoint goc (khong can prefix remapping).
    Interface giong I3DLogit:
      - __init__(config_file, architecture_config)
      - forward(inp) -> logits (B, num_classes), khong qua softmax
    Input: (B, C, T, H, W) — nhan input BGR [-1,1] tu pipeline,
           tu dong flip sang RGB va apply ImageNet mean/std ben trong.
    """

    # Preset cho ViT-Base va ViT-Large
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

        # ── Backbone (keys khop checkpoint) ───────────────────────────
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
        """
        inp: (B, C, T, H, W) — BGR [-1,1] tu pipeline cv2.
        Pipeline:
          1. BGR [-1,1] -> BGR [0,1]  : (inp + 1) / 2
          2. BGR [0,1]  -> RGB [0,1]  : flip channel dim
          3. RGB [0,1]  -> ImageNet norm: (x - mean) / std
        """
        # 1. Dua ve [0, 1]
        x = (inp + 1.0) * 0.5                    # [-1,1] -> [0,1]

        # 2. BGR -> RGB: flip channel axis (dim=1)
        x = x.flip(1)                             # (B, C, T, H, W), C: B G R -> R G B

        # 3. ImageNet mean/std normalize (RGB)
        mean = _IMAGENET_MEAN.to(x.device, x.dtype)
        std  = _IMAGENET_STD.to(x.device, x.dtype)
        x = (x - mean) / std

        # 4. ViT forward
        x = self.patch_embed(x)                  # (B, N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.fc_norm(x.mean(dim=1))          # mean pooling -> (B, D)
        x = self.dropout(x)
        return self.head(x)                       # (B, num_classes)

    def load_state_dict(self, state_dict, strict=True):
        # Bo key khong thuoc model (logit_scale tu contrastive pretraining)
        filtered = {k: v for k, v in state_dict.items()
                    if k not in ('logit_scale',)}

        missing, unexpected = super().load_state_dict(filtered, strict=False)

        # Chi bao loi thuc su (bo qua cls_token, pos_embed neu khong co)
        ignorable = {'cls_token', 'pos_embed'}
        real_missing = [k for k in missing if k not in ignorable]
        if real_missing:
            print(f"[VideoMAELogit]  Missing: {real_missing[:5]}")
        if unexpected:
            print(f"[VideoMAELogit]  Unexpected: {unexpected[:5]}")

        n_blocks = sum(1 for k in filtered if k.startswith('blocks.') and k.endswith('.norm1.weight'))
        print(f"[VideoMAELogit] Loaded — {n_blocks} transformer blocks, "
              f"patch_embed 3D, BGR-flip+ImageNet-norm enabled")


class _HeadProxy:
    """
    Gia lap interface list[-1] de driver.py co the goi:
        architecture.classifier[-1].in_features
        architecture.classifier[-1] = nn.Linear(...)
    ma thuc chat doc/ghi vao self._model.head.
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
        """Proxy vao self.model.head, gia lap subscript list[-1]."""
        return _HeadProxy(self.model)

    def forward(self, inp):
        """inp: (B, C, T, H, W) -> logits: (B, num_classes)"""
        return self.model(inp)
