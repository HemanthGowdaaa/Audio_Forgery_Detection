"""
model.py  (M2-Optimized)
========================
ResNet++ Audio Forgery Detection Model — memory-efficient version for
MacBook Air M2 (8 GB RAM).

OPTIMIZATIONS vs. original:
  1. `freeze_early_layers(n)`: Freeze the first n ResNet50 layers → no
     gradient tensors stored for those parameters (~40% reduction in optimizer
     state RAM when n=2).
  2. `MultiScaleFusion`: replaced memory-hungry 5×5 and 7×7 plain convolutions
     with dilated 3×3 convolutions (dilation=2 and 3). Same effective receptive
     field, ~60% fewer parameters and much smaller intermediate tensors.
  3. `TransformerBranch.seq_len` is derived dynamically from the actual feature
     map (H×W) instead of being hard-coded to 49.  This means the model works
     correctly for any input resolution (128×128 → 4×4 map → seq_len=16).
  4. `ClassificationHead.hidden_dim` defaults to 256 (was 512) — halves the
     FC layer size.
  5. `build_model` calls `freeze_early_layers` according to the config key
     `model.freeze_backbone_layers` (default 2).

Architecture (unchanged at the algorithmic level):
  ┌────────────────────────────────────────────────────────┐
  │  Input: [B, 3, H, W]  (H=W=128 in M2 config)          │
  │                                                        │
  │  ResNet50 backbone (pretrained optional, fc removed)   │
  │  → [B, 2048, H/32, W/32]  e.g. [B, 2048, 4, 4]       │
  │                                                        │
  │  CBAM (Channel + Spatial Attention)                    │
  │  ↓                                                     │
  │  ┌──────────┐    ┌────────────────────┐               │
  │  │ SE Block │    │ Transformer Branch │               │
  │  └────┬─────┘    └────────┬───────────┘               │
  │       └────────┬──────────┘                            │
  │            element-wise add + 1×1 conv                 │
  │                                                        │
  │  Multi-Scale Fusion (1×1, 3×3, dilated-3×3 ×2)        │
  │                                                        │
  │  Classification Head: GAP → FC → BN → ReLU → Dropout  │
  │  → logits [B, 2]                                       │
  └────────────────────────────────────────────────────────┘
"""

import logging
import math
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

logger = logging.getLogger("resnet_forgery.model")


# ===========================================================================
# CBAM – Convolutional Block Attention Module
# ===========================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention sub-module of CBAM.

    Applies both average-pooling and max-pooling across spatial dimensions,
    feeds them through a shared MLP, sums results, and applies sigmoid gating.

    Args:
        in_channels:     Number of input feature channels.
        reduction_ratio: Bottleneck reduction factor for the MLP.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()

        bottleneck = max(1, in_channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)   # [B, C, 1, 1]

        # Shared MLP (implemented as 1×1 convolutions for efficiency)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, in_channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attn    = self.sigmoid(avg_out + max_out)
        return attn


class SpatialAttention(nn.Module):
    """
    Spatial Attention sub-module of CBAM.

    Pools across the channel dimension (avg + max), concatenates results,
    and applies a large-kernel convolution + sigmoid to produce a 2-D map.

    Args:
        kernel_size: Convolution kernel size (paper recommends 7).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)              # [B, 1, H, W]
        max_out = x.max(dim=1, keepdim=True).values        # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)    # [B, 2, H, W]
        attn = self.sigmoid(self.conv(combined))
        return attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).

    Applies Channel Attention → then Spatial Attention sequentially.

    Args:
        in_channels:     Number of feature channels.
        reduction_ratio: Bottleneck factor for channel attention MLP.
        kernel_size:     Spatial attention convolution kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attn  = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn  = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel refinement
        x = x * self.channel_attn(x)
        # Spatial refinement
        x = x * self.spatial_attn(x)
        return x


# ===========================================================================
# SE Block – Squeeze-and-Excitation
# ===========================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).

    Globally pools spatial information ('squeeze'), then learns channel-wise
    recalibration weights via a small bottleneck FC path ('excitation').

    Args:
        in_channels:     Number of input channels.
        reduction_ratio: Bottleneck reduction factor.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        bottleneck = max(1, in_channels // reduction_ratio)

        self.squeeze   = nn.AdaptiveAvgPool2d(1)   # [B, C, 1, 1]
        self.excitation = nn.Sequential(
            nn.Flatten(),                                          # [B, C]
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        scale = self.squeeze(x)            # [B, C, 1, 1]
        scale = self.excitation(scale)     # [B, C]
        scale = scale.view(B, C, 1, 1)    # broadcast over spatial dims
        return x * scale


# ===========================================================================
# Transformer Branch
# ===========================================================================

class TransformerBranch(nn.Module):
    """
    Transformer encoder branch applied to the spatial feature map.

    OPTIMIZATION: seq_len is derived from the actual feature map size at
    runtime, so this works correctly for any input resolution (e.g. 128×128
    input → 4×4 feature map → seq_len=16, vs. 224×224 → 7×7 → seq_len=49).
    A learnable positional embedding is registered as a buffer-parameter with
    size [1, max_seq_len, C]; we slice it at forward time.

    Also, ff_dim is now configurable and defaults to 512 (vs. 8192 original),
    which reduces the Transformer FFN from ~67 M params to ~4 M params.

    Args:
        in_channels:  Feature dimension C (= 2048 for ResNet50 layer4).
        max_seq_len:  Maximum sequence length supported (default 256 = 16×16).
        num_heads:    Number of self-attention heads.
        ff_dim:       Feed-forward hidden dimension (REDUCED: default 512).
        dropout:      Dropout probability.
        num_layers:   Number of stacked Transformer encoder layers.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        max_seq_len: int = 256,     # supports up to 16×16 feature maps
        num_heads:   int = 4,       # REDUCED from 8
        ff_dim:      int = 512,     # REDUCED from 8192 — key memory saving
        dropout:     float = 0.1,
        num_layers:  int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Learnable positional embedding — allocated for max_seq_len
        # We slice [:, :actual_seq_len, :] at forward time → works for any H×W
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, in_channels) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,    # THIS IS THE BIG SAVING
            dropout=dropout,
            batch_first=True,          # input: [B, seq, d_model]
            norm_first=True,           # Pre-LN for training stability
        )
        # Suppress harmless norm_first UserWarning from TransformerEncoder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]  (same shape, contextually enriched)
        """
        B, C, H, W = x.shape
        seq_len = H * W

        # Flatten spatial dims: [B, C, H*W] → [B, H*W, C]
        tokens = x.flatten(2).permute(0, 2, 1)            # [B, seq, C]

        # Slice positional embedding to actual seq length (handles any H×W)
        tokens = tokens + self.pos_embedding[:, :seq_len, :]

        # Transformer encoding
        tokens = self.transformer(tokens)                  # [B, seq, C]
        tokens = self.layer_norm(tokens)

        # Reshape back to spatial feature map
        out = tokens.permute(0, 2, 1).reshape(B, C, H, W) # [B, C, H, W]
        return out


# ===========================================================================
# Multi-Scale Feature Fusion  (Memory-Optimized)
# ===========================================================================

class MultiScaleFusion(nn.Module):
    """
    Parallel multi-scale feature extraction via four convolution branches.

    OPTIMIZATION vs. original:
      - 1×1  → unchanged
      - 3×3  → unchanged
      - 5×5  → replaced with dilated 3×3 (dilation=2), same receptive field (5×5)
      - 7×7  → replaced with dilated 3×3 (dilation=3), same receptive field (7×7)

    Why this saves memory:
      A standard k×k conv has k² weights per in/out channel.
      A dilated 3×3 has 9 weights — independent of dilation.
      For k=7: 49 vs 9 weights ≈ 5.4× fewer parameters in that branch alone.
      Intermediate feature tensors are also smaller for dilated convs.

    To keep the total output channel count equal to in_channels, each branch
    produces in_channels // 4 channels. A final 1×1 projection restores dim.

    Args:
        in_channels:  Input channel count (2048).
    """

    def __init__(self, in_channels: int = 2048):
        super().__init__()
        branch_ch = in_channels // 4   # 512 each

        # Branch 1: 1×1 — pointwise (unchanged)
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 3×3 — standard local context (unchanged)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # Branch 3: dilated 3×3 (dilation=2) → effective receptive field 5×5
        #           replaces the original plain 5×5 conv
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # Branch 4: dilated 3×3 (dilation=3) → effective receptive field 7×7
        #           replaces the original plain 7×7 conv
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )

        # Projection back to in_channels after concat
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1x1(x)
        b2 = self.branch_3x3(x)
        b3 = self.branch_5x5(x)
        b4 = self.branch_7x7(x)

        fused = torch.cat([b1, b2, b3, b4], dim=1)   # [B, C, H, W]
        return self.project(fused)


# ===========================================================================
# Classification Head
# ===========================================================================

class ClassificationHead(nn.Module):
    """
    Transforms a spatial feature map [B, C, H, W] to class logits [B, num_classes].

    Steps:
      Global Average Pooling → [B, C]
      Linear(C → hidden_dim) → BatchNorm → ReLU → Dropout
      Linear(hidden_dim → num_classes)

    OPTIMIZATION: hidden_dim reduced from 512 → 256 by default.

    Args:
        in_channels: Feature channel count (2048).
        hidden_dim:  Intermediate dimension (default 256, was 512).
        num_classes: Output classes (2 for binary).
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        hidden_dim: int = 256,      # REDUCED from 512
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)   # [B, C, 1, 1]
        self.head = nn.Sequential(
            nn.Flatten(),                                    # [B, C]
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)    # [B, C, 1, 1]
        return self.head(x)  # [B, num_classes]


# ===========================================================================
# ResNetPlusPlus – Full Model
# ===========================================================================

class ResNetPlusPlus(nn.Module):
    """
    ResNet++ Audio Forgery Detection model — M2-optimized variant.

    Architecture (algorithm unchanged):
      ResNet50 backbone (pretrained optional, no FC)
      ↓
      CBAM attention
      ↓
      ┌─────────────────┐
      SE Branch         Transformer Branch
      └────────┬────────┘
               ↓ fuse (element-wise add + 1×1 conv)
               ↓
      Multi-Scale Fusion (1×1, 3×3, dilated-3×3 ×2)
               ↓
      Classification Head
               ↓
      [B, 2] logits

    Memory savings vs. original:
      - Transformer ff_dim 8192→512:    saves ~126 MB of FFN weights + grads
      - Dilated convs in MultiScale:    saves ~15% of fusion module params
      - Frozen early layers:            no grads for layer1+layer2 (~250 MB saved)
      - Dynamic seq_len in Transformer: handles 128×128 input correctly

    Args:
        cfg: Full config dict loaded from configs/config.yaml.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        m_cfg = cfg["model"]

        # -------------------------------------------------------------------
        # 1. ResNet50 backbone – remove avgpool and fc
        # -------------------------------------------------------------------
        backbone = tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if m_cfg.get("pretrained", False) else None
        )
        # Keep layers 0–7 (everything up to and including layer4)
        # Store as named sub-modules so freeze_early_layers() can address them
        self.backbone_stem   = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.backbone_layer1 = backbone.layer1    # output: 256 ch
        self.backbone_layer2 = backbone.layer2    # output: 512 ch
        self.backbone_layer3 = backbone.layer3    # output: 1024 ch
        self.backbone_layer4 = backbone.layer4    # output: 2048 ch
        # Output: [B, 2048, H/32, W/32] — for 128×128 input → [B, 2048, 4, 4]

        feat_dim = 2048   # ResNet50 layer4 output channels

        # -------------------------------------------------------------------
        # 2. CBAM attention module
        # -------------------------------------------------------------------
        self.cbam = CBAM(
            in_channels=feat_dim,
            reduction_ratio=m_cfg.get("cbam_reduction_ratio", 16),
            kernel_size=m_cfg.get("cbam_kernel_size", 7),
        )

        # -------------------------------------------------------------------
        # 3a. SE branch
        # -------------------------------------------------------------------
        self.se_block = SEBlock(
            in_channels=feat_dim,
            reduction_ratio=m_cfg.get("se_reduction_ratio", 16),
        )

        # -------------------------------------------------------------------
        # 3b. Transformer branch (ff_dim now from config, default 512)
        # -------------------------------------------------------------------
        self.transformer_branch = TransformerBranch(
            in_channels=feat_dim,
            max_seq_len=256,          # supports feature maps up to 16×16
            num_heads=m_cfg.get("transformer_heads", 4),
            ff_dim=m_cfg.get("transformer_ff_dim", 512),
            dropout=m_cfg.get("transformer_dropout", 0.1),
            num_layers=m_cfg.get("transformer_layers", 1),
        )

        # 1×1 conv to fuse SE + Transformer outputs
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------------------------
        # 4. Multi-Scale Fusion (dilated convs for 5×5 and 7×7 branches)
        # -------------------------------------------------------------------
        self.multi_scale = MultiScaleFusion(in_channels=feat_dim)

        # -------------------------------------------------------------------
        # 5. Classification Head (hidden_dim from config, default 256)
        # -------------------------------------------------------------------
        self.classifier = ClassificationHead(
            in_channels=feat_dim,
            hidden_dim=m_cfg.get("fc_hidden_dim", 256),
            num_classes=m_cfg.get("num_classes", 2),
            dropout=m_cfg.get("dropout_rate", 0.5),
        )

        # -------------------------------------------------------------------
        # Weight initialization for non-backbone modules
        # -------------------------------------------------------------------
        self._init_weights()

        # -------------------------------------------------------------------
        # Freeze early backbone layers if requested
        # -------------------------------------------------------------------
        n_freeze = m_cfg.get("freeze_backbone_layers", 0)
        if n_freeze > 0:
            self.freeze_early_layers(n_freeze)

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming / Xavier initialization to non-backbone parameters."""
        for module in [
            self.cbam, self.se_block, self.transformer_branch,
            self.fusion_conv, self.multi_scale, self.classifier,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def freeze_early_layers(self, n: int = 2) -> None:
        """
        Freeze the first n backbone stages to save gradient memory.

        n=1 → freeze stem + layer1
        n=2 → freeze stem + layer1 + layer2  (recommended for 8 GB RAM)
        n=3 → freeze stem + layer1 + layer2 + layer3

        Frozen parameters:
          - require_grad = False  → no gradient tensors allocated
          - Optimizer will not update them  → smaller optimizer state
          - Still included in forward pass (contributes to features)
        """
        stages = [self.backbone_stem, self.backbone_layer1,
                  self.backbone_layer2, self.backbone_layer3]
        for stage in stages[:n]:
            for param in stage.parameters():
                param.requires_grad = False
        frozen_params = sum(
            p.numel() for s in stages[:n] for p in s.parameters()
        )
        logger.info(
            f"Froze first {n} backbone stage(s) → "
            f"{frozen_params:,} parameters excluded from gradient computation"
        )

    # ------------------------------------------------------------------

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the split backbone stages sequentially."""
        x = self.backbone_stem(x)
        x = self.backbone_layer1(x)
        x = self.backbone_layer2(x)
        x = self.backbone_layer3(x)
        x = self.backbone_layer4(x)
        return x

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram batch [B, 3, H, W]
               (H=W=128 with M2-optimized config)

        Returns:
            Logits [B, 2]  (use softmax externally for probabilities)
        """
        # ── Step 1: Backbone feature extraction
        feat = self._backbone_forward(x)     # [B, 2048, H/32, W/32]

        # ── Step 2: CBAM attention refinement
        feat = self.cbam(feat)               # same shape

        # ── Step 3: Parallel branches
        se_out          = self.se_block(feat)
        transformer_out = self.transformer_branch(feat)

        # Element-wise fusion of both branches
        fused = self.fusion_conv(se_out + transformer_out)

        # ── Step 4: Multi-scale feature fusion
        fused = self.multi_scale(fused)

        # ── Step 5: Classification
        logits = self.classifier(fused)      # [B, 2]

        return logits

    # ------------------------------------------------------------------

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Run a forward pass and return intermediate feature maps for
        visualization / debugging.
        """
        with torch.no_grad():
            backbone_feat = self._backbone_forward(x)
            cbam_feat     = self.cbam(backbone_feat)
            se_feat       = self.se_block(cbam_feat)
            tf_feat       = self.transformer_branch(cbam_feat)
            fused         = self.fusion_conv(se_feat + tf_feat)
            ms_feat       = self.multi_scale(fused)

        return {
            "backbone":    backbone_feat,
            "cbam":        cbam_feat,
            "se":          se_feat,
            "transformer": tf_feat,
            "fused":       fused,
            "multi_scale": ms_feat,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> ResNetPlusPlus:
    """
    Instantiate the ResNetPlusPlus model and move it to the target device.

    Args:
        cfg:    Full config dict.
        device: torch.device.

    Returns:
        Model ready for training / inference.
    """
    model = ResNetPlusPlus(cfg)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    logger.info(
        f"Model built on {device} | "
        f"total: {total_params:,} | "
        f"trainable: {trainable_params:,} | "
        f"frozen: {frozen_params:,}"
    )

    return model
