"""
model.py
========
ResNet++ Audio Forgery Detection Model.

Architecture overview:
  ┌────────────────────────────────────────────────────────┐
  │  Input: [B, 3, 224, 224]                               │
  │                                                        │
  │  ┌─────────────────┐                                   │
  │  │  ResNet50        │  (pretrained, fc removed)        │
  │  │  → [B, 2048, 7, 7]                                  │
  │  └────────┬─────────┘                                  │
  │           │                                            │
  │  ┌────────▼─────────┐                                  │
  │  │  CBAM Module     │  Channel + Spatial Attention     │
  │  │  → [B, 2048, 7, 7]                                  │
  │  └────────┬─────────┘                                  │
  │           │                                            │
  │    ┌──────┴───────┐                                    │
  │    │              │                                    │
  │  ┌─▼──────┐  ┌───▼────────────┐                       │
  │  │SE Block│  │Transformer Branch│                      │
  │  └─┬──────┘  └───┬────────────┘                       │
  │    │              │                                    │
  │    └──────┬───────┘                                    │
  │           │  (element-wise sum / concat+conv)          │
  │  ┌────────▼─────────┐                                  │
  │  │Multi-Scale Fusion │  1×1, 3×3, 5×5, 7×7 parallel   │
  │  └────────┬─────────┘                                  │
  │  ┌────────▼─────────┐                                  │
  │  │Classification Head│  GAP → FC(2048→512) → BN →     │
  │  │                   │  ReLU → Dropout → FC(512→2)     │
  │  └────────┬─────────┘                                  │
  │           │                                            │
  │  Output logits: [B, 2]                                 │
  └────────────────────────────────────────────────────────┘
"""

import logging
import math
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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Channel attention map: [B, C, 1, 1]
        """
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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Spatial attention map: [B, 1, H, W]
        """
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

    Process:
      1. Flatten [B, C, H, W] → [B, H*W, C]  (sequence of spatial patches)
      2. Add learnable positional embedding
      3. Apply N Transformer encoder layers (multi-head self-attention + FFN)
      4. Reshape back to [B, C, H, W]

    Args:
        in_channels:  Feature dimension C (= 2048 for ResNet50 layer4).
        seq_len:      Sequence length = H * W (= 49 for 7×7 feature map).
        num_heads:    Number of self-attention heads.
        ff_dim:       Feed-forward hidden dimension.
        dropout:      Dropout probability.
        num_layers:   Number of stacked Transformer encoder layers.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        seq_len:     int = 49,        # 7 × 7
        num_heads:   int = 8,
        ff_dim:      int = 8192,
        dropout:     float = 0.1,
        num_layers:  int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len     = seq_len

        # Learnable positional embedding [1, seq_len, C]
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, in_channels) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,        # input: [B, seq, d_model]
            norm_first=True,         # Pre-LN for training stability
        )
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

        # Flatten spatial dims: [B, C, H*W] → [B, H*W, C]
        tokens = x.flatten(2).permute(0, 2, 1)            # [B, seq, C]

        # Positional encoding (broadcast across batch)
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1], :]

        # Transformer encoding
        tokens = self.transformer(tokens)                  # [B, seq, C]
        tokens = self.layer_norm(tokens)

        # Reshape back to spatial feature map
        out = tokens.permute(0, 2, 1).reshape(B, C, H, W) # [B, C, H, W]
        return out


# ===========================================================================
# Multi-Scale Feature Fusion
# ===========================================================================

class MultiScaleFusion(nn.Module):
    """
    Applies parallel convolutions with kernels of size 1×1, 3×3, 5×5, 7×7
    on the same input and concatenates the outputs channel-wise.

    To keep the total output channel count equal to in_channels, each branch
    produces in_channels // 4 channels (assuming in_channels divisible by 4).
    A final 1×1 projection restores the channel dimension to in_channels.

    Args:
        in_channels:  Input channel count (2048).
    """

    def __init__(self, in_channels: int = 2048):
        super().__init__()
        branch_ch = in_channels // 4   # 512 each

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=7, padding=3, bias=False),
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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]  (same spatial, multi-scale context fused)
        """
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

    Args:
        in_channels: Feature channel count (2048).
        hidden_dim:  Intermediate dimension (512).
        num_classes: Output classes (2 for binary).
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        hidden_dim: int = 512,
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
    ResNet++ Audio Forgery Detection model.

    Architecture:
      ResNet50 backbone (pretrained, no FC)
      ↓
      CBAM attention
      ↓
      ┌─────────────────┐
      SE Branch         Transformer Branch
      └────────┬────────┘
               ↓ fuse (element-wise add)
               ↓
      Multi-Scale Fusion (1×1, 3×3, 5×5, 7×7)
               ↓
      Classification Head
               ↓
      [B, 2] logits

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
            weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if m_cfg["pretrained"] else None
        )
        # Keep layers 0–7 (everything up to and including layer4)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # Output: [B, 2048, 7, 7] for 224×224 input

        feat_dim = 2048   # ResNet50 layer4 output channels

        # -------------------------------------------------------------------
        # 2. CBAM attention module
        # -------------------------------------------------------------------
        self.cbam = CBAM(
            in_channels=feat_dim,
            reduction_ratio=m_cfg["cbam_reduction_ratio"],
            kernel_size=m_cfg["cbam_kernel_size"],
        )

        # -------------------------------------------------------------------
        # 3a. SE branch
        # -------------------------------------------------------------------
        self.se_block = SEBlock(
            in_channels=feat_dim,
            reduction_ratio=m_cfg["se_reduction_ratio"],
        )

        # -------------------------------------------------------------------
        # 3b. Transformer branch
        # -------------------------------------------------------------------
        self.transformer_branch = TransformerBranch(
            in_channels=feat_dim,
            seq_len=7 * 7,           # 49 tokens from 7×7 feature map
            num_heads=m_cfg["transformer_heads"],
            ff_dim=m_cfg["transformer_ff_dim"],
            dropout=m_cfg["transformer_dropout"],
            num_layers=m_cfg["transformer_layers"],
        )

        # 1×1 conv to fuse SE + Transformer outputs back to feat_dim
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------------------------
        # 4. Multi-Scale Fusion
        # -------------------------------------------------------------------
        self.multi_scale = MultiScaleFusion(in_channels=feat_dim)

        # -------------------------------------------------------------------
        # 5. Classification Head
        # -------------------------------------------------------------------
        self.classifier = ClassificationHead(
            in_channels=feat_dim,
            hidden_dim=m_cfg["fc_hidden_dim"],
            num_classes=m_cfg["num_classes"],
            dropout=m_cfg["dropout_rate"],
        )

        # -------------------------------------------------------------------
        # Weight initialization for non-backbone modules
        # -------------------------------------------------------------------
        self._init_weights()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram batch [B, 3, 224, 224]

        Returns:
            Logits [B, 2]  (use softmax externally for probabilities)
        """
        # ── Step 1: Backbone feature extraction
        feat = self.backbone(x)             # [B, 2048, 7, 7]

        # ── Step 2: CBAM attention refinement
        feat = self.cbam(feat)              # [B, 2048, 7, 7]

        # ── Step 3: Parallel branches
        se_out          = self.se_block(feat)            # [B, 2048, 7, 7]
        transformer_out = self.transformer_branch(feat)  # [B, 2048, 7, 7]

        # Element-wise fusion of both branches
        fused = self.fusion_conv(se_out + transformer_out)  # [B, 2048, 7, 7]

        # ── Step 4: Multi-scale feature fusion
        fused = self.multi_scale(fused)     # [B, 2048, 7, 7]

        # ── Step 5: Classification
        logits = self.classifier(fused)     # [B, 2]

        return logits

    # ------------------------------------------------------------------

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Run a forward pass and return intermediate feature maps for
        visualization / debugging.

        Returns dict with keys:
          'backbone', 'cbam', 'se', 'transformer', 'fused', 'multi_scale'
        """
        with torch.no_grad():
            backbone_feat = self.backbone(x)
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
    logger.info(
        f"Model built | total params: {total_params:,} | "
        f"trainable: {trainable_params:,}"
    )

    return model
