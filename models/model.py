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
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        bottleneck = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return attn


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# ===========================================================================
# SE Block – Squeeze-and-Excitation
# ===========================================================================

class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        bottleneck = max(1, in_channels // reduction_ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        scale = scale.view(B, C, 1, 1)
        return x * scale


# ===========================================================================
# Transformer Branch
# ===========================================================================

class TransformerBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 2048,
        seq_len: int = 49,
        num_heads: int = 8,
        ff_dim: int = 8192,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, in_channels) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1], :]
        tokens = self.transformer(tokens)
        tokens = self.layer_norm(tokens)
        out = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return out


# ===========================================================================
# Multi-Scale Feature Fusion
# ===========================================================================

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        branch_ch = in_channels // 4
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
        fused = torch.cat([b1, b2, b3, b4], dim=1)
        return self.project(fused)


# ===========================================================================
# Classification Head
# ===========================================================================

class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int = 2048, hidden_dim: int = 512, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        return self.head(x)


# ===========================================================================
# ResNetPlusPlus – Full Model
# ===========================================================================

class ResNetPlusPlus(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m_cfg = cfg["model"]
        backbone = tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if m_cfg.get("pretrained", False) else None
        )
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
        feat_dim = 2048
        self.cbam = CBAM(
            in_channels=feat_dim,
            reduction_ratio=m_cfg.get("cbam_reduction_ratio", 16),
            kernel_size=m_cfg.get("cbam_kernel_size", 7),
        )
        self.se_block = SEBlock(
            in_channels=feat_dim,
            reduction_ratio=m_cfg.get("se_reduction_ratio", 16),
        )
        self.transformer_branch = TransformerBranch(
            in_channels=feat_dim,
            seq_len=7 * 7,
            num_heads=m_cfg.get("transformer_heads", 8),
            ff_dim=m_cfg.get("transformer_ff_dim", 4096),
            dropout=m_cfg.get("transformer_dropout", 0.1),
            num_layers=m_cfg.get("transformer_layers", 1),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.multi_scale = MultiScaleFusion(in_channels=feat_dim)
        self.classifier = ClassificationHead(
            in_channels=feat_dim,
            hidden_dim=m_cfg.get("fc_hidden_dim", 512),
            num_classes=m_cfg.get("num_classes", 2),
            dropout=m_cfg.get("dropout_rate", 0.5),
        )
        self._init_weights()

    def _init_weights(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.cbam(feat)
        se_out = self.se_block(feat)
        transformer_out = self.transformer_branch(feat)
        fused = self.fusion_conv(se_out + transformer_out)
        fused = self.multi_scale(fused)
        logits = self.classifier(fused)
        return logits

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            backbone_feat = self.backbone(x)
            cbam_feat = self.cbam(backbone_feat)
            se_feat = self.se_block(cbam_feat)
            tf_feat = self.transformer_branch(cbam_feat)
            fused = self.fusion_conv(se_feat + tf_feat)
            ms_feat = self.multi_scale(fused)
        return {
            "backbone": backbone_feat,
            "cbam": cbam_feat,
            "se": se_feat,
            "transformer": tf_feat,
            "fused": fused,
            "multi_scale": ms_feat,
        }


# ===========================================================================
# Factory function
# ===========================================================================

def build_model(cfg: dict, device: torch.device) -> ResNetPlusPlus:
    model = ResNetPlusPlus(cfg)
    model = model.to(device)
    return model
