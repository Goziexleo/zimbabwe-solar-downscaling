"""Canonical ClimateU-Net architecture (Section 3.6.6) and checkpoint helpers.

Shared by train_unet_downscaler.py, evaluate_unet.py, and
generate_future_projections.py so the architecture and checkpoint schema can
never drift apart between training and inference.
"""

import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

PAD_MULTIPLE = 32  # 4 encoder pools + 1 bottleneck pool = 2^5
SCHEMA_VERSION = 5

# Deployed value, now zero. The sweep in optimise_unet.py found dropout costly,
# and Section 4.5 confirmed it under the deployed procedure: fitted at the same
# learning rate, under the same two-phase selection and the same seed, the
# dropout-free configuration scores 8.62 W/m2 over Zimbabwe against 11.66 with
# dropout at 0.3, is better on the inner split that selects, and passes the
# Section 4.4 scenario screen. Retraining the old configuration reproduces it
# bit for bit, so that gap is the setting and not the draw. Overridable, so the
# 0.3 configuration can still be measured as the comparison Section 4.5 reports.
DROPOUT_P = float(os.environ.get("UNET_DROPOUT", "0.0"))


def next_multiple(n, m):
    return int(np.ceil(n / m) * m)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU, with a
    spatial dropout after each activation. With only 312 training months
    feeding a ~31M-parameter network, unregularised training overfits almost
    immediately (validation loss rising every epoch past epoch 1) - dropout
    and weight decay (added in train_unet_downscaler.py's optimiser) are the
    standard mitigation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_P),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_P),
        )

    def forward(self, x):
        return self.double_conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // reduction, 1), in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.fc(self.avg_pool(x))
        return x * y.expand_as(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Bottleneck(nn.Module):
    """A single 5th (deepest) pooling stage followed by a 1x1 conv to 1024
    filters, as specified in Section 3.6.6."""
    def __init__(self, in_channels, out_channels=1024):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(DROPOUT_P)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(self.pool(x)))))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.ca(self.conv(x))


class ClimateUNet(nn.Module):
    """Pre-upsampling super-resolution U-Net (Lin et al. 2023; Fuentes-Franco
    et al. 2025): the coarse *atmospheric* channels are bilinearly upsampled
    to the fine target grid, then concatenated with static topographic and
    astronomical channels that are already known at native fine resolution
    (Section 3.5.2's elevation/slope/SVF are computed at 0.1 deg; SZA is
    analytically evaluable at any resolution) - unlike the CNN (Section
    3.6.5), which has an explicit coarse-only (C+3) channel formula, Section
    3.6.6 specifies no such constraint for U-Net, so degrading these
    already-precise static fields to the coarse grid would only discard
    information for no methodological reason. The concatenated tensor is then
    refined by a symmetric 4-level encoder-decoder (64/128/256/512) with a
    1024-filter bottleneck and skip connections."""
    def __init__(self, coarse_channels, fine_channels, out_channels, fine_shape):
        super().__init__()
        self.fine_shape = tuple(fine_shape)
        pad_h = next_multiple(self.fine_shape[0], PAD_MULTIPLE)
        pad_w = next_multiple(self.fine_shape[1], PAD_MULTIPLE)
        self.padded_shape = (pad_h, pad_w)

        in_channels = coarse_channels + fine_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bottleneck = Bottleneck(512, 1024)

        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)

        # No ReLU here: the network predicts a variance-scaled CSI *anomaly*
        # (target - climatology) / std, which is legitimately negative for
        # below-average months. The Section 3.5.4 non-negativity/[0, 1.1]
        # constraint is enforced downstream in predict_csi_field(), once the
        # anomaly has been denormalised back into actual CSI space.
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x_coarse, x_fine):
        x_coarse_up = F.interpolate(x_coarse, size=self.fine_shape, mode='bilinear', align_corners=True)
        x = torch.cat([x_coarse_up, x_fine], dim=1)

        pad_h = self.padded_shape[0] - self.fine_shape[0]
        pad_w = self.padded_shape[1] - self.fine_shape[1]
        x = F.pad(x, [0, pad_w, 0, pad_h])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)

        return out[:, :, :self.fine_shape[0], :self.fine_shape[1]]


def load_unet_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if checkpoint.get('schema_version') != SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint at {model_path} has schema_version={checkpoint.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}. Retrain with train_unet_downscaler.py."
        )
    model = ClimateUNet(
        coarse_channels=checkpoint['coarse_channels'],
        fine_channels=checkpoint['fine_channels'],
        out_channels=1,
        fine_shape=checkpoint['fine_shape'],
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def predict_csi_field(model, x_coarse, x_fine, climatology_mean, anomaly_std):
    """x_coarse/x_fine: (1, C, H, W) normalised inputs. Returns a
    (fine_H, fine_W) numpy CSI field, denormalised and clipped to [0, 1.1]
    (Section 3.5.4)."""
    pred_norm = model(x_coarse, x_fine)
    pred_anomaly = pred_norm.detach().cpu().numpy()[0, 0] * anomaly_std
    return np.clip(pred_anomaly + climatology_mean, 0.0, 1.1)
