"""Optional PyTorch 3D model builders; imports Torch only when a builder is called.

Both networks consume/produce channel-first tensors ``(batch, channels, inline,
crossline, sample)``. PyTorch is supplied by the authorized HPC environment; this
module deliberately does not install, initialize CUDA, or train at import time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvAutoencoder3DConfig:
    in_channels: int = 1
    channels: tuple[int, ...] = (8, 16, 32, 64)

    def __post_init__(self) -> None:
        if self.in_channels < 1 or not self.channels or any(value < 2 for value in self.channels):
            raise ValueError("autoencoder channel sizes must be positive and nonempty")


def _torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for 3D neural models; use the authorized GPU environment") from exc
    return torch, nn, F


def _group_count(channels: int) -> int:
    return next(group for group in range(min(8, channels), 0, -1) if channels % group == 0)


def _conv_block(nn, in_channels: int, out_channels: int, *, stride: int = 1):
    groups = _group_count(out_channels)
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.GELU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.GELU())


def build_siamese_3d(*, in_channels: int = 1, channels: tuple[int, ...] = (8, 16, 32)):
    """Build a shared-encoder 3D network returning voxel change logits, not probabilities."""
    torch, nn, F = _torch()
    if in_channels < 1 or not channels or any(size < 2 for size in channels):
        raise ValueError("invalid Siamese channel configuration")

    class SharedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            blocks, previous = [], in_channels
            for width in channels:
                blocks.append(_conv_block(nn, previous, width, stride=2))
                previous = width
            self.blocks = nn.ModuleList(blocks)

        def forward(self, tensor):
            outputs = []
            for block in self.blocks:
                tensor = block(tensor)
                outputs.append(tensor)
            return outputs

    class Siamese3DNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SharedEncoder()  # same parameters for baseline and monitor
            self.projections = nn.ModuleList([nn.Conv3d(width*2, channels[0], 1) for width in channels])
            self.change_head = nn.Sequential(_conv_block(nn, channels[0], channels[0]),
                                             nn.Conv3d(channels[0], 1, kernel_size=1))

        def forward(self, baseline, monitor):
            if baseline.shape != monitor.shape or baseline.ndim != 5:
                raise ValueError("Siamese inputs must be matching B,C,inline,crossline,sample tensors")
            base_features, monitor_features = self.encoder(baseline), self.encoder(monitor)
            feature_size = base_features[0].shape[2:]
            fused = None
            for base, monitor_value, projection in zip(base_features, monitor_features, self.projections):
                # Signed differences preserve polarity; absolute differences encode magnitude.
                compared = torch.cat((monitor_value-base, torch.abs(monitor_value-base)), dim=1)
                level = projection(compared)
                if level.shape[2:] != feature_size:
                    level = F.interpolate(level, size=feature_size, mode="trilinear", align_corners=False)
                fused = level if fused is None else fused + level
            return F.interpolate(self.change_head(fused), size=baseline.shape[2:],
                                 mode="trilinear", align_corners=False)

    return Siamese3DNetwork()


def build_conv_autoencoder_3d(config: ConvAutoencoder3DConfig = ConvAutoencoder3DConfig()):
    """Build a nonlinear 3D convolutional autoencoder returning reconstruction tensors.

    There are no encoder-decoder skip connections, so the bottleneck constrains direct
    identity copying. Train on a justified reference population and calibrate residuals
    separately; a reconstruction residual alone is not a probability or fluid indicator.
    """
    _, nn, F = _torch()

    class ConvAutoencoder3D(nn.Module):
        def __init__(self):
            super().__init__()
            encoder, previous = [], config.in_channels
            for width in config.channels:
                encoder.append(_conv_block(nn, previous, width, stride=2))
                previous = width
            self.encoder = nn.ModuleList(encoder)
            decoder, decoder_width = [], config.channels[-1]
            for width in reversed(config.channels[:-1]):
                decoder.append(_conv_block(nn, decoder_width, width))
                decoder_width = width
            self.decoder = nn.ModuleList(decoder)
            self.output_head = nn.Conv3d(decoder_width, config.in_channels, kernel_size=1)

        def forward(self, inputs):
            if inputs.ndim != 5 or inputs.shape[1] != config.in_channels:
                raise ValueError("autoencoder input must use B,C,inline,crossline,sample axes")
            original_size = inputs.shape[2:]
            hidden = inputs
            encoded_sizes = []
            for block in self.encoder:
                hidden = block(hidden)
                encoded_sizes.append(hidden.shape[2:])
            targets = list(reversed(encoded_sizes[:-1])) + [original_size]
            for block, target_size in zip(self.decoder, targets[:-1]):
                hidden = F.interpolate(hidden, size=target_size, mode="trilinear", align_corners=False)
                hidden = block(hidden)
            hidden = F.interpolate(hidden, size=original_size, mode="trilinear", align_corners=False)
            return self.output_head(hidden)

    return ConvAutoencoder3D()
