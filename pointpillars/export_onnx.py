#!/usr/bin/env python3
"""Export the official MMDetection3D KITTI PointPillars checkpoint to ONNX.

The checkpoint was trained with a legacy 10-value point decoration.  To keep
voxelization and this legacy decoration on the CPU, the pillar encoder accepts
an already decorated, zero-padded tensor with shape [40000, 32, 10].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn


MAX_PILLARS = 40000
MAX_POINTS = 32
DECORATED_CHANNELS = 10
BEV_SHAPE = (1, 64, 496, 432)


class PFNLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(DECORATED_CHANNELS, 64, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.linear(features)
        x = self.norm(x.permute(0, 2, 1).contiguous())
        x = torch.relu(x).permute(0, 2, 1).contiguous()
        return torch.max(x, dim=1).values


class PillarEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pfn_layers = nn.ModuleList([PFNLayer()])

    def forward(self, decorated_features: torch.Tensor) -> torch.Tensor:
        return self.pfn_layers[0](decorated_features)


def conv_bn_relu(in_channels: int, out_channels: int, stride: int = 1) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=False),
    ]


class SECOND(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_channels = [64, 64, 128]
        out_channels = [64, 128, 256]
        layer_nums = [3, 5, 5]
        blocks = []
        for cin, cout, count in zip(in_channels, out_channels, layer_nums):
            layers = conv_bn_relu(cin, cout, stride=2)
            for _ in range(count):
                layers.extend(conv_bn_relu(cout, cout))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs[0], outputs[1], outputs[2]


class SECONDFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_channels = [64, 128, 256]
        strides = [1, 2, 4]
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(cin, 128, kernel_size=s, stride=s, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False),
            )
            for cin, s in zip(in_channels, strides)
        ])

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.cat([layer(x) for layer, x in zip(self.deblocks, features)], dim=1)


class AnchorHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_cls = nn.Conv2d(384, 18, 1)
        self.conv_reg = nn.Conv2d(384, 42, 1)
        self.conv_dir_cls = nn.Conv2d(384, 12, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.conv_cls(x), self.conv_reg(x), self.conv_dir_cls(x)


class BackboneHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SECOND()
        self.neck = SECONDFPN()
        self.bbox_head = AnchorHead()

    def forward(self, spatial_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bbox_head(self.neck(self.backbone(spatial_features)))


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    expected = state["voxel_encoder.pfn_layers.0.linear.weight"].shape
    if tuple(expected) != (64, 10):
        raise RuntimeError(f"Expected legacy PFN weight [64,10], got {tuple(expected)}")
    return state


def load_submodule(module: nn.Module, state: dict[str, torch.Tensor], prefix: str) -> None:
    subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    incompatible = module.load_state_dict(subset, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))


def export_model(module: nn.Module, inputs: torch.Tensor, output: Path,
                 input_name: str, output_names: list[str], opset: int) -> None:
    module.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=output_names,
            dynamic_axes=None,
            dynamo=False,
        )
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    print(f"OK: {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
    for value in list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        print(f"  {value.name}: {dims}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(0)
    state = load_checkpoint(args.checkpoint)

    encoder = PillarEncoder()
    load_submodule(encoder, state, "voxel_encoder.")
    export_model(
        encoder,
        torch.zeros(MAX_PILLARS, MAX_POINTS, DECORATED_CHANNELS),
        args.output_dir / "pillar_encoder.onnx",
        "decorated_features",
        ["pillar_features"],
        args.opset,
    )

    network = BackboneHead()
    # The full state also contains the encoder. Load only network prefixes.
    network_state = {
        key: value for key, value in state.items()
        if key.startswith(("backbone.", "neck.", "bbox_head."))
    }
    incompatible = network.load_state_dict(network_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    export_model(
        network,
        torch.zeros(BEV_SHAPE),
        args.output_dir / "backbone_head.onnx",
        "spatial_features",
        ["cls_scores", "bbox_preds", "dir_cls_preds"],
        args.opset,
    )


if __name__ == "__main__":
    main()
