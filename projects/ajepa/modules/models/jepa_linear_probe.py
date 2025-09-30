# ajepa/modules/models/ijepa_linear_probe.py

import torch
import torch.nn as nn
from typing import Union, Tuple

from ajepa.ijepa.src.models.vision_transformer import (
    vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant
)

# registry
VIT_FNS = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_huge": vit_huge,
    "vit_giant": vit_giant,
}


class LinearProbeJepa(nn.Module):
    """
    Minimal linear probing:
      - load pretrained encoder
      - freeze encoder
      - average pool over patch embeddings
      - train only a linear classifier
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 21,
        model_name: str = "vit_large",
        patch_size: int = 16,
        crop_size: Union[int, Tuple[int, int]] = 224,
        *args, **kwargs
    ):
        super().__init__()

        img_size = (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)

        # encoder backbone
        self.encoder = VIT_FNS[model_name](patch_size=patch_size, img_size=img_size)
        self._load_checkpoint(checkpoint_path)

        # freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False

        # classifier
        self.classifier = nn.Linear(self.encoder.embed_dim, num_classes)
        self.class_mask = None

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))

        # keep only encoder weights
        cleaned = {k.replace("model.encoder.", "").replace("encoder.", ""): v
                   for k, v in state_dict.items() if "encoder" in k}

        info = self.encoder.load_state_dict(cleaned, strict=False)
        print(f"[INFO] Loaded encoder from {checkpoint_path}")
        print(f"[INFO] Missing/unexpected keys: {info}")

    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor:
        z = self.encoder(input_values)  # (B, N, D)
        pooled = z.mean(dim=1)          # average pooling
        return self.classifier(pooled)

