
import torch
import torch.nn as nn
from typing import Union, Tuple
import matplotlib.pyplot as plt
import math
import os
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


class AttentiveProbeJepa(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 21,
        model_name: str = "vit_large",
        patch_size: int = 16,
        crop_size: Union[int, Tuple[int, int]] = 224,
        attn_dim: int = 256,
        vis_every: int = 1000,   
        save_dir: str = "attn_vis",
        *args, **kwargs
    ):
        super().__init__()

        img_size = (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)
        self.encoder = VIT_FNS[model_name](patch_size=patch_size, img_size=img_size)
        self._load_checkpoint(checkpoint_path)

        for p in self.encoder.parameters():
            p.requires_grad = False

        embed_dim = self.encoder.embed_dim

        # attentive pooling
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

        # visualization params
        self.vis_every = vis_every
        self.counter = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        cleaned = {k.replace("model.encoder.", "").replace("encoder.", ""): v
                   for k, v in state_dict.items() if "encoder" in k}
        info = self.encoder.load_state_dict(cleaned, strict=False)
        print(f"[INFO] Loaded encoder from {checkpoint_path}")
        print(f"[INFO] Missing/unexpected keys: {info}")

    def forward(self, input_values: torch.Tensor, return_attn: bool = False, **kwargs):
        z = self.encoder(input_values)  
        attn_scores = self.attention(z)              
        attn_weights = torch.softmax(attn_scores, 1) 

        pooled = torch.sum(attn_weights * z, dim=1)  
        logits = self.classifier(pooled)

        # optional visualization
        self.counter += input_values.size(0)
        if self.counter % self.vis_every == 0:
            self._visualize_attention(attn_weights, input_values)

        if return_attn:
            return logits, attn_weights.squeeze(-1)  
        return logits

    def _visualize_attention(self, attn_weights, input_values):
        """
        Save heatmaps of attention masks (supports rectangular inputs).
        """
        B, N, _ = attn_weights.shape

        # input shape
        _, _, H, W = input_values.shape

        # handle tuple or int patch size
        patch_size = self.encoder.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size

        grid_h, grid_w = H // patch_h, W // patch_w

        for i in range(min(B, 1)):
            attn = attn_weights[i].detach().cpu().squeeze(-1)  # (N,)

            if attn.numel() != grid_h * grid_w:
                print(f"[WARN] Attn size {attn.numel()} != {grid_h}x{grid_w}, skipping")
                return

            attn_map = attn.reshape(grid_h, grid_w)

            plt.imshow(attn_map, cmap="viridis", aspect="auto")
            plt.colorbar()
            save_path = os.path.join(self.save_dir, f"attn_{self.counter}_sample{i}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[INFO] Saved attention visualization: {save_path}")
