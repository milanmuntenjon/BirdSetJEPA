# ajepa/modules/models/ijepa_pretrain_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from ajepa.ijepa.src.models.vision_transformer import (
    vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant, vit_predictor
)
from ajepa.ijepa2.src.masks.multiblock import MaskCollator as MultiBlockMaskCollator
from ajepa.ijepa2.src.masks.curriculum import MaskCollator as TimeFreqMaskCollator
from ajepa.ijepa2.src.masks.harmonics import MaskCollator as HarmonicsMaskCollator
from ajepa.ijepa2.src.masks.utils import apply_masks
from ajepa.ijepa2.src.utils.tensors import repeat_interleave_batch


VIT_FNS = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_huge": vit_huge,
    "vit_giant": vit_giant,
}


class JEPA(nn.Module):
    """
    Pretraining-only wrapper for I-JEPA:
    - Student encoder + predictor
    - EMA teacher encoder
    - Masking strategy
    """

    def __init__(
        self,
        model_name: str = "vit_large",
        patch_size: int = 16,
        crop_size: Union[int, Tuple[int, int], List[int]] = (224, 224),

        # predictor / pretraining bits
        pred_depth: int = 24,
        pred_emb_dim: int = 1024,

        # masking
        num_enc_masks: int = 4,
        num_pred_masks: int = 4,
        enc_mask_scale: Tuple[float, float] = (0.85, 1.0),
        pred_mask_scale: Tuple[float, float] = (0.15, 0.2),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        min_keep: int = 10,
        allow_overlap: bool = False,
        masking_strategy: str = "multiblock",

        ema_decay: float = 0.996,

        # checkpoint for init
        checkpoint_path: Optional[str] = None,
        init_teacher_from_checkpoint: bool = True,
        *args, **kwargs
    ):
        super().__init__()

        # ---- image size ----
        if isinstance(crop_size, int):
            img_size = (crop_size, crop_size)
        else:
            img_size = tuple(crop_size)

        # ---- student encoder ----
        self.encoder = VIT_FNS[model_name](patch_size=patch_size, img_size=img_size)

        # ---- predictor ----
        self.predictor = vit_predictor(
            grid_size=self.encoder.patch_embed.grid_size,
            embed_dim=self.encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=self.encoder.num_heads,
        )

        # ---- teacher encoder ----
        self.teacher = VIT_FNS[model_name](patch_size=patch_size, img_size=img_size)
        self.teacher.load_state_dict(self.encoder.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ---- optional checkpoint load ----
        if checkpoint_path is not None:
            target = "teacher" if init_teacher_from_checkpoint else "student"
            self._load_checkpoint(checkpoint_path, target=target)

        # ---- masking strategy ----
        if masking_strategy == "multiblock":
            self.mask_collator = MultiBlockMaskCollator(
                input_size=img_size,
                patch_size=patch_size,
                pred_mask_scale=pred_mask_scale,
                enc_mask_scale=enc_mask_scale,
                aspect_ratio=aspect_ratio,
                nenc=num_enc_masks,
                npred=num_pred_masks,
                allow_overlap=allow_overlap,
                min_keep=min_keep,
            )
        elif masking_strategy == "timefreq":
            self.mask_collator = TimeFreqMaskCollator(
                input_size=img_size,
                patch_size=patch_size,
                pred_mask_scale=pred_mask_scale,
                enc_mask_scale=enc_mask_scale,
                aspect_ratio=aspect_ratio,
                nenc=num_enc_masks,
                npred=num_pred_masks,
                allow_overlap=allow_overlap,
                min_keep=min_keep,
            )
        elif masking_strategy == "harmonics":
            self.mask_collator = HarmonicsMaskCollator(
                input_size=img_size,
                patch_size=patch_size,
                pred_mask_scale=pred_mask_scale,
                enc_mask_scale=enc_mask_scale,
                aspect_ratio=aspect_ratio,
                nenc=num_enc_masks,
                npred=num_pred_masks,
                allow_overlap=allow_overlap,
                min_keep=min_keep,
            )
        else:
            raise ValueError(f"Unknown masking strategy: {masking_strategy}")

        # ---- misc ----
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)
        self.ema_decay = ema_decay

    # -----------------------------------------------------------------
    # Checkpoint loader (encoder-only weights)
    # -----------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_path: str, target: str = "student") -> None:
        """Load only encoder weights from a (Lightning) checkpoint into student/teacher."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model.encoder."):
                cleaned[k[len("model.encoder."):]] = v
            elif k.startswith("encoder."):
                cleaned[k[len("encoder."):]] = v
            else:
                continue

        if "pos_embed" in cleaned:
            v = cleaned["pos_embed"]
            tgt_len = self.encoder.pos_embed.shape[1]
            src_len = v.shape[1]
            if src_len == tgt_len + 1:
                cleaned["pos_embed"] = v[:, 1:, :]
            if cleaned["pos_embed"].shape[1] != tgt_len:
                cleaned.pop("pos_embed")

        module = self.teacher if target == "teacher" else self.encoder
        info = module.load_state_dict(cleaned, strict=False)

        if target == "student":
            self.teacher.load_state_dict(self.encoder.state_dict())
            
    def forward(self, x: torch.Tensor):
        """
        Forward for I-JEPA pretraining.
        Returns:
            z: student predictions
            h: teacher targets
        """
        _, masks_enc, masks_pred = self.mask_collator(x)


        # Teacher targets
        with torch.no_grad():
            h = self.teacher(x)                  
            h = F.layer_norm(h, (h.size(-1),))   # normalize tokens
            h = apply_masks(h, masks_pred)     
            B = x.shape[0]
            repeat = len(masks_enc)
            h = repeat_interleave_batch(h, B, repeat)

        # Student predictions
        z = self.encoder(x, masks_enc)           
        z = self.predictor(z, masks_enc, masks_pred)

        return z, h

    @torch.no_grad()
    def update_teacher(self):
        m = self.ema_decay
        for p_t, p_s in zip(self.teacher.parameters(), self.encoder.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1 - m)
