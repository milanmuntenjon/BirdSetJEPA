# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from multiprocessing import Value
from logging import getLogger

import torch
from .multiblock import MaskCollator as BlockMaskCollator

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):
    """
    Curriculum masking strategy (A-JEPA style).
    - Early: random block masking (Multiblock).
    - Late: time–frequency masking (mask whole rows/cols).
    - Transition controlled by sqrt-annealed probability.
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        total_steps=100000,
        c0=0.01,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.patch_size = patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.total_steps = total_steps
        self.c0 = c0

        self._itr_counter = Value("i", -1)
        # fallback collator (block masking)
        self.block_collator = BlockMaskCollator(
            input_size=input_size,
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=nenc,
            npred=npred,
            min_keep=min_keep,
            allow_overlap=allow_overlap,
        )

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _schedule(self, step):
        """Annealing probability for time–frequency masking."""
        return min(1.0, math.sqrt(step / (self.total_steps + self.c0**2)) + self.c0)

    def _time_frequency_mask(self, h, w, mask_h, mask_w, min_ratio=0.35):
        while True:
            top = torch.randint(0, h - mask_h + 1, (1,)).item()
            left = torch.randint(0, w - mask_w + 1, (1,)).item()
            mask = torch.zeros((h, w), dtype=torch.int32)
            mask[top : top + mask_h, left : left + mask_w] = 1
            if mask.sum() > min_ratio * mask_h * mask_w:
                break

        # Encoder mask: keep everything outside the block
        mask_enc = torch.ones((h, w), dtype=torch.int32)
        mask_enc[top : top + mask_h, left : left + mask_w] = 0

        # Predictor mask: drop whole rows + whole columns
        mask_pred = torch.ones((h, w), dtype=torch.int32)
        mask_pred[top : top + mask_h, :] = 0
        mask_pred[:, left : left + mask_w] = 0

        # Flatten indices
        mask_enc = torch.nonzero(mask_enc.flatten()).squeeze()
        mask_pred = torch.nonzero(mask_pred.flatten()).squeeze()
        return mask_enc, mask_pred

    def __call__(self, batch):
        step = self.step()
        prob_tf = self._schedule(step)

        if torch.rand(1).item() < prob_tf:
            # --- Time-frequency masking ---
            B = len(batch)
            collated_batch = torch.utils.data.default_collate(batch)

            collated_masks_enc, collated_masks_pred = [], []
            min_keep_enc = self.height * self.width
            min_keep_pred = self.height * self.width

            for _ in range(B):
                mask_h = torch.randint(int(self.height * 0.15), int(self.height * 0.2) + 1, (1,)).item()
                mask_w = torch.randint(int(self.width * 0.15), int(self.width * 0.2) + 1, (1,)).item()
                m_enc, m_pred = self._time_frequency_mask(self.height, self.width, mask_h, mask_w)
                collated_masks_enc.append([m_enc])
                collated_masks_pred.append([m_pred])
                min_keep_enc = min(min_keep_enc, len(m_enc))
                min_keep_pred = min(min_keep_pred, len(m_pred))

            # --- truncate to same length across batch (like multiblock does) ---
            collated_masks_enc = [[m[:min_keep_enc] for m in mlist] for mlist in collated_masks_enc]
            collated_masks_pred = [[m[:min_keep_pred] for m in mlist] for mlist in collated_masks_pred]

            collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
            collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

            return collated_batch, collated_masks_enc, collated_masks_pred
        else:
            # --- Fall back to block masking ---
            return self.block_collator(batch)
