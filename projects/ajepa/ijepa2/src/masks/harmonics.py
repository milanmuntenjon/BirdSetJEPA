import math
from multiprocessing import Value
from logging import getLogger

import torch
from torch.utils.data import default_collate

from .multiblock import MaskCollator as BlockMaskCollator

logger = getLogger(__name__)


class MaskCollator(object):

    # -----------------------------------------------------------------------------
    # Curriculum masking with 3 phases (A-JEPA style):
    #   1. Block masking (early, via multiblock collator)
    #   2. Time masking (mid)
    #   3. Harmonic masking (late, preferred)
    #
    # Annealing follows Bengio et al. (2009); Hacohen & Weinshall (2019);
    # Fei (2021); Platanios et al. (2019).
    # -----------------------------------------------------------------------------
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
            return i.value

    # Scheduling weights for block / time / harmonics
    def _schedule_weights(self, step: int):
        S = self.total_steps
        c0 = self.c0

        # Block: high early, decays
        f1 = 1.0 - math.sqrt(step / (S + c0**2))

        # Time: Gaussian bump mid-training
        sigma = 0.2 * S
        f2 = math.exp(-((step - 0.5 * S) ** 2) / (2 * sigma**2))

        # Harmonics: low early, grows
        f3 = math.sqrt(step / (S + c0**2))

        weights = torch.tensor([f1, f2, f3], dtype=torch.float32)
        weights = torch.clamp(weights, min=1e-6)
        weights = weights / weights.sum()
        return weights


    # Time masking (SpecAugment-style, but patch-level
    def _time_mask(self):
        span = torch.randint(max(1, self.width // 20), max(2, self.width // 8), (1,)).item()
        start = torch.randint(0, max(1, self.width - span), (1,)).item()

        enc = torch.ones((self.height, self.width), dtype=torch.int32)
        pred = torch.ones((self.height, self.width), dtype=torch.int32)

        enc[:, start:start+span] = 0
        pred[:, start:start+span] = 0

        return torch.nonzero(enc.flatten()).squeeze(), torch.nonzero(pred.flatten()).squeeze()

    # Harmonic masking (mask harmonic rows across time)
    def _harmonic_mask(self):
        base_bin = torch.randint(max(1, self.height // 20), max(2, self.height // 10), (1,)).item()
        nbands = torch.randint(3, 6, (1,)).item()
        width = max(1, self.height // 40)

        enc = torch.ones((self.height, self.width), dtype=torch.int32)
        pred = torch.ones((self.height, self.width), dtype=torch.int32)

        for k in range(1, nbands + 1):
            center = k * base_bin
            if center >= self.height:
                break
            r0 = max(0, center - width)
            r1 = min(self.height, center + width + 1)
            enc[r0:r1, :] = 0
            pred[r0:r1, :] = 0

        return torch.nonzero(enc.flatten()).squeeze(), torch.nonzero(pred.flatten()).squeeze()

    # Collate
    def __call__(self, batch):
        if torch.is_tensor(batch):
            collated_batch = batch
            B = batch.size(0)
        else:
            B = len(batch)
            collated_batch = default_collate(batch)

        step = self.step()
        probs = self._schedule_weights(step)
        choice = torch.multinomial(probs, num_samples=1).item()  

        collated_masks_enc, collated_masks_pred = [], []
        min_keep_enc = self.height * self.width
        min_keep_pred = self.height * self.width

        for _ in range(B):
            masks_e, masks_p = [], []

            if choice == 0:
                _, enc_b, pred_b = self.block_collator([torch.zeros(1)])
                for m in enc_b[0]:
                    masks_e.append(m)
                    min_keep_enc = min(min_keep_enc, int(m.numel()))
                for m in pred_b[0]:
                    masks_p.append(m)
                    min_keep_pred = min(min_keep_pred, int(m.numel()))

            elif choice == 1:
                m_enc, m_pred = self._time_mask()
                for _ in range(self.nenc):
                    masks_e.append(m_enc)
                    min_keep_enc = min(min_keep_enc, int(m_enc.numel()))
                for _ in range(self.npred):
                    masks_p.append(m_pred)
                    min_keep_pred = min(min_keep_pred, int(m_pred.numel()))

            else:
                m_enc, m_pred = self._harmonic_mask()
                for _ in range(self.nenc):
                    masks_e.append(m_enc)
                    min_keep_enc = min(min_keep_enc, int(m_enc.numel()))
                for _ in range(self.npred):
                    masks_p.append(m_pred)
                    min_keep_pred = min(min_keep_pred, int(m_pred.numel()))

            collated_masks_enc.append(masks_e)
            collated_masks_pred.append(masks_p)

        collated_masks_enc = [[m[:min_keep_enc] for m in mlist] for mlist in collated_masks_enc]
        collated_masks_pred = [[m[:min_keep_pred] for m in mlist] for mlist in collated_masks_pred]

        collated_masks_enc = default_collate(collated_masks_enc)
        collated_masks_pred = default_collate(collated_masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred
