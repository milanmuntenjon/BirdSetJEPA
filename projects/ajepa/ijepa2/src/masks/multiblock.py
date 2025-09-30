import math
from multiprocessing import Value
from logging import getLogger
import torch

logger = getLogger()


class MaskCollator(object):
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
            input_size = (input_size, input_size)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)

        self.height = input_size[0] // patch_size[0]
        self.width = input_size[1] // patch_size[1]
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

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def _schedule(self, step):
        # sqrt annealing
        return min(1.0, math.sqrt(step / (self.total_steps + self.c0**2)) + self.c0)

    def _timefreq_mask(self, h, w, mask_h, mask_w):
        """Return indices to DROP (mask) for encoder + predictor."""
        # encoder: mask inside the block
        enc_grid = torch.zeros((h, w), dtype=torch.int32)
        top = torch.randint(0, h - mask_h + 1, (1,)).item()
        left = torch.randint(0, w - mask_w + 1, (1,)).item()
        enc_grid[top:top + mask_h, left:left + mask_w] = 1

        # predictor: mask whole rows + cols crossing the block
        pred_grid = torch.zeros((h, w), dtype=torch.int32)
        pred_grid[top:top + mask_h, :] = 1
        pred_grid[:, left:left + mask_w] = 1

        m_enc = torch.nonzero(enc_grid.flatten(), as_tuple=False).squeeze(-1).to(torch.long)
        m_pred = torch.nonzero(pred_grid.flatten(), as_tuple=False).squeeze(-1).to(torch.long)
        return m_enc, m_pred

    def __call__(self, batch):
        # normalize batch shape like multiblock
        if torch.is_tensor(batch):
            collated_batch = batch
            B = batch.size(0)
        else:
            B = len(batch)
            collated_batch = torch.utils.data.default_collate(batch)

        step = self.step()
        if torch.rand(1).item() < self._schedule(step):
            # --- Time–frequency masking ---
            collated_masks_enc, collated_masks_pred = [], []
            min_keep_enc = self.height * self.width
            min_keep_pred = self.height * self.width

            for _ in range(B):
                # predictor: generate npred masks
                masks_p = []
                for _ in range(self.npred):
                    mh = torch.randint(int(self.height * 0.15), int(self.height * 0.2) + 1, (1,)).item()
                    mw = torch.randint(int(self.width * 0.15), int(self.width * 0.2) + 1, (1,)).item()
                    m_enc, m_pred = self._timefreq_mask(self.height, self.width, mh, mw)
                    masks_p.append(m_pred)
                    min_keep_pred = min(min_keep_pred, m_pred.numel())

                # encoder: generate nenc masks
                masks_e = []
                for _ in range(self.nenc):
                    mh = torch.randint(int(self.height * 0.15), int(self.height * 0.2) + 1, (1,)).item()
                    mw = torch.randint(int(self.width * 0.15), int(self.width * 0.2) + 1, (1,)).item()
                    m_enc, _ = self._timefreq_mask(self.height, self.width, mh, mw)
                    masks_e.append(m_enc)
                    min_keep_enc = min(min_keep_enc, m_enc.numel())

                collated_masks_pred.append(masks_p)
                collated_masks_enc.append(masks_e)

            # truncate to uniform lengths
            collated_masks_pred = [[m[:min_keep_pred] for m in mlist] for mlist in collated_masks_pred]
            collated_masks_enc = [[m[:min_keep_enc] for m in mlist] for mlist in collated_masks_enc]

            collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)  # (B, npred, L)
            collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)    # (B, nenc, L)

            return collated_batch, collated_masks_enc, collated_masks_pred

        else:
            # --- Fallback: behave like multiblock (safe path) ---
            from .multiblock import MaskCollator as BlockMaskCollator
            block_collator = BlockMaskCollator(
                input_size=(self.height * self.patch_size[0], self.width * self.patch_size[1]),
                patch_size=self.patch_size,
                enc_mask_scale=self.enc_mask_scale,
                pred_mask_scale=self.pred_mask_scale,
                aspect_ratio=self.aspect_ratio,
                nenc=self.nenc,
                npred=self.npred,
                min_keep=self.min_keep,
                allow_overlap=self.allow_overlap,
            )
            return block_collator(batch)
