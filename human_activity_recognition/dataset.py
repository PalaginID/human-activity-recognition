import random
from collections.abc import Sequence
from typing import Any

import lightning as L
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .preproccessing import behavior_to_phase, make_feature_from_np
from .utils import get_phase1_behavior


class GestureDataset(Dataset):
    """Dataset holding one sequence = one sample.

    Parameters
    ----------
    df: pl.DataFrame
        Filtered dataframe containing required columns.
    seq_ids: Sequence[str]
        Sequence IDs belonging to this split.
    label2idx: Dict[str, int]
        Mapping from gesture string to integer label.
    """

    def __init__(
        self, df: pl.DataFrame, seq_ids: Sequence[Any], label2idx: dict[str, int], **kwargs
    ):
        super().__init__()
        self.seq_ids = list(seq_ids)
        self.label2idx = label2idx
        self.use_tof_mask_augmentation_prob = kwargs.get("use_tof_mask_augmentation_prob", 0)
        # Convert all sequences into numpy arrays here and cache
        self.samples: list[
            tuple[np.ndarray, np.ndarray, int, int, np.ndarray, str]
        ] = []  # (feat, tof, length, label_idx, phase, subject)
        self.use_tof = kwargs.get("use_tof", False)
        self.is_train = kwargs.get("is_train", False)
        self.rot_zero = kwargs.get("rot_zero", False)

        # Run per-sequence processing in parallel
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=36) as executor:
            # Submit each sequence_id processing in parallel
            futures = [
                executor.submit(self._process_sequence, sid, df, self.label2idx)
                for sid in self.seq_ids
            ]

            # Retrieve results in the original order and append to samples
            for future in futures:
                self.samples.append(future.result())

    def _process_sequence(self, sid, df, label2idx):
        """Process one sequence_id and return a sample tuple"""
        grp = df.filter(pl.col("sequence_id") == sid).sort("sequence_counter")

        # Features [T, F]
        subject = grp.select("subject").to_numpy().flatten()[0]
        feat = grp.select(
            ["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "rot_w", "handedness"]
        ).to_numpy()
        tof_cols = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
        tof = grp.select(tof_cols).to_numpy()

        # Extract and convert phase information
        behaviors = grp.select("behavior").to_numpy().flatten()
        phases = np.array([behavior_to_phase(b) for b in behaviors])

        # Determine label as a class index of (orientation, gesture, phase1_behavior)
        row0 = grp.row(0)
        gesture = row0[grp.columns.index("gesture")]
        orientation = row0[grp.columns.index("orientation")]
        phase1_behavior = get_phase1_behavior(df, sid)
        label_idx = label2idx[(orientation, gesture, phase1_behavior)]

        feat, tof = make_feature_from_np(feat, tof)

        return (feat, tof, len(feat), label_idx, phases, subject)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat_np, tof_np, length_int, label_idx, phases_np, subject = self.samples[idx]

        if self.rot_zero:
            feat_np[:, 3:15] = 0

        # Special subjects: SUBJ_019262, SUBJ_045235
        if subject in ["SUBJ_019262", "SUBJ_045235"]:
            feat_np[:, 0] *= -1
            feat_np[:, 1] *= -1
            feat_np[:, 3] *= -1
            feat_np[:, 4] *= -1
            feat_np[:, 5] *= -1
            feat_np[:, 6] *= -1
            feat_np[:, 7] *= -1
            feat_np[:, 8] *= -1
            feat_np[:, 9] *= -1
            feat_np[:, 10] *= -1
            feat_np[:, 11] *= -1
            feat_np[:, 12] *= -1
            feat_np[:, 13] *= -1

            tof_np[:, :] = 0

        if self.use_tof:
            if random.random() < self.use_tof_mask_augmentation_prob:
                tof_np[:, :] = 0
            feat_np = np.concatenate([feat_np, tof_np], axis=1)
        x = torch.from_numpy(feat_np).clone()
        length = torch.tensor(length_int, dtype=torch.long)

        y = torch.tensor(label_idx, dtype=torch.long)

        phases = torch.from_numpy(phases_np).clone().long()

        if x.shape[0] > 200:
            x = x[-200:]
            length = torch.tensor(200, dtype=torch.long)
            phases = phases[-200:]

        return x, length, y, phases


class MixupDataset(Dataset):
    """Dataset wrapper implementing phase-wise mixup"""

    def __init__(self, dataset: GestureDataset, alpha: float = 0.2, num_classes: int = 72):
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Fetch the original sample
        x1, length1, y1, phases1 = self.dataset[idx]

        # Randomly select another sample
        idx2 = random.randint(0, len(self.dataset) - 1)
        x2, length2, y2, phases2 = self.dataset[idx2]

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            if lam < 0.5:
                lam = 1.0 - lam
        else:
            lam = 1.0

        # Perform mixup per phase
        mixed_x, mixed_length, mixed_phases = self._mixup_by_phase(
            [x1, x2], [phases1, phases2], lam
        )

        # Mixup labels (GestureDataset already returns the correct labels)
        if y1.dim() > 0:  # soft label (tensor)
            mixed_y = lam * y1 + (1 - lam) * y2
        else:  # hard label (scalar) → make one-hot then mix
            y1_onehot = torch.zeros(self.num_classes)
            y2_onehot = torch.zeros(self.num_classes)
            y1_onehot[y1] = 1.0
            y2_onehot[y2] = 1.0
            mixed_y = lam * y1_onehot + (1 - lam) * y2_onehot

        return mixed_x, mixed_length, mixed_y, mixed_phases

    def _mixup_by_phase(self, x_couple, phase_couple, lam):
        """Perform mixup per phase"""
        # Compute indices for each phase
        x1, x2 = x_couple
        phases1, phases2 = phase_couple
        phase_indices = {}
        for phase in [0, 1, 2]:  # phase0, phase1, phase2
            idx1 = (phases1 == phase).nonzero(as_tuple=True)[0]
            idx2 = (phases2 == phase).nonzero(as_tuple=True)[0]
            phase_indices[phase] = (idx1, idx2)

        mixed_segments = []
        mixed_phase_segments = []

        for phase in [0, 1, 2]:
            idx1, idx2 = phase_indices[phase]

            if len(idx1) == 0 and len(idx2) == 0:
                # Skip if neither sample contains this phase
                continue
            elif len(idx1) == 0:
                # If x1 has no such phase, use x2 phase as is
                seg2 = x2[idx2]
                mixed_segments.append(seg2)
                mixed_phase_segments.append(torch.full((len(idx2),), phase, dtype=torch.long))
            elif len(idx2) == 0:
                # If x2 has no such phase, use x1 phase as is
                seg1 = x1[idx1]
                mixed_segments.append(seg1)
                mixed_phase_segments.append(torch.full((len(idx1),), phase, dtype=torch.long))
            else:
                # Mixup if both contain the phase
                seg1 = x1[idx1]
                seg2 = x2[idx2]

                # Match lengths to seg1
                if len(seg1) < len(seg2):
                    seg2 = seg2[: len(seg1)]
                elif len(seg1) > len(seg2):
                    padding = seg1[len(seg2) :]
                    seg2 = torch.cat([seg2, padding], dim=0)

                # Run mixup (with element-wise conditional rules)
                # Standard mixup
                mixed_seg = lam * seg1 + (1 - lam) * seg2
                # If an element in seg1 is zero, set to zero
                mixed_seg = torch.where(seg1 == 0, torch.zeros_like(seg1), mixed_seg)
                # If an element in seg2 is zero, keep seg1 value
                mixed_seg = torch.where(seg2 == 0, seg1, mixed_seg)

                mixed_segments.append(mixed_seg)
                mixed_phase_segments.append(torch.full((len(seg1),), phase, dtype=torch.long))

        # Concatenate all phases
        if mixed_segments:
            mixed_x = torch.cat(mixed_segments, dim=0)
            mixed_phases = torch.cat(mixed_phase_segments, dim=0)
            mixed_length = torch.tensor(len(mixed_x), dtype=torch.long)
        else:
            # Fallback when all phases are empty (avoid errors)
            mixed_x = torch.zeros(1, x1.shape[1])
            mixed_phases = torch.tensor([-1], dtype=torch.long)
            mixed_length = torch.tensor(1, dtype=torch.long)

        return mixed_x, mixed_length, mixed_phases


def collate_fn(batch):
    xs, lengths, ys, phases = zip(*batch, strict=False)
    lengths = torch.stack(lengths)
    max_len = lengths.max().item()
    feat_dim = xs[0].shape[1]

    padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    padded_phases = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 for padding

    for i, (seq, length, phase) in enumerate(zip(xs, lengths, phases, strict=False)):
        padded[i, : length.item()] = seq
        padded_phases[i, : length.item()] = phase

    # Stack labels (support both soft and hard labels)
    if ys[0].dim() > 0 and ys[0].shape[0] > 1:  # soft label (one-hot)
        stacked_ys = torch.stack(ys)
    else:  # hard label (scalar)
        stacked_ys = torch.stack(ys)

    return padded, lengths, stacked_ys, padded_phases


class GestureDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_ds: GestureDataset,
        val_ds: GestureDataset,
        batch_size: int = 128,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )
