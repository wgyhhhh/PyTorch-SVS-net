"""PNG sequence dataset for SVS-Net.

Each sample is formed by matching files like:
  images/image_s40_i0.png ... images/image_s40_i5.png
  labels/label_s40.png

The model uses a fixed number of frames. If more frames exist, choose them with
the frame policy or explicit frame indices.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .naming import parse_frame_indices, parse_image_name, parse_label_name, select_frame_indices


@dataclass(frozen=True)
class SequenceSample:
    sample_id: str
    image_paths: tuple[Path, ...]
    label_path: Path


def discover_samples(
    images_dir: str | Path,
    labels_dir: str | Path,
    frame_count: int = 4,
    frame_policy: str = "last",
    frame_indices: str | Sequence[int] | None = None,
) -> list[SequenceSample]:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if isinstance(frame_indices, str):
        explicit_indices = parse_frame_indices(frame_indices)
    elif frame_indices is None:
        explicit_indices = None
    else:
        explicit_indices = list(frame_indices)

    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(images_dir.rglob("*.png")):
        try:
            sample_id, frame_index = parse_image_name(path)
        except ValueError:
            continue
        grouped.setdefault(sample_id, {})[frame_index] = path

    labels: dict[str, Path] = {}
    for path in sorted(labels_dir.rglob("*.png")):
        try:
            labels[parse_label_name(path)] = path
        except ValueError:
            continue

    samples: list[SequenceSample] = []
    for sample_id in sorted(grouped):
        if sample_id not in labels:
            continue
        selected = select_frame_indices(
            list(grouped[sample_id]),
            frame_count=frame_count,
            policy=frame_policy,
            explicit_indices=explicit_indices,
        )
        image_paths = tuple(grouped[sample_id][idx] for idx in selected)
        samples.append(SequenceSample(sample_id, image_paths, labels[sample_id]))

    if not samples:
        raise RuntimeError(
            "No PNG sequence samples found. Expected image_s40_i0.png and label_s40.png style names."
        )
    return samples


def split_samples(
    samples: Sequence[SequenceSample],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[SequenceSample], list[SequenceSample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = int(round(len(samples) * val_fraction))
    if len(samples) > 1 and n_val == 0 and val_fraction > 0:
        n_val = 1
    val_ids = set(indices[:n_val])
    train = [sample for idx, sample in enumerate(samples) if idx not in val_ids]
    val = [sample for idx, sample in enumerate(samples) if idx in val_ids]
    return train, val


class SVSPngSequenceDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SequenceSample],
        image_size: int = 512,
        augment: bool = False,
        mask_threshold: float = 0.5,
        intensity_shift: float = 0.2,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.mask_threshold = mask_threshold
        self.intensity_shift = intensity_shift

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        frames = [self._load_grayscale(path, resample=Image.BILINEAR) for path in sample.image_paths]
        mask = self._load_grayscale(sample.label_path, resample=Image.NEAREST)

        if self.augment:
            frames, mask = self._augment(frames, mask)

        frame_arrays = [self._to_float_array(frame) for frame in frames]
        if self.augment and self.intensity_shift > 0:
            shift = random.uniform(-self.intensity_shift, self.intensity_shift)
            frame_arrays = [np.clip(array + shift, 0.0, 1.0) for array in frame_arrays]

        mask_array = self._to_float_array(mask)
        mask_array = (mask_array > self.mask_threshold).astype(np.float32)

        image_tensor = torch.from_numpy(np.stack(frame_arrays, axis=0)).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)
        return image_tensor, mask_tensor, sample.sample_id

    def _load_grayscale(self, path: Path, resample: int) -> Image.Image:
        image = Image.open(path).convert("L")
        if self.image_size is not None and image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), resample=resample)
        return image

    @staticmethod
    def _to_float_array(image: Image.Image) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32)
        max_value = float(array.max())
        if max_value > 0:
            array = array / max_value
        return array

    def _augment(self, frames: list[Image.Image], mask: Image.Image) -> tuple[list[Image.Image], Image.Image]:
        if random.random() > 0.5:
            frames = [ImageOps.mirror(frame) for frame in frames]
            mask = ImageOps.mirror(mask)

        if random.random() > 0.5:
            frames = [ImageOps.flip(frame) for frame in frames]
            mask = ImageOps.flip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-10.0, 10.0)
            frames = [frame.rotate(angle, resample=Image.BILINEAR) for frame in frames]
            mask = mask.rotate(angle, resample=Image.NEAREST)

        if random.random() > 0.5:
            crop_size = int(self.image_size * random.uniform(0.78, 1.0))
            left = random.randint(0, self.image_size - crop_size)
            top = random.randint(0, self.image_size - crop_size)
            box = (left, top, left + crop_size, top + crop_size)
            frames = [
                frame.crop(box).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
                for frame in frames
            ]
            mask = mask.crop(box).resize((self.image_size, self.image_size), resample=Image.NEAREST)

        return frames, mask
