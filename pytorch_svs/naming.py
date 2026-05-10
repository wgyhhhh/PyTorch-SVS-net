"""Filename parsing helpers for PNG sequence data.

Expected names:
  image_s40_i0.png, image_s40_i1.png, ...
  label_s40.png
"""

from __future__ import annotations

import re
from pathlib import Path


IMAGE_RE = re.compile(r"^image_(?P<sid>.+)_i(?P<frame>\d+)\.png$", re.IGNORECASE)
LABEL_RE = re.compile(r"^label_(?P<sid>.+)\.png$", re.IGNORECASE)


def parse_image_name(path: str | Path) -> tuple[str, int]:
    name = Path(path).name
    match = IMAGE_RE.match(name)
    if match is None:
        raise ValueError(f"Image filename must look like image_s40_i0.png, got: {name}")
    return match.group("sid"), int(match.group("frame"))


def parse_label_name(path: str | Path) -> str:
    name = Path(path).name
    match = LABEL_RE.match(name)
    if match is None:
        raise ValueError(f"Label filename must look like label_s40.png, got: {name}")
    return match.group("sid")


def parse_frame_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    indices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    return indices


def has_separate_validation_dirs(
    val_images_dir: str | Path | None,
    val_labels_dir: str | Path | None,
) -> bool:
    if val_images_dir and val_labels_dir:
        return True
    if val_images_dir or val_labels_dir:
        raise ValueError("--val-images-dir and --val-labels-dir must be provided together")
    return False


def select_frame_indices(
    available: list[int],
    frame_count: int,
    policy: str = "last",
    explicit_indices: list[int] | None = None,
) -> list[int]:
    frames = sorted(available)
    if explicit_indices is not None:
        missing = sorted(set(explicit_indices) - set(frames))
        if missing:
            raise ValueError(f"Requested frame indices {missing} are missing from {frames}")
        return explicit_indices

    if len(frames) < frame_count:
        raise ValueError(f"Need at least {frame_count} frames, found {len(frames)}: {frames}")

    if policy == "first":
        return frames[:frame_count]
    if policy == "last":
        return frames[-frame_count:]
    if policy == "center":
        start = (len(frames) - frame_count) // 2
        return frames[start : start + frame_count]

    raise ValueError("frame policy must be one of: first, center, last")
