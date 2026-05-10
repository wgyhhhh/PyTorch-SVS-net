from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .naming import parse_frame_indices, parse_image_name, select_frame_indices
from .model import SVSNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SVS-Net PyTorch inference on PNG sequences.")
    parser.add_argument("--images-dir", required=True, help="Directory containing image_sXX_iY.png files.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--output-dir", default="runs/svs_predictions")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to checkpoint image_size or 512.")
    parser.add_argument("--frame-count", type=int, default=None, help="Defaults to checkpoint frame_count or 4.")
    parser.add_argument("--frame-policy", choices=["first", "center", "last"], default=None)
    parser.add_argument("--frame-indices", default=None, help="Explicit comma-separated indices, e.g. 0,1,2,3.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-probability", action="store_true")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cuda:1, or cpu.")
    return parser.parse_args()


def load_sequence(path_by_frame: dict[int, Path], frame_indices: list[int], image_size: int) -> torch.Tensor:
    arrays = []
    for frame_index in frame_indices:
        image = Image.open(path_by_frame[frame_index]).convert("L")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        max_value = float(array.max())
        if max_value > 0:
            array = array / max_value
        arrays.append(array)
    return torch.from_numpy(np.stack(arrays, axis=0)).float().unsqueeze(0).unsqueeze(0)


def save_mask(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray((array * 255.0).clip(0, 255).astype(np.uint8))
    image.save(path)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})

    frame_count = args.frame_count or int(checkpoint_args.get("frame_count", 4))
    image_size = args.image_size or int(checkpoint_args.get("image_size", 512))
    frame_policy = args.frame_policy or checkpoint_args.get("frame_policy", "last")
    explicit_indices = parse_frame_indices(args.frame_indices or checkpoint_args.get("frame_indices"))

    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(Path(args.images_dir).rglob("*.png")):
        try:
            sample_id, frame_index = parse_image_name(path)
        except ValueError:
            continue
        grouped.setdefault(sample_id, {})[frame_index] = path

    if not grouped:
        raise RuntimeError("No image_sXX_iY.png files found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SVSNet(in_channels=1, frame_count=frame_count).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        for sample_id, path_by_frame in sorted(grouped.items()):
            frame_indices = select_frame_indices(
                list(path_by_frame),
                frame_count=frame_count,
                policy=frame_policy,
                explicit_indices=explicit_indices,
            )
            tensor = load_sequence(path_by_frame, frame_indices, image_size).to(device)
            pred = model(tensor)[0, 0].detach().cpu().numpy()
            if args.save_probability:
                save_mask(pred, output_dir / f"prob_{sample_id}.png")
            binary = (pred >= args.threshold).astype(np.float32)
            save_mask(binary, output_dir / f"pred_{sample_id}.png")
            print(f"saved {sample_id}: frames={frame_indices}")


if __name__ == "__main__":
    main()
