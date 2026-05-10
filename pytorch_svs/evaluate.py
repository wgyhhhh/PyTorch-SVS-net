from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


METRIC_FIELDS = ["sample_id", "dice", "precision", "recall", "f1"]


def summarize_metric_rows(rows: Iterable[dict[str, float | str]]) -> dict[str, float | int]:
    rows = list(rows)
    if not rows:
        return {
            "num_samples": 0,
            "mean_dice": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
        }

    return {
        "num_samples": len(rows),
        "mean_dice": sum(float(row["dice"]) for row in rows) / len(rows),
        "mean_precision": sum(float(row["precision"]) for row in rows) / len(rows),
        "mean_recall": sum(float(row["recall"]) for row in rows) / len(rows),
        "mean_f1": sum(float(row["f1"]) for row in rows) / len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SVS-Net predictions against PNG labels.")
    parser.add_argument("--images-dir", required=True, help="Directory containing image_sXX_iY.png files.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing label_sXX.png files.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--output-dir", default="runs/svs_eval")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to checkpoint image_size or 512.")
    parser.add_argument("--frame-count", type=int, default=None, help="Defaults to checkpoint frame_count or 4.")
    parser.add_argument("--frame-policy", choices=["first", "center", "last"], default=None)
    parser.add_argument("--frame-indices", default=None, help="Explicit comma-separated indices, e.g. 2,3,4,5.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--save-probability", action="store_true")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cuda:1, or cpu.")
    return parser.parse_args()


def save_mask(array, path: Path) -> None:
    import numpy as np
    from PIL import Image

    image = Image.fromarray((array * 255.0).clip(0, 255).astype(np.uint8))
    image.save(path)


def write_metric_files(output_dir: Path, rows: list[dict[str, float | str]], summary: dict[str, float | int]) -> None:
    with (output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from .dataset import SVSPngSequenceDataset, discover_samples
    from .losses import binary_scores
    from .model import SVSNet
    from .naming import parse_frame_indices

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})

    frame_count = args.frame_count or int(checkpoint_args.get("frame_count", 4))
    image_size = args.image_size or int(checkpoint_args.get("image_size", 512))
    frame_policy = args.frame_policy or checkpoint_args.get("frame_policy", "last")
    explicit_indices = parse_frame_indices(args.frame_indices or checkpoint_args.get("frame_indices"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(
        args.images_dir,
        args.labels_dir,
        frame_count=frame_count,
        frame_policy=frame_policy,
        frame_indices=explicit_indices,
    )
    dataset = SVSPngSequenceDataset(
        samples,
        image_size=image_size,
        augment=False,
        mask_threshold=args.mask_threshold,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SVSNet(in_channels=1, frame_count=frame_count).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for images, masks, sample_ids in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            preds = model(images)

            for item_idx, sample_id in enumerate(sample_ids):
                pred = preds[item_idx : item_idx + 1]
                mask = masks[item_idx : item_idx + 1]
                scores = binary_scores(pred, mask, threshold=args.threshold)
                row = {
                    "sample_id": str(sample_id),
                    "dice": scores["dice"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                }
                rows.append(row)

                pred_array = pred[0, 0].detach().cpu().numpy()
                if args.save_probability:
                    save_mask(pred_array, output_dir / f"prob_{sample_id}.png")
                binary_array = (pred_array >= args.threshold).astype("float32")
                save_mask(binary_array, output_dir / f"pred_{sample_id}.png")

                print(
                    f"{sample_id}: "
                    f"dice={row['dice']:.6f} precision={row['precision']:.6f} "
                    f"recall={row['recall']:.6f} f1={row['f1']:.6f}"
                )

    summary = summarize_metric_rows(rows)
    write_metric_files(output_dir, rows, summary)
    print(
        "summary: "
        f"num_samples={summary['num_samples']} "
        f"mean_dice={summary['mean_dice']:.6f} "
        f"mean_precision={summary['mean_precision']:.6f} "
        f"mean_recall={summary['mean_recall']:.6f} "
        f"mean_f1={summary['mean_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
