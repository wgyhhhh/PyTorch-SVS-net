from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SVSPngSequenceDataset, discover_samples, split_samples
from .losses import DiceLoss, FocalLoss, NegativeDiceLoss, dice_coefficient
from .model import SVSNet
from .naming import has_separate_validation_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVS-Net with PyTorch on PNG sequences.")
    parser.add_argument("--images-dir", required=True, help="Directory containing image_sXX_iY.png files.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing label_sXX.png files.")
    parser.add_argument("--val-images-dir", default=None, help="Optional validation image directory.")
    parser.add_argument("--val-labels-dir", default=None, help="Optional validation label directory.")
    parser.add_argument("--output-dir", default="runs/svs_pytorch", help="Checkpoint/log output directory.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--frame-policy", choices=["first", "center", "last"], default="last")
    parser.add_argument("--frame-indices", default=None, help="Explicit comma-separated indices, e.g. 0,1,2,3.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=601)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--loss", choices=["dice", "negative-dice", "focal"], default="dice")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cuda:1, or cpu.")
    parser.add_argument("--resume", default=None, help="Path to a PyTorch checkpoint to resume.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(name: str) -> torch.nn.Module:
    if name == "dice":
        return DiceLoss()
    if name == "negative-dice":
        return NegativeDiceLoss()
    if name == "focal":
        return FocalLoss()
    raise ValueError(f"Unsupported loss: {name}")


def run_epoch(
    model: SVSNet,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    dice_sum = 0.0
    n_batches = 0

    for images, masks, _sample_ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                preds = model(images)
                loss = criterion(preds, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        loss_sum += float(loss.detach().cpu())
        dice_sum += float(dice_coefficient(preds.detach(), masks).detach().cpu())
        n_batches += 1

    return loss_sum / max(n_batches, 1), dice_sum / max(n_batches, 1)


def save_checkpoint(
    path: Path,
    model: SVSNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    samples = discover_samples(
        args.images_dir,
        args.labels_dir,
        frame_count=args.frame_count,
        frame_policy=args.frame_policy,
        frame_indices=args.frame_indices,
    )
    if has_separate_validation_dirs(args.val_images_dir, args.val_labels_dir):
        train_samples = samples
        val_samples = discover_samples(
            args.val_images_dir,
            args.val_labels_dir,
            frame_count=args.frame_count,
            frame_policy=args.frame_policy,
            frame_indices=args.frame_indices,
        )
    else:
        train_samples, val_samples = split_samples(samples, val_fraction=args.val_fraction, seed=args.seed)

    if not train_samples:
        raise RuntimeError("Training split is empty. Lower --val-fraction or add more samples.")

    train_dataset = SVSPngSequenceDataset(
        train_samples,
        image_size=args.image_size,
        augment=not args.no_augment,
    )
    val_dataset = SVSPngSequenceDataset(
        val_samples,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SVSNet(in_channels=1, frame_count=args.frame_count).to(device)
    criterion = build_loss(args.loss)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        start_epoch = int(checkpoint["epoch"]) + 1

    log_path = output_dir / "training_log.csv"
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=["epoch", "lr", "train_loss", "train_dice", "val_loss", "val_dice"],
        )
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_dice = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                amp=args.amp,
            )

            if len(val_dataset) > 0:
                val_loss, val_dice = run_epoch(model, val_loader, criterion, device, amp=args.amp)
                monitored_loss = val_loss
            else:
                val_loss, val_dice = float("nan"), float("nan")
                monitored_loss = train_loss

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                }
            )
            log_file.flush()

            scheduler.step()

            if monitored_loss < best_loss:
                best_loss = monitored_loss
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_loss, args)
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_loss, args)

            print(
                f"epoch={epoch:04d} lr={lr:.6g} "
                f"train_loss={train_loss:.6f} train_dice={train_dice:.6f} "
                f"val_loss={val_loss:.6f} val_dice={val_dice:.6f}"
            )


if __name__ == "__main__":
    main()
