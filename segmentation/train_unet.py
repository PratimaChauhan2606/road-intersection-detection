"""
Step 2 of segmentation: train U-Net (ResNet-34 encoder) on DeepGlobe.

- Training: random 512x512 crops + flips/rotations, BCE + Dice loss,
  AdamW, cosine learning rate, mixed precision (AMP) on the GPU.
- Validation after every epoch on full 1024x1024 validation images:
  IoU, precision, recall, F1 at threshold 0.5.
- Saves last.pt (every epoch, for resuming), best.pt (highest val IoU),
  history.csv and config.json in outputs/segmentation/<run-name>/.

Quick test (2-3 minutes):
    python segmentation/train_unet.py --data-dir "C:/datasets/deepglobe" --run-name smoke --epochs 1 --samples-per-epoch 200 --val-limit 20
Full run:
    python segmentation/train_unet.py --data-dir "C:/datasets/deepglobe"
Resume after an interruption:
    python segmentation/train_unet.py --data-dir "C:/datasets/deepglobe" --resume
"""
import argparse
import csv
import json
import os
import time

import cv2
import torch
from torch.utils.data import DataLoader, RandomSampler

from common import (RoadDataset, build_model, confusion_counts, find_train_dir,
                    metrics_from_counts, read_ids, seed_everything)


def worker_init(_):
    cv2.setNumThreads(0)  # avoid CPU over-subscription inside DataLoader workers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits-dir", default="data/segmentation/splits")
    ap.add_argument("--out", default="outputs/segmentation")
    ap.add_argument("--run-name", default="unet_r34_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = one crop per training image")
    ap.add_argument("--val-limit", type=int, default=200, help="Validate on the first N val images (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type != "cuda":
        print("WARNING: CUDA not available - training on CPU will be very slow.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_dir = os.path.join(args.out, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    image_dir = find_train_dir(args.data_dir)
    train_ids = read_ids(os.path.join(args.splits_dir, "train.txt"))
    val_ids = read_ids(os.path.join(args.splits_dir, "val.txt"))
    if args.val_limit:
        val_ids = val_ids[: args.val_limit]

    train_ds = RoadDataset(image_dir, train_ids, crop=args.crop, augment=True)
    val_ds = RoadDataset(image_dir, val_ids, crop=None, augment=False)
    n_samples = args.samples_per_epoch or len(train_ds)
    sampler = RandomSampler(train_ds, replacement=False, num_samples=n_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                          pin_memory=use_amp, drop_last=True, worker_init_fn=worker_init,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(2, args.workers),
                        pin_memory=use_amp, worker_init_fn=worker_init)
    print(f"Train images {len(train_ids)} | samples/epoch {n_samples} | val images {len(val_ids)}")

    model = build_model(pretrained=not args.no_pretrained).to(device)
    import segmentation_models_pytorch as smp
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_iou = 1, -1.0
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")
    hist_path = os.path.join(run_dir, "history.csv")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_iou = ckpt["epoch"] + 1, ckpt["best_iou"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val IoU so far {best_iou:.4f})")
    elif not args.resume:
        with open(hist_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_iou", "val_precision", "val_recall", "val_f1",
                                    "lr", "epoch_minutes", "peak_gpu_mem_mb"])

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "image_dir": image_dir, "device": str(device),
                   "torch": torch.__version__}, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if use_amp:
            torch.cuda.reset_peak_memory_stats()

        # ---------------- train
        model.train()
        running, steps = 0.0, 0
        for i, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            loss = bce(logits, y) + dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            steps += 1
            if i % 50 == 0 or i == len(train_dl):
                el = time.time() - t0
                eta = el / i * (len(train_dl) - i)
                print(f"  epoch {epoch} step {i}/{len(train_dl)} loss {running/steps:.4f} "
                      f"| {el/60:.1f} min elapsed, ~{eta/60:.1f} min left in epoch", flush=True)
        scheduler.step()

        # ---------------- validate
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    probs = torch.sigmoid(model(x).float())
                a, b, c = confusion_counts(probs, y, 0.5)
                tp, fp, fn = tp + a, fp + b, fn + c
        m = metrics_from_counts(tp, fp, fn)
        minutes = (time.time() - t0) / 60
        peak = torch.cuda.max_memory_allocated() / 2**20 if use_amp else 0
        lr_now = optimizer.param_groups[0]["lr"]

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{running/max(steps,1):.5f}", f"{m['iou']:.5f}", f"{m['precision']:.5f}",
                                    f"{m['recall']:.5f}", f"{m['f1']:.5f}", f"{lr_now:.2e}", f"{minutes:.2f}", f"{peak:.0f}"])

        is_best = m["iou"] > best_iou
        if is_best:
            best_iou = m["iou"]
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_iou": best_iou, "args": vars(args)}
        torch.save(state, last_path)
        if is_best:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": m, "args": vars(args)}, best_path)

        print(f"Epoch {epoch}/{args.epochs} | loss {running/max(steps,1):.4f} | val IoU {m['iou']:.4f} "
              f"P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} | {minutes:.1f} min | "
              f"peak GPU {peak:.0f} MB {'| NEW BEST' if is_best else ''}", flush=True)

    print(f"Done. Best val IoU {best_iou:.4f}. Files in {run_dir}")


if __name__ == "__main__":
    main()
