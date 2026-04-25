"""
Fine-tune NAFNet + ViT from the best baseline checkpoint.

This script is intentionally separate from train.py so the long baseline run is
not overwritten. Defaults are tuned for an RTX 5070 8GB:

    python fine_tune.py

Outputs:
    weights/nafnet_before_finetune.pth
    weights/nafnet_finetuned.pth
    weights/training_log_finetune.csv
    weights/training_curve_finetune.png
    weights/visual_finetune/epoch_*.png
"""

import argparse
import csv
import os
import shutil

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nafnet_vit import nafnet_vit_small
from train import (
    CLEAR_DIR,
    HAZY_DIR,
    PREPROCESS_DIR,
    TEST_SPLIT,
    VAL_SPLIT,
    WEIGHTS_DIR,
    CombinedLoss,
    UnderwaterDataset,
    get_pairs,
    plot_curves,
    psnr,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune NAFNet + ViT from weights/nafnet_final.pth")
    parser.add_argument("--resume", default=os.path.join(WEIGHTS_DIR, "nafnet_final.pth"),
                        help="checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=45,
                        help="fine-tune epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="batch size; use 4 if 384px runs out of VRAM")
    parser.add_argument("--img-size", type=int, default=256,
                        help="training crop size; keep 256 first on 8GB VRAM")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="fine-tune learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="cosine scheduler minimum LR")
    parser.add_argument("--preproc-prob", type=float, default=0.8,
                        help="probability of training on preprocessed input")
    parser.add_argument("--val-preproc-prob", type=float, default=1.0,
                        help="validation input probability; 1.0 matches inference")
    parser.add_argument("--patience", type=int, default=15,
                        help="early-stop patience on 3-epoch validation PSNR average")
    parser.add_argument("--save-every", type=int, default=5,
                        help="save numbered checkpoint every N epochs")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--visual-every", type=int, default=5,
                        help="save validation grid every N epochs")
    parser.add_argument("--visual-count", type=int, default=6,
                        help="number of validation samples in each grid")
    parser.add_argument("--output-name", default="nafnet_finetuned.pth",
                        help="best fine-tuned checkpoint filename in weights/")
    parser.add_argument("--log-name", default="training_log_finetune.csv",
                        help="fine-tune CSV filename in weights/")
    parser.add_argument("--curve-name", default="training_curve_finetune.png",
                        help="fine-tune curve filename in weights/")
    parser.add_argument("--visual-dir", default=os.path.join(WEIGHTS_DIR, "visual_finetune"),
                        help="directory for visual validation grids")
    parser.add_argument("--promote-final", action="store_true",
                        help="copy best fine-tuned checkpoint to weights/nafnet_final.pth at the end")
    return parser.parse_args()


def build_splits():
    hazy_pairs = get_pairs(HAZY_DIR, CLEAR_DIR)
    if not hazy_pairs:
        raise RuntimeError(f"No paired images found in {HAZY_DIR} / {CLEAR_DIR}")

    has_preproc = os.path.isdir(PREPROCESS_DIR) and len(os.listdir(PREPROCESS_DIR)) > 0
    preproc_pairs = get_pairs(PREPROCESS_DIR, CLEAR_DIR) if has_preproc else None

    if preproc_pairs:
        preproc_names = {os.path.basename(p[0]): p for p in preproc_pairs}
        aligned_hazy = []
        aligned_preproc = []
        for hp, cp in hazy_pairs:
            name = os.path.basename(hp)
            if name in preproc_names:
                aligned_hazy.append((hp, cp))
                aligned_preproc.append(preproc_names[name])
        hazy_pairs = aligned_hazy
        preproc_pairs = aligned_preproc
        print(f"Aligned pairs   : {len(hazy_pairs)} (hazy + preprocessed)")
    else:
        print("Input mode      : raw hazy only (no preprocessed folder found)")

    np.random.seed(42)
    idx = np.random.permutation(len(hazy_pairs)).tolist()
    hazy_pairs = [hazy_pairs[i] for i in idx]
    if preproc_pairs:
        preproc_pairs = [preproc_pairs[i] for i in idx]

    n_test = max(1, int(len(hazy_pairs) * TEST_SPLIT))
    n_val = max(1, int(len(hazy_pairs) * VAL_SPLIT))

    test_hazy = hazy_pairs[:n_test]
    val_hazy = hazy_pairs[n_test:n_test + n_val]
    train_hazy = hazy_pairs[n_test + n_val:]

    if preproc_pairs:
        val_preproc = preproc_pairs[n_test:n_test + n_val]
        train_preproc = preproc_pairs[n_test + n_val:]
    else:
        val_preproc = None
        train_preproc = None

    print(f"Total pairs     : {len(hazy_pairs)}")
    print(f"Split           : train={len(train_hazy)}  val={len(val_hazy)}  test={len(test_hazy)}")
    return train_hazy, val_hazy, train_preproc, val_preproc


def autocast_context(device, amp_dtype):
    return torch.amp.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    )


def evaluate(model, val_dl, device, amp_dtype):
    model.eval()
    val_psnr = 0.0
    with torch.no_grad(), autocast_context(device, amp_dtype):
        for inp_b, clear_b in val_dl:
            inp_b = inp_b.to(device, non_blocking=True)
            clear_b = clear_b.to(device, non_blocking=True)
            val_psnr += psnr(model(inp_b), clear_b)
    return val_psnr / len(val_dl)


def tensor_to_uint8(x):
    arr = x.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_validation_grid(model, dataset, device, amp_dtype, epoch, save_dir, count):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    rows = []
    sample_count = min(count, len(dataset))
    sep_w = 6

    with torch.no_grad(), autocast_context(device, amp_dtype):
        for i in range(sample_count):
            inp, target = dataset[i]
            pred = model(inp.unsqueeze(0).to(device)).squeeze(0)

            inp_img = tensor_to_uint8(inp)
            pred_img = tensor_to_uint8(pred)
            target_img = tensor_to_uint8(target)

            h = inp_img.shape[0]
            sep = np.full((h, sep_w, 3), 255, dtype=np.uint8)
            row = np.concatenate([inp_img, sep, pred_img, sep, target_img], axis=1)
            rows.append(row)

    if not rows:
        return

    sep_h = np.full((sep_w, rows[0].shape[1], 3), 255, dtype=np.uint8)
    grid = rows[0]
    for row in rows[1:]:
        grid = np.concatenate([grid, sep_h, row], axis=0)

    path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  visual grid -> {path}")


def main():
    args = parse_args()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    persistent_workers = args.workers > 0
    amp_dtype = torch.bfloat16

    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Missing resume checkpoint: {args.resume}")

    baseline_backup = os.path.join(WEIGHTS_DIR, "nafnet_before_finetune.pth")
    if not os.path.exists(baseline_backup):
        shutil.copy2(args.resume, baseline_backup)
        print(f"Baseline backup -> {baseline_backup}")
    else:
        print(f"Baseline backup exists -> {baseline_backup}")

    output_path = os.path.join(WEIGHTS_DIR, args.output_name)
    log_path = os.path.join(WEIGHTS_DIR, args.log_name)
    curve_path = os.path.join(WEIGHTS_DIR, args.curve_name)

    train_hazy, val_hazy, train_preproc, val_preproc = build_splits()
    hp_tr, cp_tr = zip(*train_hazy)
    hp_va, cp_va = zip(*val_hazy)
    pp_tr = [p[0] for p in train_preproc] if train_preproc else None
    pp_va = [p[0] for p in val_preproc] if val_preproc else None

    print(f"Train input     : {int(args.preproc_prob * 100)}% preprocessed + "
          f"{int((1.0 - args.preproc_prob) * 100)}% raw hazy")
    print(f"Val input       : {int(args.val_preproc_prob * 100)}% preprocessed")

    train_ds = UnderwaterDataset(
        hp_tr, cp_tr, args.img_size, augment=True,
        preproc_paths=pp_tr, preproc_prob=args.preproc_prob)
    val_ds = UnderwaterDataset(
        hp_va, cp_va, args.img_size, augment=False,
        preproc_paths=pp_va, preproc_prob=args.val_preproc_prob if pp_va else 0.0)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers)
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    model = nafnet_vit_small().to(device)
    state = torch.load(args.resume, map_location=device)
    model.load_state_dict(state, strict=True)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = CombinedLoss(
        w_l1=0.45,
        w_ssim=0.35,
        w_grad=0.10,
        w_color=0.10,
        w_percep=0.0,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)

    print(f"Parameters      : {total:,}")
    print(f"Device          : {device}")
    print(f"Fine-tune epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"Loss            : 0.45 L1 + 0.35 SSIM + 0.10 edge + 0.10 color")
    print(f"Output best     : {output_path}")
    print(f"Log             : {log_path}")

    baseline_psnr = evaluate(model, val_dl, device, amp_dtype)
    best_psnr = baseline_psnr
    best_epoch = 0
    epochs_no_impv = 0
    stopped_at = None
    hist_epochs = [0]
    hist_loss = [0.0]
    hist_psnr = [baseline_psnr]
    torch.save(model.state_dict(), output_path)
    save_validation_grid(model, val_ds, device, amp_dtype, 0, args.visual_dir, args.visual_count)
    print(f"Baseline val PSNR: {baseline_psnr:.4f} dB")

    with open(log_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_psnr", "avg3", "lr"])
        csv_writer.writerow([0, "", round(baseline_psnr, 4), round(baseline_psnr, 4), args.lr])
        csv_file.flush()

        try:
            for epoch in range(1, args.epochs + 1):
                model.train()
                train_loss = 0.0
                pbar = tqdm(train_dl, desc=f"FineTune [{epoch:>3}/{args.epochs}]", leave=False)
                for inp_b, clear_b in pbar:
                    inp_b = inp_b.to(device, non_blocking=True)
                    clear_b = clear_b.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    with autocast_context(device, amp_dtype):
                        loss = criterion(model(inp_b), clear_b)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                train_loss /= len(train_dl)
                val_psnr = evaluate(model, val_dl, device, amp_dtype)
                hist_epochs.append(epoch)
                hist_loss.append(train_loss)
                hist_psnr.append(val_psnr)
                avg_psnr = float(np.mean(hist_psnr[-3:]))
                scheduler.step()
                lr_now = optimizer.param_groups[0]["lr"]

                print(f"FineTune [{epoch:>3}/{args.epochs}]  loss={train_loss:.4f}  "
                      f"val_psnr={val_psnr:.2f} dB  avg3={avg_psnr:.2f}  "
                      f"lr={lr_now:.2e}")

                csv_writer.writerow([
                    epoch,
                    round(train_loss, 6),
                    round(val_psnr, 4),
                    round(avg_psnr, 4),
                    lr_now,
                ])
                csv_file.flush()

                if epoch % args.visual_every == 0:
                    save_validation_grid(
                        model, val_ds, device, amp_dtype, epoch,
                        args.visual_dir, args.visual_count)

                if epoch % args.save_every == 0:
                    ckpt = os.path.join(WEIGHTS_DIR, f"nafnet_finetune_epoch_{epoch}.pth")
                    torch.save(model.state_dict(), ckpt)
                    print(f"  checkpoint -> {ckpt}")

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_epoch = epoch
                    epochs_no_impv = 0
                    torch.save(model.state_dict(), output_path)
                    print(f"  best fine-tune -> {output_path}")
                else:
                    epochs_no_impv += 1
                    print(f"  no improvement ({epochs_no_impv}/{args.patience})  "
                          f"[avg_psnr={avg_psnr:.2f}]")
                    if epochs_no_impv >= args.patience:
                        stopped_at = epoch
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break
        except KeyboardInterrupt:
            stopped_at = hist_epochs[-1]
            print("\nInterrupted. Keeping the best fine-tuned checkpoint saved so far.")

    plot_curves(hist_epochs, hist_loss, hist_psnr, stopped_at, curve_path)

    if args.promote_final:
        final_path = os.path.join(WEIGHTS_DIR, "nafnet_final.pth")
        shutil.copy2(output_path, final_path)
        print(f"Promoted fine-tuned checkpoint -> {final_path}")

    print("Fine-tune complete.")
    print(f"Baseline val PSNR : {baseline_psnr:.4f} dB")
    print(f"Best fine-tune    : {best_psnr:.4f} dB @ epoch {best_epoch}")
    print(f"Weights           : {output_path}")
    print(f"Visual grids      : {args.visual_dir}")


if __name__ == "__main__":
    main()
