"""
stage3_train_model.py — AI Model Training (U-Net Semantic Segmentation)
Trains a U-Net with ResNet34 backbone to classify pixels as Forest / Deforested.

Run: python stage3_train_model.py
"""

import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
from tqdm import tqdm

from config import (
    DATA_TILES, MODEL_PATH, TILE_SIZE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    IN_CHANNELS, NUM_CLASSES, ENCODER_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── Dataset ─────────────────────────────────────────────────────

class ForestDataset(Dataset):
    """
    Loads paired (image tile, label mask) arrays from disk.
    image: (4, H, W) float32  [R, G, B, NIR] normalized 0-1
    label: (H, W)   uint8     0=Forest, 1=Deforested
    """

    def __init__(self, tiles_dir: Path, augment: bool = False):
        self.label_dir = tiles_dir / "labels"
        self.tile_paths = sorted(tiles_dir.glob("*.npy"))
        self.augment = augment

        # Only keep tiles that have a matching label
        self.tile_paths = [
            p for p in self.tile_paths
            if (self.label_dir / p.name).exists()
        ]

        if not self.tile_paths:
            raise FileNotFoundError(
                f"No paired tiles found in {tiles_dir}\n"
                "Run stage2_preprocess.py first."
            )

        log.info(f"Dataset loaded: {len(self.tile_paths)} tile pairs")

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path  = self.tile_paths[idx]
        label_path = self.label_dir / tile_path.name

        # Load image: (H, W, 4) → (4, H, W)
        image = np.load(tile_path).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))  # CHW format

        # Load label: (H, W)
        label = np.load(label_path).astype(np.int64)

        # Basic augmentation (flip)
        if self.augment and np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        if self.augment and np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()

        return torch.tensor(image), torch.tensor(label)


# ─── Metrics ─────────────────────────────────────────────────────

def iou_score(pred: torch.Tensor, target: torch.Tensor, n_classes: int = 2) -> float:
    """Mean Intersection over Union across classes."""
    ious = []
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    for cls in range(n_classes):
        p = pred_flat == cls
        t = target_flat == cls
        intersection = (p & t).sum().float()
        union = (p | t).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return sum(ious) / len(ious) if ious else 0.0


# ─── Training ─────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if device.type == "cpu":
        log.info("  (GPU not found — training on CPU, will be slower)")

    # ── Dataset & Loader ──
    full_dataset = ForestDataset(DATA_TILES, augment=True)
    val_size  = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    log.info(f"Train: {train_size} tiles | Val: {val_size} tiles")

    # ── Model ──
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet",
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: U-Net + {ENCODER_NAME} | Params: {total_params:,}")

    # ── Loss, Optimizer, Scheduler ──
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0]).to(device)  # Upweight deforested class
    )
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training Loop ──
    best_val_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    log.info("=" * 60)
    log.info(f"Starting training for {EPOCHS} epochs ...")
    log.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)                    # (B, 2, H, W)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_iou += iou_score(preds.cpu(), labels.cpu())

        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        log.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_iou:.4f}"
        )

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), MODEL_PATH)
            log.info(f"  ⭐ New best model saved (mIoU={val_iou:.4f})")

    # ── Save training history ──
    import json
    hist_path = MODEL_PATH.parent / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    log.info("=" * 60)
    log.info(f"✅ Training complete. Best Val mIoU: {best_val_iou:.4f}")
    log.info(f"   Model saved: {MODEL_PATH}")
    log.info("   Next: Run python stage4_inference.py")
    log.info("=" * 60)


def main():
    log.info("=" * 60)
    log.info("STAGE 3: U-Net Model Training")
    log.info("=" * 60)

    if MODEL_PATH.exists():
        ans = input(f"Model already exists at {MODEL_PATH}. Retrain? [y/N]: ").strip().lower()
        if ans != "y":
            log.info("Skipping training. Using existing model.")
            return

    train()


if __name__ == "__main__":
    main()
