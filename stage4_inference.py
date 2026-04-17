"""
stage4_inference.py — AI Inference on New Satellite Tiles
Loads trained U-Net, runs inference on all tiles in data/tiles/,
saves binary forest/deforested masks.

Run: python stage4_inference.py
"""

import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

from config import (
    DATA_TILES, DATA_PROCESSED, MODEL_PATH,
    IN_CHANNELS, NUM_CLASSES, ENCODER_NAME,
    TILE_SIZE, PIXEL_AREA_HA
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MASKS_DIR = DATA_PROCESSED / "masks"
MASKS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(device: torch.device) -> torch.nn.Module:
    """Load trained U-Net from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Please run stage3_train_model.py first."
        )

    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    log.info(f"Model loaded from {MODEL_PATH}")
    return model


def predict_tile(model: torch.nn.Module, tile: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run inference on a single (H, W, 4) tile.
    Returns binary mask (H, W): 0=Forest, 1=Deforested
    """
    # (H, W, 4) → (1, 4, H, W)
    x = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)                     # (1, 2, H, W)
        probs  = torch.softmax(logits, dim=1) # (1, 2, H, W)
        mask   = torch.argmax(probs, dim=1)   # (1, H, W)

    return mask.squeeze().cpu().numpy().astype(np.uint8)


def run_inference(model: torch.nn.Module, device: torch.device) -> list:
    """Run model on all tiles, save masks, return per-tile stats."""
    tile_paths = sorted(DATA_TILES.glob("*.npy"))

    if not tile_paths:
        raise FileNotFoundError(
            f"No tiles found in {DATA_TILES}\n"
            "Run stage2_preprocess.py first."
        )

    log.info(f"Running inference on {len(tile_paths)} tiles ...")

    results = []
    total_deforested_ha = 0.0
    total_area_ha = 0.0

    for tile_path in tqdm(tile_paths, desc="Inference"):
        tile = np.load(tile_path).astype(np.float32)

        # Ensure tile is (H, W, 4)
        if tile.ndim == 2:
            log.warning(f"  Skipping 2D tile: {tile_path.name}")
            continue
        if tile.shape[-1] != 4:
            log.warning(f"  Skipping tile with {tile.shape[-1]} channels: {tile_path.name}")
            continue

        mask = predict_tile(model, tile, device)

        # Calculate statistics
        total_pixels      = mask.size
        deforested_pixels = int(mask.sum())
        forest_pixels     = total_pixels - deforested_pixels
        deforested_ha     = deforested_pixels * PIXEL_AREA_HA
        total_ha          = total_pixels * PIXEL_AREA_HA

        total_deforested_ha += deforested_ha
        total_area_ha       += total_ha

        # Save mask
        mask_path = MASKS_DIR / tile_path.name
        np.save(mask_path, mask)

        results.append({
            "tile":              tile_path.name,
            "total_pixels":      total_pixels,
            "deforested_pixels": deforested_pixels,
            "forest_pixels":     forest_pixels,
            "deforested_ha":     round(deforested_ha, 2),
            "total_ha":          round(total_ha, 2),
            "deforestation_pct": round(100.0 * deforested_pixels / total_pixels, 2),
        })

    log.info(f"\nInference Summary:")
    log.info(f"  Tiles processed:    {len(results)}")
    log.info(f"  Total area:         {total_area_ha:,.1f} ha")
    log.info(f"  Deforested:         {total_deforested_ha:,.1f} ha")
    log.info(f"  Deforestation rate: {100*total_deforested_ha/max(total_area_ha,1):.2f}%")

    return results


def save_results(results: list):
    """Save inference results to CSV."""
    import csv
    out_path = DATA_PROCESSED / "inference_results.csv"
    fieldnames = ["tile", "total_pixels", "deforested_pixels", "forest_pixels",
                  "deforested_ha", "total_ha", "deforestation_pct"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log.info(f"Results saved → {out_path}")
    return out_path


def main():
    log.info("=" * 60)
    log.info("STAGE 4: AI Inference")
    log.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model   = load_model(device)
    results = run_inference(model, device)
    save_results(results)

    # Flag high-deforestation tiles
    critical = [r for r in results if r["deforestation_pct"] > 30]
    if critical:
        log.warning(f"\n⚠️  {len(critical)} tiles with >30% deforestation detected:")
        for r in critical[:5]:
            log.warning(f"  {r['tile']}: {r['deforestation_pct']}% ({r['deforested_ha']} ha)")

    log.info("=" * 60)
    log.info(f"✅ Stage 4 complete. Masks in: {MASKS_DIR}")
    log.info("   Next: Run python stage5_change_detect.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
