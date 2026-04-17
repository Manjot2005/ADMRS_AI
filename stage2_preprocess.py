"""
stage2_preprocess.py — Pre-processing & Feature Engineering
- Extracts B04 (Red), B03 (Green), B02 (Blue), B08 (NIR) bands
- Computes NDVI
- Tiles large images into 256x256 patches for model input
- Saves processed tiles as numpy arrays

Run: python stage2_preprocess.py
"""

import os
import glob
import logging
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm import tqdm

from config import (
    DATA_RAW, DATA_TILES, DATA_PROCESSED, TILE_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── Band extraction ─────────────────────────────────────────────

def extract_bands_from_safe(safe_dir: Path) -> dict:
    """
    Extract R, G, B, NIR bands from a Sentinel-2 SAFE directory.
    Sentinel-2 L2A stores bands as separate GeoTIFF files inside the SAFE folder.
    Returns dict of {band_name: file_path}
    """
    band_map = {"B02": None, "B03": None, "B04": None, "B08": None}

    # Search for 10m resolution band files
    for band in band_map:
        pattern = str(safe_dir / "**" / f"*_{band}_10m.jp2")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            band_map[band] = Path(matches[0])
        else:
            # Try without resolution suffix
            pattern2 = str(safe_dir / "**" / f"*{band}*.jp2")
            matches2 = glob.glob(pattern2, recursive=True)
            if matches2:
                band_map[band] = Path(matches2[0])

    missing = [b for b, p in band_map.items() if p is None]
    if missing:
        log.warning(f"Missing bands {missing} in {safe_dir.name}")

    return band_map


def read_band(path: Path, target_shape: tuple = None) -> np.ndarray:
    """Read a single band, optionally resample to target shape."""
    with rasterio.open(path) as src:
        if target_shape:
            data = src.read(
                1,
                out_shape=(target_shape[0], target_shape[1]),
                resampling=Resampling.bilinear,
            )
        else:
            data = src.read(1)
        return data.astype(np.float32)


# ─── NDVI Computation ─────────────────────────────────────────────

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)
    Values: -1 (water/bare) to +1 (dense vegetation)
    Threshold > 0.3 generally indicates vegetation
    """
    denominator = nir + red
    ndvi = np.where(denominator == 0, 0.0, (nir - red) / denominator)
    return ndvi.clip(-1, 1)


def save_ndvi(ndvi: np.ndarray, ref_path: Path, out_path: Path):
    """Save NDVI as GeoTIFF preserving geospatial metadata."""
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()

    profile.update(dtype=rasterio.float32, count=1, nodata=-9999)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ndvi.astype(np.float32), 1)

    log.info(f"  NDVI saved → {out_path.name} | range [{ndvi.min():.3f}, {ndvi.max():.3f}]")


# ─── Tiling ───────────────────────────────────────────────────────

def normalize_band(arr: np.ndarray, percentile: int = 98) -> np.ndarray:
    """Normalize to [0, 1] using percentile clipping to handle outliers."""
    low, high = np.percentile(arr, [2, percentile])
    if high == low:
        return np.zeros_like(arr)
    return ((arr - low) / (high - low)).clip(0, 1)


def create_tiles(image_stack: np.ndarray, tile_id: str) -> int:
    """
    Slice (H, W, C) image into TILE_SIZE x TILE_SIZE patches.
    Saves each as {tile_id}_row{r}_col{c}.npy
    Returns count of saved tiles.
    """
    H, W, C = image_stack.shape
    count = 0

    for row in range(0, H - TILE_SIZE + 1, TILE_SIZE):
        for col in range(0, W - TILE_SIZE + 1, TILE_SIZE):
            tile = image_stack[row:row+TILE_SIZE, col:col+TILE_SIZE, :]
            # Skip tiles that are mostly no-data
            if np.isnan(tile).mean() > 0.3:
                continue
            out_path = DATA_TILES / f"{tile_id}_r{row}_c{col}.npy"
            np.save(out_path, tile.astype(np.float32))
            count += 1

    return count


# ─── Main Processing Loop ─────────────────────────────────────────

def process_safe_directory(safe_dir: Path) -> bool:
    """Full preprocessing pipeline for one SAFE directory."""
    tile_id = safe_dir.stem[:40]  # Use first 40 chars of filename as ID
    log.info(f"\nProcessing: {safe_dir.name}")

    # 1. Extract band paths
    bands = extract_bands_from_safe(safe_dir)
    if any(v is None for v in bands.values()):
        log.warning(f"  Skipping — incomplete bands")
        return False

    # 2. Read bands (align all to B04's shape)
    ref_path = bands["B04"]
    with rasterio.open(ref_path) as src:
        H, W = src.height, src.width

    log.info(f"  Image size: {H} x {W} pixels")

    red = read_band(bands["B04"])
    nir = read_band(bands["B08"], target_shape=(H, W))
    grn = read_band(bands["B03"], target_shape=(H, W))
    blu = read_band(bands["B02"], target_shape=(H, W))

    # 3. Compute NDVI
    ndvi = compute_ndvi(red, nir)
    ndvi_out = DATA_PROCESSED / f"{tile_id}_ndvi.tif"
    save_ndvi(ndvi, ref_path, ndvi_out)

    # 4. Normalize and stack into (H, W, 4) array: [R, G, B, NIR]
    stack = np.stack([
        normalize_band(red),
        normalize_band(grn),
        normalize_band(blu),
        normalize_band(nir),
    ], axis=-1)

    # 5. Tile the image
    n_tiles = create_tiles(stack, tile_id)
    log.info(f"  Created {n_tiles} tiles → {DATA_TILES}")

    return True


def unzip_downloads():
    """Unzip any .zip files in data/raw/sentinel2/"""
    zip_dir = DATA_RAW / "sentinel2"
    zips = list(zip_dir.glob("*.zip")) if zip_dir.exists() else []

    if not zips:
        log.info("No zip files to extract. Assuming SAFE dirs already extracted.")
        return

    log.info(f"Extracting {len(zips)} zip files...")
    for z in tqdm(zips, desc="Unzipping"):
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(zip_dir)
        log.info(f"  Extracted: {z.name}")


def main():
    log.info("=" * 60)
    log.info("STAGE 2: Pre-processing & Feature Engineering")
    log.info("=" * 60)

    # Step 1: Unzip downloaded tiles
    unzip_downloads()

    # Step 2: Find all SAFE directories
    s2_dir = DATA_RAW / "sentinel2"
    safe_dirs = sorted(s2_dir.glob("*.SAFE")) if s2_dir.exists() else []

    if not safe_dirs:
        log.warning("No SAFE directories found in data/raw/sentinel2/")
        log.warning("Please run stage1_download.py first, or place .SAFE dirs manually.")
        log.info("\n[DEMO MODE] Creating synthetic tiles for testing...")
        create_synthetic_demo_tiles()
        return

    log.info(f"Found {len(safe_dirs)} SAFE directories")

    # Step 3: Process each
    success, failed = 0, 0
    for safe_dir in safe_dirs:
        if process_safe_directory(safe_dir):
            success += 1
        else:
            failed += 1

    log.info("=" * 60)
    log.info(f"✅ Stage 2 complete. Processed: {success}, Failed: {failed}")
    log.info(f"   Tiles in: {DATA_TILES}")
    log.info(f"   NDVI maps in: {DATA_PROCESSED}")
    log.info("   Next: Run python stage3_train_model.py")
    log.info("=" * 60)


def create_synthetic_demo_tiles():
    """
    Create synthetic demo tiles for testing stages 3-5
    without needing real satellite downloads.
    Each tile is (256, 256, 4): [R, G, B, NIR] float32 [0, 1]
    """
    rng = np.random.default_rng(42)
    n_tiles = 50

    log.info(f"Creating {n_tiles} synthetic demo tiles in {DATA_TILES} ...")
    for i in tqdm(range(n_tiles), desc="Generating"):
        # Simulate forest (high NIR, medium-low Red) vs deforested (lower NIR, higher Red)
        is_forest = rng.random() > 0.3  # 70% forest tiles

        tile = rng.random((TILE_SIZE, TILE_SIZE, 4)).astype(np.float32)

        if is_forest:
            tile[:, :, 0] *= 0.3   # Red low
            tile[:, :, 3] *= 0.7 + 0.3   # NIR high — clamp
            tile[:, :, 3] = tile[:, :, 3].clip(0, 1)
        else:
            tile[:, :, 0] *= 0.6 + 0.3   # Red higher
            tile[:, :, 3] *= 0.3          # NIR low

        # Add patch of deforestation in some forest tiles
        if is_forest and rng.random() > 0.5:
            r, c = rng.integers(64, 192, size=2)
            tile[r:r+50, c:c+50, 0] += 0.3
            tile[r:r+50, c:c+50, 3] -= 0.3
            tile = tile.clip(0, 1)

        fname = DATA_TILES / f"demo_tile_{i:04d}.npy"
        np.save(fname, tile)

    # Save matching labels (0=forest, 1=deforested) for training
    log.info("Generating synthetic labels...")
    label_dir = DATA_TILES / "labels"
    label_dir.mkdir(exist_ok=True)
    for f in DATA_TILES.glob("demo_tile_*.npy"):
        tile = np.load(f)
        ndvi = (tile[:, :, 3] - tile[:, :, 0]) / (tile[:, :, 3] + tile[:, :, 0] + 1e-8)
        label = (ndvi < 0.3).astype(np.uint8)  # 1=deforested where NDVI < 0.3
        np.save(label_dir / f.name, label)

    log.info(f"✅ {n_tiles} demo tiles + labels created.")
    log.info("   Next: Run python stage3_train_model.py")


if __name__ == "__main__":
    main()
