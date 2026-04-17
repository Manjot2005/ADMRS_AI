"""
stage1_download.py — Data Acquisition (Copernicus Data Space Ecosystem)
Uses the NEW CDSE OData API (dataspace.copernicus.eu).
The old SciHub (scihub.copernicus.eu) is SHUT DOWN — this replaces it.

Credentials: set SENTINEL_USER and SENTINEL_PASSWORD in your .env file
Register free at: https://dataspace.copernicus.eu

Run: python stage1_download.py
"""

import os
import json
import logging
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from config import (
    SENTINEL_USER, SENTINEL_PASSWORD,
    AOI_BBOX, DATE_START, DATE_END,
    MAX_CLOUD_COVER, DATA_RAW
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── CDSE endpoints ──────────────────────────────────────────────
TOKEN_URL    = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL  = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


# ─── Step 1: Get access token ────────────────────────────────────

def get_access_token(username: str, password: str) -> str:
    """
    Authenticate with Copernicus Data Space Ecosystem (Keycloak).
    Returns a short-lived Bearer token for API calls.
    """
    log.info("Authenticating with Copernicus Data Space Ecosystem...")
    data = {
        "client_id":  "cdse-public",
        "username":   username,
        "password":   password,
        "grant_type": "password",
    }
    response = requests.post(TOKEN_URL, data=data, timeout=30)
    if response.status_code != 200:
        raise ConnectionError(
            f"Authentication failed (HTTP {response.status_code}).\n"
            f"Response: {response.text}\n"
            "Check your SENTINEL_USER and SENTINEL_PASSWORD in .env"
        )
    token = response.json()["access_token"]
    log.info("✅ Authentication successful")
    return token


# ─── Step 2: Build AOI WKT polygon ───────────────────────────────

def bbox_to_wkt(bbox: tuple) -> str:
    """Convert (W, S, E, N) bbox to WKT POLYGON string for OData filter."""
    w, s, e, n = bbox
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


# ─── Step 3: Search catalogue ────────────────────────────────────

def search_sentinel2(token: str) -> list:
    """
    Search CDSE catalogue for Sentinel-2 L2A products.
    Returns list of product dicts with Id, Name, S3Path etc.
    """
    log.info("Searching Sentinel-2 L2A products on CDSE catalogue...")

    # Convert dates to ISO format
    start_iso = f"{DATE_START[:4]}-{DATE_START[4:6]}-{DATE_START[6:]}T00:00:00.000Z"
    end_iso   = f"{DATE_END[:4]}-{DATE_END[4:6]}-{DATE_END[6:]}T23:59:59.000Z"
    aoi_wkt   = bbox_to_wkt(AOI_BBOX)

    filter_str = (
        f"Collection/Name eq 'SENTINEL-2'"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')"
        f" and ContentDate/Start gt {start_iso}"
        f" and ContentDate/Start lt {end_iso}"
        f" and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {MAX_CLOUD_COVER})"
        f" and contains(Name,'L2A')"
    )

    params = {
        "$filter":  filter_str,
        "$orderby": "ContentDate/Start desc",
        "$top":     "20",
        "$expand":  "Attributes",
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(CATALOGUE_URL, params=params, headers=headers, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Catalogue search failed: HTTP {response.status_code}\n{response.text}")

    products = response.json().get("value", [])
    log.info(f"Found {len(products)} Sentinel-2 L2A products")
    return products


def search_sentinel1(token: str) -> list:
    """Search CDSE catalogue for Sentinel-1 GRD products (SAR/radar)."""
    log.info("Searching Sentinel-1 GRD products on CDSE catalogue...")

    start_iso = f"{DATE_START[:4]}-{DATE_START[4:6]}-{DATE_START[6:]}T00:00:00.000Z"
    end_iso   = f"{DATE_END[:4]}-{DATE_END[4:6]}-{DATE_END[6:]}T23:59:59.000Z"
    aoi_wkt   = bbox_to_wkt(AOI_BBOX)

    filter_str = (
        f"Collection/Name eq 'SENTINEL-1'"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')"
        f" and ContentDate/Start gt {start_iso}"
        f" and ContentDate/Start lt {end_iso}"
        f" and contains(Name,'GRD')"
    )

    params = {
        "$filter":  filter_str,
        "$orderby": "ContentDate/Start desc",
        "$top":     "10",
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(CATALOGUE_URL, params=params, headers=headers, timeout=60)

    if response.status_code != 200:
        log.warning(f"Sentinel-1 search failed: HTTP {response.status_code}")
        return []

    products = response.json().get("value", [])
    log.info(f"Found {len(products)} Sentinel-1 GRD products")
    return products


# ─── Step 4: Download products ────────────────────────────────────

def download_product(product: dict, out_dir: Path, token: str) -> bool:
    """
    Download a single product ZIP from CDSE.
    Uses the zipper endpoint with streaming to handle large files.
    """
    product_id   = product["Id"]
    product_name = product["Name"]
    out_path     = out_dir / f"{product_name}.zip"

    if out_path.exists():
        log.info(f"  Already downloaded: {product_name}")
        return True

    url     = f"{DOWNLOAD_URL}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    log.info(f"  Downloading: {product_name}")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=300)

        if response.status_code == 401:
            log.error("  Token expired. Re-authenticate.")
            return False
        if response.status_code != 200:
            log.error(f"  Download failed: HTTP {response.status_code}")
            return False

        total_bytes = int(response.headers.get("content-length", 0))
        total_mb    = total_bytes / (1024 * 1024)

        with open(out_path, "wb") as f, tqdm(
            total=total_bytes, unit="B", unit_scale=True,
            desc=f"    {product_name[:40]}", leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        actual_mb = out_path.stat().st_size / (1024 * 1024)
        log.info(f"  ✅ Saved: {out_path.name} ({actual_mb:.1f} MB)")
        return True

    except requests.exceptions.Timeout:
        log.error(f"  Timeout downloading {product_name}")
        if out_path.exists(): out_path.unlink()
        return False
    except Exception as e:
        log.error(f"  Error: {e}")
        if out_path.exists(): out_path.unlink()
        return False


def download_all(products: list, out_dir: Path, token: str) -> tuple:
    """Download all products, return (success_count, failed_count)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for product in products:
        if download_product(product, out_dir, token):
            ok += 1
        else:
            fail += 1
    return ok, fail


# ─── Step 5: Save metadata ────────────────────────────────────────

def save_metadata(s2_products: list, s1_products: list):
    """Save product metadata to JSON for later reference."""
    meta = {
        "downloaded_at":    datetime.utcnow().isoformat(),
        "platform":         "Copernicus Data Space Ecosystem (CDSE)",
        "api_url":          CATALOGUE_URL,
        "aoi_bbox":         AOI_BBOX,
        "date_range":       [DATE_START, DATE_END],
        "max_cloud_cover":  MAX_CLOUD_COVER,
        "sentinel2_count":  len(s2_products),
        "sentinel1_count":  len(s1_products),
        "sentinel2_names":  [p["Name"] for p in s2_products],
        "sentinel1_names":  [p["Name"] for p in s1_products],
    }
    meta_path = DATA_RAW / "download_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Metadata saved → {meta_path}")


# ─── Main ────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("STAGE 1: Data Acquisition (Copernicus Data Space Ecosystem)")
    log.info("=" * 60)

    # Validate credentials
    if SENTINEL_USER in ("YOUR_USERNAME", "", None):
        log.error("SENTINEL_USER not set. Edit your .env file:")
        log.error("  SENTINEL_USER=your_email@example.com")
        log.error("  SENTINEL_PASSWORD=your_password")
        log.error("Register free at: https://dataspace.copernicus.eu")
        return

    log.info(f"User:        {SENTINEL_USER}")
    log.info(f"AOI:         {AOI_BBOX}")
    log.info(f"Date range:  {DATE_START} → {DATE_END}")
    log.info(f"Max cloud:   {MAX_CLOUD_COVER}%")

    # Authenticate
    try:
        token = get_access_token(SENTINEL_USER, SENTINEL_PASSWORD)
    except ConnectionError as e:
        log.error(str(e))
        return

    # Search
    s2_products = search_sentinel2(token)
    s1_products = search_sentinel1(token)

    if not s2_products and not s1_products:
        log.warning("No products found. Try widening date range or cloud cover in config.py")
        log.warning("Running pipeline in demo mode (synthetic tiles)...")
        import stage2_preprocess as s2mod
        s2mod.create_synthetic_demo_tiles()
        return

    # Print product list
    log.info("\nSentinel-2 products found:")
    for p in s2_products[:5]:
        cloud = next(
            (a["Value"] for a in p.get("Attributes", []) if a.get("Name") == "cloudCover"),
            "N/A"
        )
        log.info(f"  {p['Name']}  | Cloud: {cloud:.1f}%" if isinstance(cloud, float) else f"  {p['Name']}")

    # Download Sentinel-2
    s2_dir = DATA_RAW / "sentinel2"
    log.info(f"\nDownloading {len(s2_products)} Sentinel-2 tiles → {s2_dir}")
    ok2, fail2 = download_all(s2_products, s2_dir, token)

    # Download Sentinel-1
    s1_dir = DATA_RAW / "sentinel1"
    if s1_products:
        log.info(f"\nDownloading {len(s1_products)} Sentinel-1 tiles → {s1_dir}")
        ok1, fail1 = download_all(s1_products, s1_dir, token)
    else:
        ok1, fail1 = 0, 0

    # Save metadata
    save_metadata(s2_products, s1_products)

    log.info("=" * 60)
    log.info(f"✅ Stage 1 complete.")
    log.info(f"   Sentinel-2: {ok2} downloaded, {fail2} failed")
    log.info(f"   Sentinel-1: {ok1} downloaded, {fail1} failed")
    log.info(f"   Files in:   {DATA_RAW}")
    log.info("   Next: Run python stage2_preprocess.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
