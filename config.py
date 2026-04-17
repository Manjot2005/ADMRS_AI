"""
config.py — Central configuration for ADMRS
Fill in your API credentials before running.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load from .env file if present

# ─── Project Paths ───────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_TILES     = BASE_DIR / "data" / "tiles"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR     = BASE_DIR / "models"
OUTPUTS_DIR    = BASE_DIR / "outputs"

# Create dirs if missing
for d in [DATA_RAW, DATA_TILES, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Sentinel Hub / Copernicus Credentials ───────────────────────
# Sign up free at: https://scihub.copernicus.eu/
SENTINEL_USER     = os.getenv("SENTINEL_USER", "YOUR_USERNAME")
SENTINEL_PASSWORD = os.getenv("SENTINEL_PASSWORD", "YOUR_PASSWORD")
SENTINEL_API_URL  = "https://scihub.copernicus.eu/dhus"

# ─── Area of Interest (AOI) ───────────────────────────────────────
# Default: A region of the Amazon Basin (lon_min, lat_min, lon_max, lat_max)
AOI_BBOX = (-62.5, -4.5, -58.5, -1.5)   # [W, S, E, N]

# Date range for download
DATE_START = "20240101"
DATE_END   = "20240630"

# Cloud cover filter (%)
MAX_CLOUD_COVER = 20

# ─── Model Settings ───────────────────────────────────────────────
TILE_SIZE       = 256        # Pixels per tile (H x W)
BATCH_SIZE      = 8
EPOCHS          = 30
LEARNING_RATE   = 1e-4
NUM_CLASSES     = 2          # 0=Forest, 1=Deforested
IN_CHANNELS     = 4          # R, G, B, NIR bands
ENCODER_NAME    = "resnet34"
MODEL_PATH      = MODELS_DIR / "unet_forest.pth"

# ─── Change Detection ─────────────────────────────────────────────
DEFORESTATION_THRESHOLD_HA = 50.0    # Alert if loss > 50 ha
PIXEL_AREA_HA              = 0.01    # 10m pixel = 0.01 ha

# ─── Alerting ─────────────────────────────────────────────────────
# Twilio SMS — sign up free at https://www.twilio.com/
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "YOUR_TWILIO_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",  "YOUR_TWILIO_TOKEN")
TWILIO_FROM_PHONE  = os.getenv("TWILIO_FROM_PHONE",  "+1XXXXXXXXXX")
ALERT_TO_PHONE     = os.getenv("ALERT_TO_PHONE",     "+1XXXXXXXXXX")

# SendGrid Email — sign up free at https://sendgrid.com/
SENDGRID_API_KEY   = os.getenv("SENDGRID_API_KEY",   "YOUR_SENDGRID_KEY")
ALERT_FROM_EMAIL   = os.getenv("ALERT_FROM_EMAIL",   "alerts@yourproject.com")
ALERT_TO_EMAIL     = os.getenv("ALERT_TO_EMAIL",     "ranger@yourproject.com")

# ─── Dashboard ────────────────────────────────────────────────────
DASHBOARD_PORT = 8501
ALERTS_LOG     = OUTPUTS_DIR / "alerts_log.csv"
