"""
stage5_change_detect.py — Change Detection & Automated Alerting
Compares two time-period mask sets to detect new deforestation.
Fires SMS (Twilio) and Email (SendGrid) alerts when threshold is exceeded.

Run: python stage5_change_detect.py
"""

import csv
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from config import (
    DATA_PROCESSED, OUTPUTS_DIR,
    DEFORESTATION_THRESHOLD_HA, PIXEL_AREA_HA,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM_PHONE, ALERT_TO_PHONE,
    SENDGRID_API_KEY, ALERT_FROM_EMAIL, ALERT_TO_EMAIL,
    ALERTS_LOG
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MASKS_T1 = DATA_PROCESSED / "masks_t1"   # Earlier period masks
MASKS_T2 = DATA_PROCESSED / "masks"      # Current period masks (from stage 4)


# ─── Change Detection ─────────────────────────────────────────────

def detect_change(mask_t1: np.ndarray, mask_t2: np.ndarray) -> dict:
    """
    Compare two binary masks.
    Returns pixel counts for each transition type.

    Transitions:
      Forest → Forest    : stable (good)
      Deforested → Deforested: persistent loss
      Forest → Deforested: NEW loss ← the critical signal
      Deforested → Forest: regrowth (rare)
    """
    new_loss      = ((mask_t1 == 0) & (mask_t2 == 1)).sum()  # Forest → Deforested
    regrowth      = ((mask_t1 == 1) & (mask_t2 == 0)).sum()  # Deforested → Forest
    stable_forest = ((mask_t1 == 0) & (mask_t2 == 0)).sum()
    persistent    = ((mask_t1 == 1) & (mask_t2 == 1)).sum()

    return {
        "new_loss_pixels":   int(new_loss),
        "regrowth_pixels":   int(regrowth),
        "stable_pixels":     int(stable_forest),
        "persistent_pixels": int(persistent),
        "new_loss_ha":       round(new_loss * PIXEL_AREA_HA, 2),
        "regrowth_ha":       round(regrowth * PIXEL_AREA_HA, 2),
    }


def run_change_detection() -> list:
    """Compare matching tile pairs between T1 and T2 periods."""
    if not MASKS_T1.exists():
        log.warning(f"No T1 masks found at {MASKS_T1}")
        log.info("Creating synthetic T1 masks for demo (10% less deforestation)...")
        create_synthetic_t1_masks()

    t2_masks = sorted(MASKS_T2.glob("*.npy"))
    if not t2_masks:
        raise FileNotFoundError(f"No T2 masks found at {MASKS_T2}. Run stage4_inference.py first.")

    log.info(f"Comparing {len(t2_masks)} tile pairs (T1 vs T2)...")

    changes = []
    for t2_path in t2_masks:
        t1_path = MASKS_T1 / t2_path.name
        if not t1_path.exists():
            continue

        mask_t1 = np.load(t1_path)
        mask_t2 = np.load(t2_path)
        change  = detect_change(mask_t1, mask_t2)
        change["tile"] = t2_path.name
        changes.append(change)

    return changes


# ─── Alerting ─────────────────────────────────────────────────────

def send_sms_alert(sector: str, area_ha: float, alert_time: str):
    """Send SMS via Twilio."""
    if TWILIO_ACCOUNT_SID == "YOUR_TWILIO_SID":
        log.warning("  Twilio not configured — skipping SMS (set in config.py or .env)")
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = (
            f"🚨 DEFORESTATION ALERT\n"
            f"Sector: {sector}\n"
            f"New loss: {area_ha:.1f} ha\n"
            f"Time: {alert_time} UTC\n"
            f"[ForestWatch AI]"
        )
        message = client.messages.create(
            body=msg,
            from_=TWILIO_FROM_PHONE,
            to=ALERT_TO_PHONE
        )
        log.info(f"  ✅ SMS sent (SID: {message.sid})")
        return True
    except Exception as e:
        log.error(f"  SMS failed: {e}")
        return False


def send_email_alert(sector: str, area_ha: float, alert_time: str, details: dict):
    """Send email via SendGrid."""
    if SENDGRID_API_KEY == "YOUR_SENDGRID_KEY":
        log.warning("  SendGrid not configured — skipping email (set in config.py or .env)")
        return False

    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail

        html = f"""
        <h2 style="color:#ef4444;">🚨 Deforestation Alert — {sector}</h2>
        <table border="1" cellpadding="8" style="border-collapse:collapse;">
          <tr><td><b>New Forest Loss</b></td><td>{area_ha:.1f} ha</td></tr>
          <tr><td><b>Stable Forest</b></td><td>{details.get('stable_pixels',0) * PIXEL_AREA_HA:.1f} ha</td></tr>
          <tr><td><b>Vegetation Regrowth</b></td><td>{details.get('regrowth_ha',0):.1f} ha</td></tr>
          <tr><td><b>Detected At</b></td><td>{alert_time} UTC</td></tr>
        </table>
        <p>Please verify using the ForestWatch AI dashboard.</p>
        """

        message = Mail(
            from_email=ALERT_FROM_EMAIL,
            to_emails=ALERT_TO_EMAIL,
            subject=f"[ForestWatch] 🚨 {area_ha:.0f} ha deforestation detected — {sector}",
            html_content=html,
        )
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        response = sg.send(message)
        log.info(f"  ✅ Email sent (status: {response.status_code})")
        return True
    except Exception as e:
        log.error(f"  Email failed: {e}")
        return False


def log_alert_to_csv(sector: str, area_ha: float, change: dict):
    """Append alert record to CSV log file."""
    row = {
        "timestamp":    datetime.utcnow().isoformat(),
        "sector":       sector,
        "new_loss_ha":  area_ha,
        "regrowth_ha":  change.get("regrowth_ha", 0),
        "tile":         change.get("tile", ""),
    }
    write_header = not ALERTS_LOG.exists()
    with open(ALERTS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def process_alerts(changes: list):
    """Check each tile for threshold exceedance and fire alerts."""
    alert_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    alert_count = 0

    log.info(f"\nChecking {len(changes)} tiles against threshold ({DEFORESTATION_THRESHOLD_HA} ha)...")

    for change in sorted(changes, key=lambda x: -x["new_loss_ha"]):
        area_ha = change["new_loss_ha"]
        tile    = change["tile"]

        if area_ha < DEFORESTATION_THRESHOLD_HA:
            continue

        # Derive a human-readable sector name from tile filename
        sector = tile.replace(".npy", "").replace("demo_tile_", "Sector-").replace("_", " ")[:20]

        log.warning(f"\n⚠️  ALERT: {sector}")
        log.warning(f"   New loss: {area_ha:.1f} ha (Threshold: {DEFORESTATION_THRESHOLD_HA} ha)")

        # Fire notifications
        send_sms_alert(sector, area_ha, alert_time)
        send_email_alert(sector, area_ha, alert_time, change)
        log_alert_to_csv(sector, area_ha, change)
        alert_count += 1

    return alert_count


# ─── Demo T1 masks ─────────────────────────────────────────────────

def create_synthetic_t1_masks():
    """Create synthetic T1 masks (slightly less deforested than T2) for demo."""
    MASKS_T1.mkdir(parents=True, exist_ok=True)
    t2_masks = sorted(MASKS_T2.glob("*.npy"))
    rng = np.random.default_rng(99)

    for t2_path in t2_masks:
        mask_t2 = np.load(t2_path)
        # T1 has ~20% fewer deforested pixels (simulate new clearing since T1)
        mask_t1 = mask_t2.copy()
        deforested_idx = np.argwhere(mask_t2 == 1)
        if len(deforested_idx) > 0:
            n_revert = int(0.25 * len(deforested_idx))
            chosen = rng.choice(len(deforested_idx), size=n_revert, replace=False)
            for idx in chosen:
                r, c = deforested_idx[idx]
                mask_t1[r, c] = 0  # Was forest in T1
        np.save(MASKS_T1 / t2_path.name, mask_t1)

    log.info(f"  Created {len(t2_masks)} synthetic T1 masks in {MASKS_T1}")


# ─── Summary Report ───────────────────────────────────────────────

def print_summary(changes: list):
    """Print a summary table of all changes."""
    total_new_loss  = sum(c["new_loss_ha"] for c in changes)
    total_regrowth  = sum(c["regrowth_ha"] for c in changes)
    total_tiles     = len(changes)
    affected_tiles  = sum(1 for c in changes if c["new_loss_ha"] > 0)

    log.info("\n" + "=" * 60)
    log.info("CHANGE DETECTION REPORT")
    log.info("=" * 60)
    log.info(f"  Tiles compared:      {total_tiles}")
    log.info(f"  Tiles with new loss: {affected_tiles}")
    log.info(f"  Total new loss:      {total_new_loss:,.1f} ha")
    log.info(f"  Total regrowth:      {total_regrowth:,.1f} ha")
    log.info(f"  Net change:          {total_new_loss - total_regrowth:,.1f} ha")

    # Top 5 worst tiles
    top5 = sorted(changes, key=lambda x: -x["new_loss_ha"])[:5]
    log.info("\n  Top 5 tiles by new loss:")
    for r in top5:
        log.info(f"    {r['tile']}: {r['new_loss_ha']} ha")

    # Save JSON report
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_tiles": total_tiles,
            "affected_tiles": affected_tiles,
            "total_new_loss_ha": round(total_new_loss, 2),
            "total_regrowth_ha": round(total_regrowth, 2),
        },
        "tiles": changes,
    }
    report_path = OUTPUTS_DIR / "change_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info(f"\n  Full report saved → {report_path}")


def main():
    log.info("=" * 60)
    log.info("STAGE 5: Change Detection & Alerting")
    log.info("=" * 60)

    changes     = run_change_detection()
    print_summary(changes)
    alert_count = process_alerts(changes)

    log.info("=" * 60)
    log.info(f"✅ Stage 5 complete. Alerts fired: {alert_count}")
    log.info(f"   Alert log: {ALERTS_LOG}")
    log.info("   Next: Run streamlit run dashboard/app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
