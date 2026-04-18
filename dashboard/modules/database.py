"""
database.py — SQLite persistence layer for ADMRS
Replaces fragile CSV writes with thread-safe SQLite transactions.
"""
import sqlite3, json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import os as _os
# On Streamlit Cloud the repo is read-only; use /tmp which is always writable.
# Locally, use the outputs/ folder next to the project root.
if _os.getenv("STREAMLIT_SHARING_MODE") or _os.getenv("HOME", "").startswith("/home/adminuser"):
    DB_PATH = Path("/tmp/admrs.db")
else:
    DB_PATH = Path(__file__).parent.parent.parent / "outputs" / "admrs.db"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Connection context manager ─────────────────────────────────────
@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent read-write safe
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ── Schema bootstrap ───────────────────────────────────────────────
def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS validations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            alert_id    TEXT NOT NULL,
            sector      TEXT,
            new_loss_ha REAL,
            verdict     TEXT,
            notes       TEXT
        );
        CREATE TABLE IF NOT EXISTS dispatches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            alert_id    TEXT NOT NULL,
            sector      TEXT,
            ranger      TEXT,
            status      TEXT,
            notes       TEXT,
            mission_pdf TEXT
        );
        CREATE TABLE IF NOT EXISTS evidence (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            alert_id    TEXT NOT NULL,
            ranger      TEXT,
            filename    TEXT,
            file_bytes  BLOB,
            notes       TEXT
        );
        CREATE TABLE IF NOT EXISTS forecast_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            data_json   TEXT NOT NULL
        );
        """)

# ── Validations ────────────────────────────────────────────────────
def save_validation(alert_id, sector, ha, verdict, notes=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO validations(timestamp,alert_id,sector,new_loss_ha,verdict,notes) "
            "VALUES(?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), alert_id, sector, ha, verdict, notes))

def load_validations():
    import pandas as pd
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM validations ORDER BY timestamp DESC").fetchall()
    if rows:
        return pd.DataFrame([dict(r) for r in rows])
    return pd.DataFrame(columns=["id","timestamp","alert_id","sector","new_loss_ha","verdict","notes"])

# ── Dispatches ─────────────────────────────────────────────────────
def save_dispatch(alert_id, sector, ranger, status, notes="", mission_pdf=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO dispatches(timestamp,alert_id,sector,ranger,status,notes,mission_pdf) "
            "VALUES(?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), alert_id, sector, ranger, status, notes, mission_pdf))

def load_dispatch():
    import pandas as pd
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM dispatches ORDER BY timestamp DESC").fetchall()
    if rows:
        return pd.DataFrame([dict(r) for r in rows])
    return pd.DataFrame(columns=["id","timestamp","alert_id","sector","ranger","status","notes"])

def get_dispatch_ids():
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT alert_id FROM dispatches").fetchall()
    return {r["alert_id"] for r in rows}

# ── Evidence ───────────────────────────────────────────────────────
def save_evidence(alert_id, ranger, filename, file_bytes, notes=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO evidence(timestamp,alert_id,ranger,filename,file_bytes,notes) "
            "VALUES(?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), alert_id, ranger, filename, file_bytes, notes))

def load_evidence(alert_id=None):
    import pandas as pd
    with get_conn() as conn:
        if alert_id:
            rows = conn.execute(
                "SELECT id,timestamp,alert_id,ranger,filename,notes FROM evidence "
                "WHERE alert_id=? ORDER BY timestamp DESC", (alert_id,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id,timestamp,alert_id,ranger,filename,notes FROM evidence "
                "ORDER BY timestamp DESC").fetchall()
    if rows:
        return pd.DataFrame([dict(r) for r in rows])
    return pd.DataFrame(columns=["id","timestamp","alert_id","ranger","filename","notes"])

def get_evidence_file(evidence_id):
    """Return (filename, bytes) for download."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT filename,file_bytes FROM evidence WHERE id=?", (evidence_id,)).fetchone()
    if row:
        return row["filename"], bytes(row["file_bytes"])
    return None, None

# ── Forecast cache ─────────────────────────────────────────────────
def save_forecast(data_dict):
    with get_conn() as conn:
        conn.execute("DELETE FROM forecast_cache")   # keep only latest
        conn.execute(
            "INSERT INTO forecast_cache(created_at,data_json) VALUES(?,?)",
            (datetime.utcnow().isoformat(), json.dumps(data_dict)))

def load_forecast():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM forecast_cache ORDER BY created_at DESC LIMIT 1").fetchone()
    if row:
        return json.loads(row["data_json"]), row["created_at"]
    return None, None

# Initialise on import
init_db()
