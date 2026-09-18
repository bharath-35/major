"""
utils.py — Shared utility helpers for the live DDoS detection pipeline.

Responsibilities:
  • Resolve project-relative paths (model, scaler, data, outputs).
  • Generate and persist scaler.pkl from the training dataset.
  • Validate that extracted live features match training features exactly.
  • Provide a consistent logger used by all live-pipeline modules.

NOTE: This file is NEW.  It does NOT touch ddos_detector.py / ddos_main.py.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Canonical feature list (must match ddos_detector.py FEATURES exactly) ─────
FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Total",
    "Bwd IAT Total",
    "Packet Length Mean",
    "Packet Length Std",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
]

NUM_FEATURES = len(FEATURES)   # 22

# ── Project-relative path helpers ──────────────────────────────────────────────
# src/ lives inside the project root; all outputs are relative to the root.
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

def _proj(relpath: str) -> str:
    """Return an absolute path relative to the project root."""
    return os.path.join(_PROJECT_DIR, relpath)

# Well-known paths used across the live pipeline
MODEL_PATH        = _proj(os.path.join("docs", "global_model.h5"))
SCALER_PATH       = _proj("scaler.pkl")
TRAINING_CSV_PATH = _proj(
    os.path.join(
        "data",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    )
)
PCAP_PATH                = _proj("traffic.pcap")
LIVE_FEATURES_CSV        = _proj("live_features.csv")        # legacy
LIVE_DATASET_CSV         = _proj("live_dataset.csv")         # CICIDS2017-style output
LIVE_DATASET_PREDICTED_CSV = _proj("live_dataset_predicted.csv")  # post-prediction
PREDICTIONS_CSV          = _proj("predictions.csv")


# ── Logger ─────────────────────────────────────────────────────────────────────
def get_logger(name: str = "ddos_live") -> logging.Logger:
    """
    Return a module-level logger with a UTF-8-safe stream handler.

    All live-pipeline modules should call this to get a shared logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:           # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    # Wrap in a UTF-8 reconfigure if the terminal can't handle Unicode
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation):
        pass

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ── Scaler helpers ─────────────────────────────────────────────────────────────
#
# IMPORTANT — two separate loading strategies:
#
#   load_scaler_strict()      → TESTING / INFERENCE phase
#                               Loads scaler.pkl ONLY.  Raises SystemExit if missing.
#                               Called by live_predict.py.  Never fits a scaler.
#
#   load_or_create_scaler()   → TRAINING phase setup only
#                               Auto-generates scaler.pkl if absent.
#                               Must NOT be called from the live pipeline.
# ─────────────────────────────────────────────────────────────────────────────
def generate_scaler(csv_path: str = TRAINING_CSV_PATH,
                    save_path: str = SCALER_PATH) -> StandardScaler:
    """
    Fit a StandardScaler on the CICIDS2017 training CSV using the same 22
    features and preprocessing steps as ddos_detector.preprocess().

    The fitted scaler is saved to *save_path* so it can be reused by
    live_predict.py without touching the original training code.

    Args:
        csv_path:  Path to the CICIDS2017 DDoS CSV file.
        save_path: Where to persist the scaler (joblib format).

    Returns:
        Fitted StandardScaler instance.
    """
    log = get_logger()
    log.info("Generating scaler from training data: %s", csv_path)

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Keep only the features that exist in the CSV (safety guard)
    available = [f for f in FEATURES if f in df.columns]
    if len(available) < NUM_FEATURES:
        missing = set(FEATURES) - set(available)
        log.warning("CSV missing %d feature(s): %s", len(missing), missing)

    X = (
        df[available]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .values.astype(np.float32)
    )

    scaler = StandardScaler()
    scaler.fit(X)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(scaler, save_path)
    log.info("Scaler saved -> %s  (fitted on %d rows × %d features)",
             save_path, X.shape[0], X.shape[1])
    return scaler


def load_scaler_strict(scaler_path: str = SCALER_PATH) -> StandardScaler:
    """
    INFERENCE-ONLY scaler loader.

    Loads and returns the existing scaler.pkl.  If the file does not exist,
    prints a clear error and stops execution — it will NEVER fit a new scaler
    or generate one from the training dataset.

    Call this from live_predict.py and nowhere else.

    Args:
        scaler_path: Path to scaler.pkl.

    Returns:
        Loaded StandardScaler instance.

    Raises:
        SystemExit: if scaler.pkl is missing.
    """
    log = get_logger()
    if not os.path.exists(scaler_path):
        log.error(
            "\n"
            "  ================================================================\n"
            "  ERROR: scaler.pkl not found at:\n"
            "         %s\n"
            "\n"
            "  The live testing pipeline requires a pre-fitted scaler.\n"
            "  scaler.pkl must be generated ONCE during the training phase:\n"
            "\n"
            "      python generate_scaler.py\n"
            "\n"
            "  The live pipeline will NOT create or fit a new scaler.\n"
            "  ================================================================",
            scaler_path,
        )
        raise SystemExit(1)

    scaler = joblib.load(scaler_path)
    log.info("Scaler loaded (inference-only): %s", scaler_path)
    return scaler


def load_or_create_scaler(scaler_path: str = SCALER_PATH,
                           csv_path: str = TRAINING_CSV_PATH) -> StandardScaler:
    """
    Load scaler.pkl if it exists; otherwise generate it from the training CSV.

    This is the single entry-point for live_predict.py to obtain the scaler.
    """
    log = get_logger()
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        log.info("Scaler loaded from %s", scaler_path)
        return scaler

    log.info("scaler.pkl not found — generating from training dataset …")
    return generate_scaler(csv_path=csv_path, save_path=scaler_path)


# ── Feature validation ─────────────────────────────────────────────────────────
def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and reorder *df* columns to match the canonical FEATURES list.

    Rules applied:
      • If a required feature column is missing, it is filled with 0.0.
      • Columns are reordered to the canonical order.
      • Values of ±inf are replaced with NaN then filled with 0.
      • NaN values are filled with 0.
      • dtypes are cast to float32.

    Args:
        df: DataFrame whose columns are (a subset of) FEATURES.

    Returns:
        DataFrame with exactly 22 columns in the canonical order, float32.

    Raises:
        ValueError: if *df* has no rows.
    """
    log = get_logger()

    if df.empty:
        raise ValueError("validate_features(): received an empty DataFrame.")

    # Fill any missing feature columns with 0
    for feat in FEATURES:
        if feat not in df.columns:
            log.warning("Feature '%s' missing from extracted data — filled with 0", feat)
            df[feat] = 0.0

    # Enforce canonical column order
    df = df[FEATURES].copy()

    # Sanitise values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype(np.float32)

    return df


# ── Misc helpers ───────────────────────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    """Create the directory for *path* (file or directory) if it doesn't exist."""
    target = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if target:
        os.makedirs(target, exist_ok=True)


def safe_div(numerator: float, denominator: float,
             default: float = 0.0) -> float:
    """Division that returns *default* when denominator is zero."""
    return numerator / denominator if denominator != 0 else default
