
from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional

# ── UTF-8 stdout ───────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure src/ is on path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import (  # noqa: E402
    FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    LIVE_DATASET_CSV,
    LIVE_DATASET_PREDICTED_CSV,
    PREDICTIONS_CSV,
    get_logger,
    validate_features,
    load_scaler_strict,         # inference-only — errors if scaler.pkl missing
    ensure_dir,
)

log = get_logger("live_predict")

# ── TensorFlow ─────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
except ImportError as exc:
    log.error("TensorFlow is not installed.  Run: pip install tensorflow")
    raise SystemExit(1) from exc

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Load the pre-trained Keras model (read-only, compile=False).
    No weights are changed. No retraining occurs.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run ddos_main.py first to generate global_model.h5."
        )
    log.info("Loading model: %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    log.info("Model loaded.  Input shape: %s", model.input_shape)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prediction core
# ─────────────────────────────────────────────────────────────────────────────

def predict_flows(
    df_features: pd.DataFrame,
    model: tf.keras.Model,
    scaler,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Scale *df_features* and run inference.

    Args:
        df_features: DataFrame with exactly 22 feature columns (canonical order).
        model:       Loaded Keras model (sigmoid output — NOT retrained).
        scaler:      Fitted StandardScaler (same as used during training).
        threshold:   Decision boundary (default 0.5).

    Returns:
        DataFrame with columns: Flow_ID, Prediction (int), Confidence, Label (str).
    """
    if df_features.empty:
        log.warning("Empty feature set — returning empty predictions.")
        return pd.DataFrame(columns=["Flow_ID", "Prediction", "Confidence", "Label"])

    df_valid = validate_features(df_features.copy())           # enforce order / dtype
    X        = df_valid.values.astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)          # transform only

    raw_scores  = model.predict(X_scaled, verbose=0).flatten()  # sigmoid output
    preds       = (raw_scores >= threshold).astype(int)
    label_names = ["DDoS" if p == 1 else "Normal" for p in preds]

    return pd.DataFrame({
        "Flow_ID":    range(len(raw_scores)),
        "Prediction": preds,
        "Confidence": np.round(raw_scores, 6),
        "Label":      label_names,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Console output (color-coded)
# ─────────────────────────────────────────────────────────────────────────────

_RED   = "\033[91m"
_GREEN = "\033[92m"
_RESET = "\033[0m"

def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}" if sys.stdout.isatty() else text


def print_predictions(results: pd.DataFrame, show_all: bool = True) -> None:
    """Print a clean preview table and summary — DDoS in red, Normal in green."""
    if results.empty:
        print("  [!] No predictions to display.")
        return

    col_w   = max(10, max(len(str(r)) for r in results.get("Flow_ID", [0])) + 2)
    header  = f"  {'Flow_ID':>{col_w}}  {'Prediction':<10}  {'Confidence':>12}"
    sep     = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    ddos_n   = int((results["Prediction"] == 1).sum())
    normal_n = int((results["Prediction"] == 0).sum())
    total    = len(results)

    if total <= 16:
        rows_to_show = results
        omitted = 0
    else:
        rows_to_show = pd.concat([results.iloc[:5], results.iloc[-5:]])
        omitted = total - 10

    first_half_done = False
    for i, (_, row) in enumerate(rows_to_show.iterrows()):
        if omitted > 0 and i == 5 and not first_half_done:
            dots = "..."
            msg  = f"({omitted} flows omitted — all saved to predictions.csv)"
            print(f"  {dots:>{col_w}}  {msg}")
            first_half_done = True

        fid   = int(row["Flow_ID"])
        pred  = int(row["Prediction"])
        conf  = float(row["Confidence"])
        label = str(row["Label"])

        line = f"  {fid:>{col_w}}  {label:<10}  {conf:>12.6f}"
        if pred == 1:
            print(_c(line, _RED))
        else:
            print(_c(line, _GREEN))

    print(sep)
    ddos_pct = 100 * ddos_n / total if total else 0
    print(
        f"  Summary : {total} flows | "
        f"{_c(f'DDoS: {ddos_n}', _RED)}  "
        f"{_c(f'Normal: {normal_n}', _GREEN)}  "
        f"DDoS rate: {ddos_pct:.1f}%\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def _save_predictions_csv(results: pd.DataFrame, out_path: str) -> None:
    """
    Save predictions.csv with columns: Flow_ID, Prediction, Confidence.

    Example:
        Flow_ID,Prediction,Confidence
        0,Normal,0.001758
        1,DDoS,0.972341
    """
    ensure_dir(out_path)
    df_out = pd.DataFrame({
        "Flow_ID":    results["Flow_ID"],
        "Prediction": results["Label"],        # human-readable string
        "Confidence": results["Confidence"],
    })
    df_out.to_csv(out_path, index=False)
    log.info("predictions.csv saved → %s  (%d rows)", out_path, len(df_out))


def _save_predicted_dataset(
    df_features: pd.DataFrame,
    results: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Save live_dataset_predicted.csv:
        [22 feature columns in canonical order] + Prediction + Confidence

    This mirrors a labeled CICIDS2017 dataset and is suitable for demos.

    Example:
        Flow Duration,...,Avg Bwd Segment Size,Prediction,Confidence
        1234.0,...,250.2,Normal,0.001758
    """
    ensure_dir(out_path)

    # Ensure canonical feature order + float64 (matching CICIDS2017 dtypes)
    df_feat = validate_features(df_features.copy()).astype("float64")

    df_out = df_feat.copy()
    df_out["Prediction"] = results["Label"].values
    df_out["Confidence"] = results["Confidence"].values

    df_out.to_csv(out_path, index=False)
    log.info(
        "live_dataset_predicted.csv saved → %s  (%d rows × %d cols)",
        out_path, len(df_out), len(df_out.columns),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    dataset_csv: str   = LIVE_DATASET_CSV,
    model_path: str    = MODEL_PATH,
    scaler_path: str   = SCALER_PATH,
    out_csv: str       = PREDICTIONS_CSV,
    predicted_csv: str = LIVE_DATASET_PREDICTED_CSV,
    threshold: float   = 0.5,
    show_all: bool     = True,
) -> pd.DataFrame:
    """
    Strictly inference-only pipeline.
    Calls ONLY scaler.transform() and model.predict().
    Will never call scaler.fit(), fit_transform(), model.fit(),
    generate_scaler(), or any federated learning operation.

    Args:
        dataset_csv:    Path to live_dataset.csv (22 features + Label="Unknown").
        model_path:     Path to global_model.h5  (loaded read-only, compile=False).
        scaler_path:    Path to scaler.pkl — MUST already exist.
                        If missing: error message is printed and execution stops.
        out_csv:        Destination for predictions.csv.
        predicted_csv:  Destination for live_dataset_predicted.csv.
        threshold:      Sigmoid confidence threshold for DDoS classification.
        show_all:       Print both Normal and DDoS rows in the console.

    Returns:
        Predictions DataFrame (Flow_ID, Prediction, Confidence, Label).
    """
    # ── Step 1: Load model (read-only, compile=False, no weight changes) ─────
    model = load_model(model_path)

    # ── Step 2: Load existing scaler.pkl (strict — never fits a new one) ─────
    # load_scaler_strict() raises SystemExit if scaler.pkl does not exist.
    # The live pipeline ONLY calls scaler.transform(), never scaler.fit().
    scaler = load_scaler_strict(scaler_path)

    # Validate dimensionality (sanity check only — no regeneration)
    expected = model.input_shape[-1]
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != expected:
        log.error(
            "scaler.pkl was fitted on %d features but model expects %d.\n"
            "Delete scaler.pkl and re-run generate_scaler.py to rebuild it.",
            scaler.n_features_in_, expected,
        )
        raise SystemExit(1)

    # ── Step 3: Read live_dataset.csv ─────────────────────────────────────────
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_csv}\n"
            "Run feature_extractor.py first."
        )
    log.info("Reading live_dataset.csv: %s", dataset_csv)
    df_raw = pd.read_csv(dataset_csv)
    log.info("Loaded %d rows × %d columns", len(df_raw), len(df_raw.columns))

    # Drop the "Label" column — it is "Unknown" at this stage and must not
    # be passed to the model or scaler.
    df_raw.drop(columns=["Label"], errors="ignore", inplace=True)

    # Extract only the 22 feature columns (guard against extra columns)
    feat_cols   = [c for c in FEATURES if c in df_raw.columns]
    df_features = df_raw[feat_cols].copy()

    if df_features.empty:
        log.warning("No feature rows found in %s", dataset_csv)
        return pd.DataFrame(columns=["Flow_ID", "Prediction", "Confidence", "Label"])

    # ── Step 4+5: Scale + predict ─────────────────────────────────────────────
    results = predict_flows(df_features, model, scaler, threshold)

    # ── Step 6: Console output ────────────────────────────────────────────────
    print_predictions(results, show_all=show_all)

    # ── Step 7: Save predictions.csv ──────────────────────────────────────────
    _save_predictions_csv(results, out_csv)

    # ── Step 8: Save live_dataset_predicted.csv ───────────────────────────────
    _save_predicted_dataset(df_features, results, predicted_csv)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Watch mode (poll live_dataset.csv for new rows)
# ─────────────────────────────────────────────────────────────────────────────

def watch_and_predict(
    dataset_csv: str   = LIVE_DATASET_CSV,
    model_path: str    = MODEL_PATH,
    scaler_path: str   = SCALER_PATH,
    out_csv: str       = PREDICTIONS_CSV,
    predicted_csv: str = LIVE_DATASET_PREDICTED_CSV,
    threshold: float   = 0.5,
    poll_interval: float = 5.0,
) -> None:
    """
    Continuously poll live_dataset.csv for new rows and classify them.
    Strictly inference-only: only scaler.transform() + model.predict() are called.
    scaler.pkl must already exist — execution stops if it is missing.
    """
    log.info(
        "Watch mode — polling '%s' every %.1f s  (CTRL-C to stop)",
        dataset_csv, poll_interval,
    )

    model  = load_model(model_path)
    scaler = load_scaler_strict(scaler_path)   # strict — never fits

    last_row_count = 0
    ensure_dir(out_csv)
    ensure_dir(predicted_csv)

    try:
        while True:
            if os.path.exists(dataset_csv):
                try:
                    df_raw = pd.read_csv(dataset_csv)
                except Exception as e:
                    log.warning("Could not read %s: %s", dataset_csv, e)
                    time.sleep(poll_interval)
                    continue

                if len(df_raw) > last_row_count:
                    new_rows = df_raw.iloc[last_row_count:].copy()
                    last_row_count = len(df_raw)

                    new_rows.drop(columns=["Label"], errors="ignore", inplace=True)
                    feat_cols   = [c for c in FEATURES if c in new_rows.columns]
                    df_features = new_rows[feat_cols].copy()

                    results = predict_flows(df_features, model, scaler, threshold)
                    print_predictions(results, show_all=False)

                    # Append to predictions.csv
                    write_header = not os.path.exists(out_csv)
                    _save_predictions_csv(results, out_csv) if write_header else \
                        results[["Flow_ID", "Label", "Confidence"]].rename(
                            columns={"Label": "Prediction"}
                        ).to_csv(out_csv, mode="a", header=False, index=False)

                    # Append to live_dataset_predicted.csv
                    write_pred_header = not os.path.exists(predicted_csv)
                    df_pred_feat = validate_features(df_features.copy()).astype("float64")
                    df_pred_feat["Prediction"] = results["Label"].values
                    df_pred_feat["Confidence"] = results["Confidence"].values
                    df_pred_feat.to_csv(
                        predicted_csv, mode="a",
                        header=write_pred_header, index=False
                    )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Watch mode stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DDoS inference on live_dataset.csv → predictions.csv "
            "+ live_dataset_predicted.csv\n"
            "global_model.h5 is loaded read-only — no retraining."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=LIVE_DATASET_CSV,
        help=f"Input live_dataset.csv (default: {LIVE_DATASET_CSV})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Path to global_model.h5 (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--scaler",
        default=SCALER_PATH,
        help="Path to scaler.pkl (must already exist — generated by ddos_main.py / generate_scaler.py)",
    )
    parser.add_argument(
        "--out",
        default=PREDICTIONS_CSV,
        help=f"Output predictions.csv (default: {PREDICTIONS_CSV})",
    )
    parser.add_argument(
        "--predicted",
        default=LIVE_DATASET_PREDICTED_CSV,
        help=f"Output live_dataset_predicted.csv (default: {LIVE_DATASET_PREDICTED_CSV})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Sigmoid threshold for DDoS classification (default: 0.5)",
    )
    parser.add_argument(
        "--all",
        dest="show_all",
        action="store_true",
        default=True,
        help="Print both Normal and DDoS rows (default: True)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll live_dataset.csv continuously for new rows",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=5.0,
        metavar="S",
        help="Poll interval in seconds for --watch mode (default: 5)",
    )
    args = parser.parse_args()

    if args.watch:
        watch_and_predict(
            dataset_csv=args.dataset,
            model_path=args.model,
            scaler_path=args.scaler,
            out_csv=args.out,
            predicted_csv=args.predicted,
            threshold=args.threshold,
            poll_interval=args.poll,
        )
    else:
        run_inference(
            dataset_csv=args.dataset,
            model_path=args.model,
            scaler_path=args.scaler,
            out_csv=args.out,
            predicted_csv=args.predicted,
            threshold=args.threshold,
            show_all=args.show_all,
        )


if __name__ == "__main__":
    main()
