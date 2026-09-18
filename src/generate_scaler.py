

import os
import sys
import argparse

# ── UTF-8 stdout ───────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure src/ on path ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import (
    TRAINING_CSV_PATH,
    SCALER_PATH,
    generate_scaler,
    get_logger,
)

log = get_logger("generate_scaler")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Training-phase utility: fit StandardScaler on CICIDS2017 CSV → scaler.pkl\n"
            "Run ONCE after training.  The live testing pipeline will then load this file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=TRAINING_CSV_PATH,
        help=f"Path to CICIDS2017 training CSV (default: {TRAINING_CSV_PATH})",
    )
    parser.add_argument(
        "--out",
        default=SCALER_PATH,
        help=f"Destination path for scaler.pkl (default: {SCALER_PATH})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        log.error("Training CSV not found: %s", args.csv)
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("  Scaler Generation  (Training Phase Setup)")
    print("=" * 60)

    scaler = generate_scaler(csv_path=args.csv, save_path=args.out)

    print(f"\n  [✓] scaler.pkl saved → {args.out}")
    print(f"      Features : {scaler.n_features_in_}")
    print(f"      Samples  : {scaler.n_samples_seen_:,}")
    print("\n  The live testing pipeline will load this file for inference.")
    print("  DO NOT run this script again unless the model has been retrained.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
