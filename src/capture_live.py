

from __future__ import annotations

import os
import sys
import time
import signal
import argparse
from typing import List, Optional, Tuple
import pandas as pdq 

# ── UTF-8 stdout ───────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure src/ is on path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import get_logger, PCAP_PATH, ensure_dir  # noqa: E402

log = get_logger("capture_live")

# ── Scapy import ───────────────────────────────────────────────────────────────
try:
    from scapy.all import (
        get_if_list,
        sniff,
        PcapWriter,
        IFACES,
    )
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except ImportError as exc:
    log.error("Scapy is not installed.  Run: pip install scapy")
    raise SystemExit(1) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Duration prompt
# ─────────────────────────────────────────────────────────────────────────────
def prompt_duration() -> int:

    while True:
        try:
            raw = input("\nEnter capture duration in minutes: ").strip()
            minutes = float(raw)
            if minutes <= 0:
                print("  [!] Please enter a number greater than 0.")
                continue
            seconds = max(1, int(minutes * 60))
            print(f"  [✓] Will capture for {minutes} min ({seconds} s)\n")
            return seconds
        except ValueError:
            print("  [!] Invalid input — please enter a number (e.g. 5 or 0.5).")
        except (EOFError, KeyboardInterrupt):
            print("\n  [!] Cancelled.")
            raise SystemExit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Interface detection
# ─────────────────────────────────────────────────────────────────────────────
_PREFERRED_KEYWORDS = ("wi-fi", "wifi", "wireless", "ethernet", "eth", "local area")


def _score_iface(name: str) -> int:
    lower = name.lower()
    if "loopback" in lower or lower.startswith("lo"):
        return 0
    if "vmware" in lower or "virtualbox" in lower or "vethernet" in lower:
        return 1
    for i, kw in enumerate(reversed(_PREFERRED_KEYWORDS)):
        if kw in lower:
            return 10 + i
    return 2


def auto_detect_interface() -> str:
    """Return the highest-priority active network interface (Wi-Fi > Ethernet)."""
    candidates: List[Tuple[int, str]] = []

    try:
        for iface_obj in IFACES.values():
            name = getattr(iface_obj, "name", None) or getattr(iface_obj, "description", None)
            if name:
                candidates.append((_score_iface(name), name))
    except Exception:
        pass

    for name in get_if_list():
        candidates.append((_score_iface(name), name))

    if not candidates:
        raise RuntimeError(
            "No network interfaces found. "
            "Ensure Npcap is installed (https://npcap.com/)."
        )

    candidates.sort(key=lambda t: (-t[0], t[1]))
    best = candidates[0][1]
    log.info("Detected interfaces (scored): %s", candidates)
    log.info("Selected interface: '%s'", best)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Packet writer
# ─────────────────────────────────────────────────────────────────────────────
_BPF_FILTER = "tcp or udp"


class _StreamWriter:
    """PcapWriter wrapper — streams packets directly to disk, zero RAM."""

    def __init__(self, path: str) -> None:
        ensure_dir(path)
        self._writer    = PcapWriter(path, append=False, sync=True)
        self._pkt_count = 0
        self._byte_count = 0

    def write(self, pkt) -> None:
        self._writer.write(pkt)
        self._pkt_count  += 1
        self._byte_count += len(pkt)
        if self._pkt_count % 5000 == 0:
            log.info(
                "Captured %d packets  |  %.2f KB written",
                self._pkt_count,
                self._byte_count / 1024,
            )

    def close(self) -> None:
        self._writer.close()

    @property
    def packet_count(self) -> int:
        return self._pkt_count

    @property
    def byte_count(self) -> int:
        return self._byte_count


# ─────────────────────────────────────────────────────────────────────────────
# Capture logic
# ─────────────────────────────────────────────────────────────────────────────
_stop_capture = False


def _sigint_handler(signum, frame):   # noqa: ARG001
    global _stop_capture
    _stop_capture = True
    log.info("Interrupt received — stopping capture …")


def _stop_fn(_pkt) -> bool:
    return _stop_capture


def start_capture(
    iface: Optional[str] = None,
    out_path: str = PCAP_PATH,
    duration: Optional[int] = None,
    bpf_filter: str = _BPF_FILTER,
) -> str:
    """
    Capture live packets and stream them to *out_path*.

    Args:
        iface:      Interface name.  Auto-detected when None.
        out_path:   Destination .pcap file.
        duration:   Seconds to capture.  None = run until CTRL-C.
        bpf_filter: BPF filter string.

    Returns:
        Absolute path to the written .pcap file.
    """
    global _stop_capture
    _stop_capture = False

    if iface is None:
        iface = auto_detect_interface()

    log.info("Starting capture on interface '%s'", iface)
    log.info("Output file : %s", out_path)
    log.info("BPF filter  : '%s'", bpf_filter)
    if duration:
        mins = duration / 60
        log.info("Duration    : %d s (%.2f min)", duration, mins)
    else:
        log.info("Duration    : unlimited — press CTRL-C to stop")

    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    writer   = _StreamWriter(out_path)
    start_ts = time.time()

    try:
        sniff(
            iface=iface,
            filter=bpf_filter,
            prn=writer.write,
            store=False,
            stop_filter=_stop_fn,
            timeout=duration,
        )
    except PermissionError:
        log.error(
            "Permission denied.  Run as Administrator and ensure "
            "Npcap is installed (https://npcap.com/)."
        )
        raise
    except OSError as exc:
        log.error("Capture failed on '%s': %s", iface, exc)
        raise
    finally:
        writer.close()
        elapsed = time.time() - start_ts
        log.info(
            "Capture finished.  Packets: %d  |  Bytes: %d  |  Time: %.1f s",
            writer.packet_count,
            writer.byte_count,
            elapsed,
        )

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Automated pipeline (capture → extract → predict)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    duration_seconds: int,
    iface: Optional[str] = None,
    pcap_path: str = PCAP_PATH,
    capture_only: bool = False,
) -> None:
    """
    Full automated pipeline:
        1. Capture live traffic → traffic.pcap
        2. Extract features     → live_dataset.csv
        3. Run inference        → predictions.csv + live_dataset_predicted.csv

    Args:
        duration_seconds: How long to capture.
        iface:            Network interface (auto-detected if None).
        pcap_path:        Output .pcap path.
        capture_only:     If True, stop after capture (skip steps 2 & 3).
    """
    # ── Step 1: Capture ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1/3 — Live Traffic Capture")
    print("=" * 60)
    start_capture(iface=iface, out_path=pcap_path, duration=duration_seconds)
    print(f"\n  [✓] traffic.pcap saved → {pcap_path}\n")

    if capture_only:
        print("  [!] --capture-only flag set.  Stopping after capture.")
        return

    # ── Step 2: Feature extraction ────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 2/3 — Feature Extraction (CICIDS2017-style)")
    print("=" * 60)
    try:
        from feature_extractor import extract_from_pcap
        from utils import LIVE_DATASET_CSV
        df_features = extract_from_pcap(pcap_path=pcap_path, out_csv=LIVE_DATASET_CSV)
        print(f"\n  [✓] live_dataset.csv saved → {LIVE_DATASET_CSV}")
        print(f"      Flows extracted: {len(df_features)}\n")
    except Exception as exc:
        log.error("Feature extraction failed: %s", exc)
        raise

    if df_features.empty:
        print("  [!] No usable flows extracted — cannot run prediction.")
        return

    # ── Step 3: Inference ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 3/3 — Model Inference (global_model.h5)")
    print("=" * 60)
    try:
        from live_predict import run_inference
        from utils import LIVE_DATASET_CSV, PREDICTIONS_CSV
        run_inference(
            dataset_csv=LIVE_DATASET_CSV,
            out_csv=PREDICTIONS_CSV,
            show_all=True,
        )
    except Exception as exc:
        log.error("Inference failed: %s", exc)
        raise

    print("=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live DDoS Detection Pipeline  (capture → extract → predict)\n"
            "Run without arguments for interactive duration prompt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="MINUTES",
        help="Capture duration in minutes (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--iface",
        default=None,
        help="Network interface name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out",
        default=PCAP_PATH,
        help=f"Output .pcap file (default: {PCAP_PATH})",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Stop after capture — do not run feature extraction or prediction",
    )
    args = parser.parse_args()

    # Determine duration
    if args.duration is not None:
        if args.duration <= 0:
            print("[!] --duration must be > 0")
            raise SystemExit(1)
        duration_seconds = max(1, int(args.duration * 60))
        print(f"\n  Capture duration: {args.duration} min ({duration_seconds} s)")
    else:
        duration_seconds = prompt_duration()

    run_pipeline(
        duration_seconds=duration_seconds,
        iface=args.iface,
        pcap_path=args.out,
        capture_only=args.capture_only,
    )


if __name__ == "__main__":
    main()
