"""
DDoS Detection using Federated Learning on CICIDS2017 Dataset.
Simulates FedAvg across 3 client nodes to detect DDoS vs BENIGN traffic.
Run via: python ddos_main.py
"""

import os
import sys

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")           # no display needed — saves to file
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# ── Experiment flag ───────────────────────────────────────────────────────────
# Set True to distribute data unevenly (DDoS-heavy / balanced / BENIGN-heavy)
NON_IID = True

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_DIR          = "../docs"
CONVERGENCE_PLOT = os.path.join(OUT_DIR, "convergence_plot.png")
ROC_PLOT         = os.path.join(OUT_DIR, "roc_curve.png")
MODEL_PATH       = os.path.join(OUT_DIR, "global_model.h5")

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Total", "Bwd IAT Total", "Packet Length Mean", "Packet Length Std",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size"
]

LABEL_COL  = "Label"
DDOS_LABEL = "DDoS"


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_cicids2017(file_path):
    """Load CICIDS2017 CSV, strip whitespace from column names and labels."""
    print(f"[✓] Loading {file_path} ...")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df[LABEL_COL] = df[LABEL_COL].str.strip()
    print(f"[✓] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ── Labelling ─────────────────────────────────────────────────────────────────
def label_ddos(df):
    """Binary label: 1 = DDoS, 0 = BENIGN. Drop any other labels."""
    df = df[df[LABEL_COL].isin([DDOS_LABEL, "BENIGN"])].copy()
    df["is_ddos"] = (df[LABEL_COL] == DDOS_LABEL).astype(int)
    ddos_n = int(df["is_ddos"].sum())
    norm_n = int((df["is_ddos"] == 0).sum())
    print(f"[✓] Labeled  →  DDoS: {ddos_n:,}  |  BENIGN: {norm_n:,}  "
          f"|  DDoS rate: {ddos_n / len(df) * 100:.1f}%")
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df, scaler=None, fit_scaler=True):
    """Select features, replace inf/nan, scale."""
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = df["is_ddos"].values.astype(np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    return X, y, scaler


# ── IID Split ────────────────────────────────────────────────────────────────
def split_clients_iid(X, y, num_clients=3):
    """Split data equally across clients (IID)."""
    n, size = len(X), len(X) // num_clients
    clients = []
    for i in range(num_clients):
        start = i * size
        end = (i + 1) * size if i < num_clients - 1 else n
        clients.append((X[start:end], y[start:end]))
    return clients


# ── Non-IID Split ─────────────────────────────────────────────────────────────
def split_clients_non_iid(X, y, num_clients=3):
    """
    Non-IID distribution across num_clients:
      Client 1   → ~80% DDoS-heavy
      Client N   → ~80% BENIGN-heavy
      Middle clients → linearly interpolated ratios
    The DDoS ratio is linearly spaced from 0.8 down to 0.2.
    """
    ddos_idx   = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]
    np.random.seed(42)
    np.random.shuffle(ddos_idx)
    np.random.shuffle(benign_idx)

    n_ddos, n_benign = len(ddos_idx), len(benign_idx)
    shard = min(n_ddos, n_benign) // num_clients

    # Linearly space DDoS ratios from 0.8 (DDoS-heavy) to 0.2 (BENIGN-heavy)
    if num_clients == 1:
        ddos_ratios = [0.5]
    else:
        ddos_ratios = np.linspace(0.8, 0.2, num_clients).tolist()

    clients = []
    ddos_ptr, benign_ptr = 0, 0

    for ratio in ddos_ratios:
        n_d = int(shard * ratio)
        n_b = shard - n_d
        # Guard against running out of samples
        n_d = min(n_d, n_ddos - ddos_ptr)
        n_b = min(n_b, n_benign - benign_ptr)
        c_idx = np.concatenate([
            ddos_idx[ddos_ptr: ddos_ptr + n_d],
            benign_idx[benign_ptr: benign_ptr + n_b]
        ])
        ddos_ptr   += n_d
        benign_ptr += n_b
        np.random.shuffle(c_idx)
        clients.append((X[c_idx], y[c_idx]))

    return clients


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(input_dim):
    """Build model using explicit Input() layer (avoids Keras warning)."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),          # Fix #1: explicit Input layer
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ── FedAvg Helpers ─────────────────────────────────────────────────────────────
def local_train(weights, X, y, epochs=3, batch_size=256):
    """Train locally; returns (weights, num_samples, final_epoch_loss)."""
    model = build_model(X.shape[1])
    model.set_weights(weights)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    client_loss = history.history["loss"][-1]   # loss at final local epoch
    return model.get_weights(), len(X), client_loss


def federated_average(all_weights, all_sizes):
    total = sum(all_sizes)
    return [
        sum(w * (s / total) for w, s in zip(layers, all_sizes))
        for layers in zip(*all_weights)
    ]


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate_model(model, X, y):
    loss, acc = model.evaluate(X, y, verbose=0)
    y_pred = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    tp = int(((y_pred == 1) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return dict(loss=loss, acc=acc, precision=precision, recall=recall,
                f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


# ── Plot 1: Per-Client Loss per Round ─────────────────────────────────────────
def plot_client_losses(client_losses_per_round, save_path):
    """
    client_losses_per_round: list of lists
        outer index = round (0-based), inner index = client index
        e.g. [[c1_loss_r1, c2_loss_r1, ...], [c1_loss_r2, ...], ...]
    """
    num_rounds  = len(client_losses_per_round)
    num_clients = len(client_losses_per_round[0])
    rounds      = list(range(1, num_rounds + 1))

    # Distinct colour palette for clients
    client_colors  = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
                      "#f39c12", "#1abc9c", "#e67e22", "#34495e"]
    client_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for c in range(num_clients):
        losses = [client_losses_per_round[r][c] for r in range(num_rounds)]
        color  = client_colors[c % len(client_colors)]
        marker = client_markers[c % len(client_markers)]
        ax.plot(rounds, losses,
                marker=marker, linewidth=2.2, markersize=7,
                color=color, label=f"Client {c + 1}",
                alpha=0.9)
        # Annotate final loss value
        ax.annotate(
            f"{losses[-1]:.4f}",
            xy=(rounds[-1], losses[-1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=color, va="center"
        )

    ax.set_title("Federated Learning — Per-Client Loss per Round",
                 fontsize=14, fontweight="bold", color="white", pad=14)
    ax.set_xlabel("Communication Round", fontsize=11, color="#aaaaaa")
    ax.set_ylabel("Training Loss (Final Local Epoch)", fontsize=11, color="#aaaaaa")
    ax.set_xticks(rounds)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, alpha=0.2, color="#888888", linestyle="--")
    legend = ax.legend(title="Clients", title_fontsize=10, fontsize=9,
                       facecolor="#1a1a2e", edgecolor="#444466",
                       labelcolor="white", loc="upper right")
    legend.get_title().set_color("#aaaaaa")

    # Round info annotations on x-axis
    for rnd in rounds:
        ax.axvline(x=rnd, color="#444466", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Per-client loss plot saved → {save_path}")


# ── Plot 2: Average Federated Loss & Accuracy per Round ───────────────────────
def plot_avg_convergence(round_losses, round_accs, save_path):
    """Two-panel chart: average global loss and accuracy across all rounds."""
    rounds = list(range(1, len(round_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.grid(True, alpha=0.2, color="#888888", linestyle="--")
        ax.set_xticks(rounds)

    # — Loss panel —
    ax1.plot(rounds, round_losses, "o-", color="#e74c3c",
             linewidth=2.5, markersize=7, label="Avg Global Loss")
    ax1.fill_between(rounds, round_losses, alpha=0.15, color="#e74c3c")
    ax1.set_title("Average Federated Training — Loss",
                  fontsize=13, fontweight="bold", color="white", pad=12)
    ax1.set_xlabel("Communication Round", fontsize=11, color="#aaaaaa")
    ax1.set_ylabel("Loss", fontsize=11, color="#aaaaaa")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="white", fontsize=9)
    # Annotate min loss
    min_loss_rnd = round_losses.index(min(round_losses)) + 1
    ax1.annotate(f"Min: {min(round_losses):.4f}",
                 xy=(min_loss_rnd, min(round_losses)),
                 xytext=(8, 10), textcoords="offset points",
                 fontsize=9, color="#e74c3c",
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))

    # — Accuracy panel —
    accs_pct  = [a * 100 for a in round_accs]
    min_acc   = min(accs_pct)
    max_acc   = max(accs_pct)
    y_margin  = max(0.3, (max_acc - min_acc) * 0.5)   # at least 0.3% headroom
    ax2.set_ylim(min_acc - y_margin, min(100.1, max_acc + y_margin))

    ax2.plot(rounds, accs_pct, "o-", color="#2ecc71",
             linewidth=2.5, markersize=7, label="Avg Global Accuracy")
    # fill only between the min value and the curve so changes are visible
    ax2.fill_between(rounds, accs_pct, min_acc - y_margin,
                     alpha=0.2, color="#2ecc71")
    ax2.set_title("Average Federated Training — Accuracy",
                  fontsize=13, fontweight="bold", color="white", pad=12)
    ax2.set_xlabel("Communication Round", fontsize=11, color="#aaaaaa")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, color="#aaaaaa")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="white", fontsize=9)
    # Annotate max accuracy
    max_acc_rnd = accs_pct.index(max_acc) + 1
    ax2.annotate(f"Max: {max_acc:.2f}%",
                 xy=(max_acc_rnd, max_acc),
                 xytext=(8, 10), textcoords="offset points",
                 fontsize=9, color="#2ecc71",
                 arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.2))
    # Annotate min accuracy (first round — shows improvement)
    min_acc_rnd = accs_pct.index(min_acc) + 1
    ax2.annotate(f"Start: {min_acc:.2f}%",
                 xy=(min_acc_rnd, min_acc),
                 xytext=(8, -14), textcoords="offset points",
                 fontsize=9, color="#f39c12",
                 arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.2))

    fig.suptitle("Federated Learning — Average Global Convergence",
                 fontsize=15, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Average convergence plot saved → {save_path}")


# ── ROC Curve ─────────────────────────────────────────────────────────────────
def plot_roc(model, X_test, y_test, save_path):
    y_scores = model.predict(X_test, verbose=0).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#3498db", lw=2,
             label=f"Global Model  (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve — DDoS Detection (Global Model)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] ROC curve saved → {save_path}  (AUC = {roc_auc:.4f})")
    return roc_auc


# ── Output paths (extended) ───────────────────────────────────────────────────
CLIENT_LOSS_PLOT = os.path.join(OUT_DIR, "client_loss_plot.png")
AVG_CONV_PLOT    = os.path.join(OUT_DIR, "avg_convergence_plot.png")


# ── Main Entry ────────────────────────────────────────────────────────────────
def run_ddos_detection(data_path, num_clients=3, num_rounds=10,
                       test_size=0.2, non_iid=NON_IID):

    print("\n" + "=" * 60)
    print("   DDoS Detection — Federated Learning (CICIDS2017)")
    mode_tag = "Non-IID" if non_iid else "IID"
    print(f"   Distribution mode: {mode_tag}")
    print("=" * 60)

    # Load & label
    df = load_cicids2017(data_path)
    df = label_ddos(df)

    # Preprocess
    X, y, scaler = preprocess(df, fit_scaler=True)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[✓] Split  →  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Distribute across clients
    if non_iid:
        clients = split_clients_non_iid(X_train, y_train, num_clients)
        print(f"\n[✓] Non-IID data distributed across {num_clients} federated clients:")
    else:
        clients = split_clients_iid(X_train, y_train, num_clients)
        print(f"\n[✓] IID data distributed across {num_clients} federated clients:")

    for i, (cx, cy) in enumerate(clients):
        print(f"    Client {i+1}: {len(cx):,} samples | DDoS rate: {cy.mean()*100:.1f}%")

    # FedAvg training loop
    global_model   = build_model(X_train.shape[1])
    global_weights = global_model.get_weights()

    print(f"\n{'─'*60}")
    print(f"  Starting FedAvg — {num_rounds} communication rounds  [{mode_tag}]")
    print(f"{'─'*60}")

    round_losses, round_accs    = [], []
    client_losses_per_round     = []    # [ [c1_loss, c2_loss, ...], ... ] per round
    last_local_weights          = []    # keep last round's per-client weights

    for rnd in range(1, num_rounds + 1):
        local_wts, local_sizes, local_losses = [], [], []
        for cx, cy in clients:
            lw, ls, cl = local_train(global_weights, cx, cy)
            local_wts.append(lw)
            local_sizes.append(ls)
            local_losses.append(cl)
        global_weights = federated_average(local_wts, local_sizes)
        last_local_weights      = local_wts          # save for comparison
        client_losses_per_round.append(local_losses)  # record per-client losses

        eval_m = build_model(X_test.shape[1])
        eval_m.set_weights(global_weights)
        loss, acc = eval_m.evaluate(X_test, y_test, verbose=0)
        round_losses.append(loss)
        round_accs.append(acc)
        client_loss_str = "  ".join(
            f"C{i+1}: {cl:.4f}" for i, cl in enumerate(local_losses)
        )
        print(f"  Round {rnd:2d}/{num_rounds} │ Loss: {loss:.4f} │ Acc: {acc*100:.2f}%  │  [{client_loss_str}]")

    # ── Final global model ────────────────────────────────────────────────────
    final_model = build_model(X_test.shape[1])
    final_model.set_weights(global_weights)
    m = evaluate_model(final_model, X_test, y_test)

    print(f"\n{'─'*60}")
    print("  Final Global Model Evaluation  (Test Set)")
    print(f"{'─'*60}")
    print(f"  Loss       : {m['loss']:.4f}")
    print(f"  Accuracy   : {m['acc']*100:.2f}%")
    print(f"  Precision  : {m['precision']*100:.2f}%")
    print(f"  Recall     : {m['recall']*100:.2f}%")
    print(f"  F1-Score   : {m['f1']*100:.2f}%")
    print(f"\n  True Positives  (DDoS caught)       : {m['tp']:,}")
    print(f"  True Negatives  (BENIGN correct)    : {m['tn']:,}")
    print(f"  False Positives (False alarms)      : {m['fp']:,}")
    print(f"  False Negatives (Missed DDoS)       : {m['fn']:,}")

    # ── Local vs Global comparison ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Local vs Global Model Comparison  (Test Set Accuracy)")
    print(f"{'─'*60}")
    print(f"  {'Model':<12} {'Accuracy':>10}")
    print(f"  {'─'*22}")
    for i, lw in enumerate(last_local_weights):
        lm = build_model(X_test.shape[1])
        lm.set_weights(lw)
        _, local_acc = lm.evaluate(X_test, y_test, verbose=0)
        print(f"  {'Client ' + str(i+1):<12} {local_acc*100:>9.2f}%")
    print(f"  {'─'*22}")
    print(f"  {'Global':<12} {m['acc']*100:>9.2f}%")

    # ── Save global model ─────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    final_model.save(MODEL_PATH)
    print(f"\n[✓] Global model saved → {MODEL_PATH}")

    # ── Per-client loss plot ──────────────────────────────────────────────────
    plot_client_losses(client_losses_per_round, CLIENT_LOSS_PLOT)

    # ── Average federated convergence plot ───────────────────────────────────
    plot_avg_convergence(round_losses, round_accs, AVG_CONV_PLOT)

    # ── ROC curve + AUC ───────────────────────────────────────────────────────
    roc_auc = plot_roc(final_model, X_test, y_test, ROC_PLOT)

    print(f"\n  AUC Score  : {roc_auc:.4f}")
    print(f"\n{'='*60}")
    print("  DDoS Detection Complete!")
    print(f"{'='*60}\n")
