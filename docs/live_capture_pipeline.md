# Live Traffic Capture & Real-Time DDoS Detection Pipeline

## 1. Overview & Purpose
This subsystem bridges **offline-trained federated models** and **real-world production networks**. It provides an automated, end-to-end pipeline that sniffs raw network traffic, reconstructs network conversations into bidirectional statistical flows, normalizes them against the training baseline, and executes real-time inference using the trained Federated Global Model (`global_model.h5`).

---

## 2. End-to-End Pipeline Architecture

```
                       [ Live Network Interface (Wi-Fi / Ethernet) ]
                                            │
                                            │ (Npcap / Scapy Sniffer)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Packet Capture (capture_live.py)                                               │
│ Streams raw layer-3/4 packets to disk with zero-RAM footprint via PcapWriter.          │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │ Outputs: traffic.pcap
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Feature Extraction (feature_extractor.py)                                      │
│ Groups packets into bidirectional flows by 5-tuple key:                                │
│   (Source IP, Destination IP, Source Port, Destination Port, Protocol)                 │
│ Calculates 22 statistical temporal & volumetric features matching CICIDS2017.         │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │ Outputs: live_dataset.csv (22 features + Label="Unknown")
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Model Inference (live_predict.py)                                              │
│ 1. Loads pre-fitted scaler.pkl (Strict transform only — no refitting).                 │
│ 2. Loads trained global_model.h5 (Read-only — weights unchanged).                      │
│ 3. Applies Sigmoid classification threshold (Default: 0.50).                           │
│ 4. Assigns: Normal (p < 0.5) vs DDoS (p >= 0.5) with exact confidence scores.          │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ├──► predictions.csv (Flow_ID, Prediction, Confidence)
                                    └──► live_dataset_predicted.csv (22 features + Labels)
```

---

## 3. Core Modules & Technical Specifications

### A. `capture_live.py` (Packet Capture Engine)
* **Technology**: Built on `Scapy` with the `Npcap` packet capture driver for Windows.
* **Mechanism**:
  * Auto-detects active physical interfaces (prioritizes active Wi-Fi and Ethernet adapters).
  * Applies Berkeley Packet Filters (`bpf_filter = "tcp or udp"`).
  * Uses streaming disk writes (`PcapWriter(sync=True)`) to ensure packets are streamed directly to disk without memory accumulation.
  * Controlled by a duration timer (e.g., 30s, 60s) or manual interrupt (`Ctrl+C`).
* **Artifact Generated**: `traffic.pcap`

### B. `feature_extractor.py` (Flow Reconstruction & Feature Engineering)
* **Mechanism**:
  * Parses raw `.pcap` packets sequentially.
  * Reconstructs packet streams into bidirectional flows using bidirectional hashing on `(src_ip, dst_ip, src_port, dst_port, protocol)`.
  * Manages active flows using flow timeouts (`FLOW_IDLE_TIMEOUT = 120s`, `FLOW_ACTIVE_TIMEOUT = 600s`).
  * Discards micro-noise flows with fewer than 2 packets (`MIN_PACKETS = 2`).
* **The 22 Statistical Features Extracted**:
  1. `Flow Duration`: Total duration of the network flow in microseconds.
  2. `Total Fwd Packets`: Count of packets traveling in the forward direction (source → destination).
  3. `Total Backward Packets`: Count of packets traveling in the reverse direction (destination → source).
  4. `Total Length of Fwd Packets`: Sum of payload bytes sent forward.
  5. `Total Length of Bwd Packets`: Sum of payload bytes sent backward.
  6. `Fwd Packet Length Max`: Largest forward packet size (bytes).
  7. `Fwd Packet Length Min`: Smallest forward packet size (bytes).
  8. `Fwd Packet Length Mean`: Average forward packet size.
  9. `Bwd Packet Length Max`: Largest backward packet size.
  10. `Bwd Packet Length Min`: Smallest backward packet size.
  11. `Bwd Packet Length Mean`: Average backward packet size.
  12. `Flow Bytes/s`: Transmission rate in bytes per second.
  13. `Flow Packets/s`: Transmission rate in packets per second.
  14. `Flow IAT Mean`: Mean Inter-Arrival Time between consecutive packets.
  15. `Flow IAT Std`: Standard deviation of Inter-Arrival Times.
  16. `Fwd IAT Total`: Cumulative time between forward packets.
  17. `Bwd IAT Total`: Cumulative time between backward packets.
  18. `Packet Length Mean`: Overall mean length of all packets in the flow.
  19. `Packet Length Std`: Standard deviation of all packet lengths.
  20. `Average Packet Size`: Mean size across all transmitted frames.
  21. `Avg Fwd Segment Size`: Average forward TCP/UDP segment size.
  22. `Avg Bwd Segment Size`: Average backward TCP/UDP segment size.
* **Artifact Generated**: `live_dataset.csv`

### C. `live_predict.py` (Inference & Decision Engine)
* **Strict Guarantees**: Operates strictly in **read-only inference mode**. It never calls `fit()`, `fit_transform()`, or updates model weights.
* **Mechanism**:
  * Loads `scaler.pkl` (StandardScaler fitted during federated training on the CICIDS2017 dataset).
  * Applies `scaler.transform(X)` to scale live features to zero mean and unit variance.
  * Loads `docs/global_model.h5` and computes sigmoid output $P(\text{DDoS} \mid X) \in [0, 1]$.
  * Renders a concise terminal table and exports full audit files.
* **Artifacts Generated**:
  * `predictions.csv`: Lightweight lookup table (`Flow_ID, Prediction, Confidence`).
  * `live_dataset_predicted.csv`: Full tabular export containing all 22 features, prediction labels, and confidence values.

### D. `simulate_ddos.py` (Controlled Demonstration Traffic Generator)
* **Purpose**: Generates live, non-destructive network traffic with statistical signatures that mimic volumetric DDoS attacks for real-time testing.
* **Mechanism**:
  * Broadcasts rapid bursts of UDP packets targeting port `9999` across the local subnet (`255.255.255.255`).
  * Each burst allocates a unique ephemeral source port, generating a distinct flow entry in the sniffer.
  * Burst parameters are calibrated (7 packets per flow with 10ms intervals) to match the volumetric threshold expected by the CICIDS2017 trained model.

---

## 4. How to Run the Complete Live Pipeline

### Scenario 1: Ambient Normal Traffic Inspection
Captures genuine background computer traffic (web browsing, DNS queries, streaming) and evaluates it:
```powershell
cd D:\fraud-detection-federated
python src/capture_live.py
# Enter duration: e.g., 0.5 (30 seconds)
```
*Expected Result: 100% Normal traffic classification with confidence < 0.04.*

---

### Scenario 2: Active Threat Demonstration (Multi-Terminal)
Demonstrates live interception and classification of an active attack concurrently:

1. **Terminal 1 (Sniffer & Classifier)**:
   ```powershell
   cd D:\fraud-detection-federated
   python src/capture_live.py
   # Enter duration: 1 (60 seconds)
   ```

2. **Terminal 2 (Attack Generator — run while Terminal 1 is capturing)**:
   ```powershell
   cd D:\fraud-detection-federated
   python src/simulate_ddos.py --continuous --interval 2
   # Let it emit 3–5 bursts, then press Ctrl+C
   ```

*Expected Result: Terminal 1 intercepts the live packets, reconstructs the flows, and outputs a split table classifying attack bursts as `DDoS` (confidence ~0.897) and background traffic as `Normal`.*
