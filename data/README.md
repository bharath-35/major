# Data Directory

This folder contains network traffic data used for DDoS detection via Federated Learning.

---

## 📄 Dataset

### `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

Source: **CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Systems 2017  
Reference: https://www.unb.ca/cic/datasets/ids-2017.html

- **Rows:** ~225,000 network flow records
- **Columns:** 85 (80+ numeric flow features + Label)
- **Label column:** `Label` — values: `BENIGN` or `DDoS`
- **Attack type:** DDoS (HTTP, UDP, TFTP flood traffic)

---

## Features Used for Training

The model uses 22 numeric network flow features including:
`Flow Duration`, `Total Fwd/Bwd Packets`, `Packet Length Mean/Std`,
`Flow Bytes/s`, `Flow Packets/s`, `Flow IAT Mean/Std`, etc.

---

> ⚠️ No personally identifiable information (PII) is present. This is network flow metadata only.
