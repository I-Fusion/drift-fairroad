# Drone Resource Budget (ACFL Classification)

Assumptions for a **typical small UAS** with a companion computer (e.g. Raspberry Pi 4 / Jetson Nano class) participating in federated learning over a **lossy, low-bandwidth** GCS or mesh link.

## Design targets

| Resource | Target | How we enforce it |
|----------|--------|-------------------|
| **RAM** | ≤512 MB for ML | TinyPacketCNN (~14k params); `BATCH_SIZE=16`; 3 concurrent FL clients max on ground station; single client onboard |
| **Storage** | ≤32 MB checkpoints | `SERVER_CHECKPOINT_KEEP_LAST=3`; `SAVE_CLIENT_CHECKPOINTS=False` |
| **Comms** | ≤100 KB per FL round upload | fp16 weights + gzip (`fl_comm.py`); ~54k→~14k params |
| **Compute** | Bursty local training | `WINDOWS_PER_ROUND=16`, `LOCAL_EPOCHS=2` — short bursts between radio uploads |

## Config profile

Use `config_classification_drone.py`:

- Model: `models.tiny_packet_cnn.TinyPacketCNN`
- `NUM_CLIENTS=3`, `MIN_CLIENTS=2`
- `DEFAULT_FL_ROUNDS=12` (set via `--max-fl-rounds 12` or `FL_MAX_ROUNDS`)
- `WEIGHT_TRANSFER_FP16=True`, `WEIGHT_TRANSFER_GZIP=True`

## Data collection

`scripts/collect_drone_missions.sh` adds missions sized for flash budgets:

- 6 missions default (~3–7 MB processed CSV each)
- Concurrency 2 (ground station); onboard drone runs one mission at a time
- Missions merge into **agents** via `scripts/build_agent_mission_dataset.py` (not held-out agents)

## Estimating your deployment

```bash
./acfl_env/bin/python scripts/estimate_drone_resources.py
./acfl_env/bin/python scripts/estimate_drone_resources.py --config config_classification_drone --compare config_classification
```

## Not in scope (future work)

- True onboard quantization (INT8) and TFLite export
- Delta-only uploads (client − global) for further comms savings
- Streaming window loader for missions that exceed RAM
