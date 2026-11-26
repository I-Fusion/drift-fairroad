# ACFL Threat Model

## Adversary Capabilities

ACFL is evaluated against adversaries operating in the federated learning layer of avionics/UAS cyber defense systems.

### Attack Types

| Attack | Description | ACFL Countermeasure | Implementation |
|--------|-------------|---------------------|----------------|
| Gradient Poisoning | Malicious clients submit crafted model updates | Layer 1 subspace + temporal scoring | `gradient_scaling` |
| Backdoor Injection | Hidden triggers keyed to adversary inputs | L1 probes + shortlist-veto + post-agg scrub | `backdoor` + data poison |
| Byzantine Updates | Arbitrary or anti-convergent updates | Layer 2 coordinate-wise robust filtering | `byzantine` |
| Label Flipping | Corrupted local training labels | Shortlist-clean geometry + L2 trust | `label_flip` (50% flip) |
| Model Replacement | Full model substitution | Layer 1 trust + attack-scoped hard norm rejection | `model_replacement` (anti-direction scale) |

**Evaluation status (drone profile, 10 seeds):** Byzantine, gradient scaling, and backdoor are fully benchmarked in `drone_journal_benchmark.*` (backdoor ACFL trigger $P$ **0.052** vs FedAvg 0.339 / Krum 0.187). Label flip and model replacement ($n{=}3$) are evaluated via `benchmarks/run_drone_threat_gaps.py` → `drone_threat_gaps_summary.json`. Five-client scaling (20% malicious) covers all five learning-plane attacks via `benchmarks/run_drone_5client_*_benchmark.py`; backdoor uses minimum quorum **four** with shortlist-veto geometry plus post-aggregation trigger scrub (`drone_5client_backdoor_benchmark_summary.json`: ACFL trigger **0.059**, F1 0.760 vs Krum 0.177 / 0.760); label flip uses shortlist-clean geometry (`drone_5client_label_flip_benchmark_summary.json`: ACFL F1 **0.751** vs Krum 0.746); model replacement uses minimum quorum **four** with ACFL hard rejection enabled by default (`drone_5client_model_replacement_benchmark_summary.json`: ACFL 0.607 vs Krum 0.286 mean F1). Layer 3 network-plane spoofing is validated on held-out SITL missions (`scripts/evaluate_layer3_sitl.py`, AUROC **0.96**). **Combined network+learning validation** (poisoned FL checkpoint → Layer 3 on waypoint/GNSS missions) runs via `benchmarks/run_drone_combined_layer3_benchmark.py` → `evaluation_results/combined_layer3/`.

### Adversary Knowledge

Evaluated under two knowledge assumptions:

1. **Gray-box**: Adversary knows FL architecture but not ACFL defense parameters
2. **White-box**: Adversary knows defense mechanisms and adapts attack accordingly

### Attack Configurations

| Parameter | Range | Default |
|-----------|-------|---------|
| Fraction of malicious clients | 10% – 30% | 20% |
| Attack budget per round | 1 – all rounds | Every round |
| Attack type | Poisoning, backdoor, Byzantine | Combined sweep |
| Non-IID severity | Low, medium, high | Medium |

### Operational Constraints (Avionics-Specific)

Unlike general-purpose FL defenses, ACFL accounts for:

- **Bandwidth limits**: Soft-filtering preserves effective sample size vs. hard rejection
- **False-alarm budgets**: Mission-continuity thresholds from drift-fairroad evaluation
- **Non-IID heterogeneity**: Mission-context scoring separates adversarial from legitimate heterogeneity
- **Multi-modal data**: GNSS, ADS-B, MAVLink telemetry, IMU fusion

### Network-Layer + Learning-Layer Combined Attacks

The most demanding evaluation scenario combines:

1. **Learning-plane**: Poisoned FL updates from compromised endpoints
2. **Network-plane**: GNSS spoofing, ADS-B injection, MAVLink waypoint injection, jamming

Layer 3 provides inference-time detection for network-plane attacks using the Layer 1+2 hardened global model.

### Evaluation Metrics

All metrics collected via drift-fairroad evaluation pipeline:

- Detection accuracy (AUROC, F1, Precision, Recall)
- Adversarial robustness (accuracy degradation vs. malicious client fraction)
- False alarm rate (per hour, mission-continuity compliance)
- Time-to-detect (attack onset to first positive)
- Update trust score AUC (trust vs. ground-truth malicious labels)
- Drift and convergence stability
- Communication efficiency (model/update size per round)

### Ground Truth Labels

For trust score AUC evaluation, malicious clients are specified at runtime:

```bash
python run_acfl_system.py --malicious-clients client_2,client_3 ...
```

Trust scores are exported to `checkpoints/acfl_trust_scores.json` for correlation analysis.
