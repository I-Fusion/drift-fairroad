"""
ACFL configuration parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ACFLConfig:
    """Configuration for Adversarially-Conditioned Federated Learning."""

    # Layer 1: Counterfactual Update Validation
    enable_layer1: bool = True
    subspace_dim: int = 16
    subspace_anomaly_threshold: float = 2.5
    # With only two participants the median norm is inflated toward the
    # attacker, so a 10x gradient-scaling attack yields ratio ~1.82 (missed
    # at 2.0). Use a lower bar for pairwise rounds.
    norm_ratio_threshold: float = 2.0
    norm_ratio_threshold_two_clients: float = 1.5
    subspace_weight: float = 0.50
    temporal_consistency_weight: float = 0.20
    counterfactual_weight: float = 0.30
    min_trust_weight: float = 0.05
    soft_filter_temperature: float = 1.0

    # Counterfactual harm scoring (soft, confidence-based)
    # Per-scenario detection-confidence drops smaller than the tolerance are
    # treated as benign drift; harm beyond it is amplified by the gain.
    counterfactual_tolerance: float = 0.05
    counterfactual_gain: float = 6.0

    # Layer 2: Byzantine-Tolerant Trust-Weighted Aggregation
    enable_layer2: bool = True
    trim_fraction: float = 0.1
    coordinate_deviation_threshold: float = 3.0
    mission_context_weight: float = 0.4
    statistical_consistency_weight: float = 0.6
    benign_round_history: int = 5
    # Exponent applied to the combined trust score before normalization.
    # Values >1 sharpen separation when Layer 1/2 already identify a weak
    # client, while keeping all clients represented above min_trust_weight.
    trust_weight_power: float = 1.0

    # Aggregation guard: hold the global model when participation is too low
    # or a sole participant looks suspicious (prevents late-round poisoning).
    aggregation_guard_enabled: bool = True
    min_aggregation_participants: int = 2
    single_participant_trust_threshold: float = 0.95

    # Pairwise inversion guard: in 2-client rounds, downweight a participant
    # whose robustness is high while a more trusted peer was soft-filtered.
    pairwise_inversion_guard_enabled: bool = True
    pairwise_trust_margin: float = 0.05
    pairwise_robustness_asymmetry: float = 0.35
    pairwise_high_robustness_floor: float = 0.99
    pairwise_suspicious_weight_multiplier: float = 0.4

    # Hard-reject client updates with extreme norms before aggregation.
    update_norm_hard_rejection_enabled: bool = False
    update_norm_hard_rejection_ratio: float = 4.0
    update_norm_hard_rejection_ratio_two_clients: float = 2.5
    update_norm_hard_rejection_deviation: float = 4.0
    # Attack-scoped hard rejection for model_replacement. Default on for
    # 5-client (dense) rounds; if rejection would leave < min participants,
    # defense withholds and keeps soft trust instead of holding the global model.
    update_norm_hard_rejection_for_model_replacement: bool = True

    # Pre-Layer-2 soft cap on Layer 1 trust from peer-aware update statistics.
    robust_norm_soft_cap_enabled: bool = True

    # When Layer 2 is on, avoid compounding L1 penalties on clients Layer 2
    # already flags via peer robustness (fixes Byzantine benign-client crush).
    layer1_layer2_gating_enabled: bool = True
    layer1_layer2_gating_robustness_floor: float = 0.5
    # Adaptive arbitration: per-client hinge blends raw L1 into the binary gate.
    # g_i = 0 when conflict is below tau; g_i = 1 once conflict exceeds tau+width.
    # The client both signals rank as worst stays on raw L1 (g_i forced to 0).
    adaptive_l1l2_gating_enabled: bool = False
    adaptive_l1l2_gating_conflict_threshold: float = 0.25
    adaptive_l1l2_gating_steepness: float = 12.0
    adaptive_l1l2_gating_width: float = 0.25

    # Parallel Trust Arbitration: two server-side candidates (gated vs ungated)
    # from the same client updates, then risk-aware select/blend. Off by default.
    # method: hard | soft | conservative. Selector uses probe/runtime quantities
    # only — never ground-truth attack labels.
    parallel_trust_arbitration_enabled: bool = False
    parallel_trust_arbitration_method: str = "hard"
    parallel_trust_arbitration_default_branch: str = "ungated"
    parallel_trust_arbitration_tau: float = 0.05
    parallel_trust_arbitration_mixture_steepness: float = 12.0
    parallel_trust_arbitration_lambda_H: float = 1.0
    parallel_trust_arbitration_lambda_R: float = 1.0
    parallel_trust_arbitration_lambda_S: float = 0.1
    parallel_trust_arbitration_lambda_trigger: float = 1.0
    # Extra J cost when a branch holds the current global (skip/no-op).
    # 3-seed grid: 0.25 recovered Byzantine/gradient-scaling without
    # collapsing model-replacement the way larger tau / λ_S=0 did.
    parallel_trust_arbitration_lambda_skip: float = 0.25
    parallel_trust_arbitration_risk_reduce: str = "max"

    # When set to "backdoor", Layer 1 probe scores modulate aggregation trust
    # while Byzantine-style gating (neutral L1 + soft cap) stays for other attacks.
    evaluation_attack_type: Optional[str] = None
    # Model-replacement: route Layer 1 probe scores into aggregation plus
    # attack-scoped hard norm rejection (5-client fair compare: 0.607 mean F1).
    use_layer1_for_model_replacement: bool = True
    # Backdoor-only: soft cap from backdoor_trigger probe harm (research path).
    # Disabled by default: 3-seed ablation showed worse trigger P vs legacy min-blend.
    backdoor_probe_aware_soft_cap_enabled: bool = False
    backdoor_probe_tolerance: float = 0.01
    backdoor_probe_gain: float = 10.0
    backdoor_probe_blend_with_raw_l1: bool = True
    # Backdoor production path: probe-scoped hard reject drops clear trigger-harm
    # outliers from aggregation (prefer agg-guard hold over soft blending).
    backdoor_probe_hard_reject_enabled: bool = True
    # Retuned + L1-gated: only reject harm outliers that also look untrusted.
    backdoor_probe_hard_reject_harm: float = 0.10
    backdoor_probe_hard_reject_peer_margin: float = 0.02
    backdoor_probe_hard_reject_warmup_rounds: int = 1
    backdoor_probe_hard_reject_cohort_factor: float = 1.5
    # Require Layer-1 raw trust at/below this to accept a harm-based reject
    # (blocks high-trust benign false positives like client_1).
    backdoor_probe_hard_reject_max_trust: float = 0.85
    backdoor_probe_hard_reject_min_keep: int = 2
    # Geometry hard-reject: Krum peer-distance signal for backdoor.
    # shortlist_veto: multi-Krum shortlist then keep lowest absolute trigger P
    # (production default; nearly ties Krum on 10-seed means).
    # shortlist_blend: blend eligible shortlist only on near-tied triggers
    # (research; did not beat veto on smoke seeds).
    # keep_best / reject_worst: legacy geometry modes.
    backdoor_geometry_hard_reject_enabled: bool = True
    backdoor_geometry_hard_reject_mode: str = "shortlist_veto"
    backdoor_geometry_hard_reject_score_ratio: float = 1.25
    backdoor_geometry_hard_reject_warmup_rounds: int = 0
    backdoor_geometry_hard_reject_min_keep: int = 1
    backdoor_geometry_shortlist_k: int = 2
    # Prefer trigger veto over pure geometry when cf_p exceeds shortlist min
    # by this margin; within margin, clients remain eligible for blend/veto.
    backdoor_geometry_trigger_veto_margin: float = 0.01
    # Only blend when eligible trigger spread is within this band; otherwise
    # keep the single lowest-trigger client (avoids diluting a clear winner).
    backdoor_geometry_blend_tie_eps: float = 0.005
    # Post-aggregation trigger scrub: a few SGD steps on server probe sets
    # after Layer-2 aggregation (backdoor-only). Descends on benign + trigger
    # (attack labels) to undo trigger→benign without wrecking clean utility.
    post_agg_trigger_scrub_enabled: bool = True
    post_agg_trigger_scrub_steps: int = 5
    post_agg_trigger_scrub_lr: float = 5e-3
    post_agg_trigger_scrub_trigger_coef: float = 1.5
    post_agg_trigger_scrub_max_delta_norm: float = 1.0
    post_agg_trigger_scrub_warmup_rounds: int = 0
    # Label-flip geometry (parked): shortlist_clean k=2 + clean_margin 0.025.
    # 10-seed mean F1 slightly above Krum (~0.750 vs 0.746). shortlist_clean_blend
    # did not widen the gap (mean +0.005 but 4W/6L). keep_best ≈ ties classic Krum.
    label_flip_geometry_hard_reject_enabled: bool = True
    label_flip_geometry_hard_reject_mode: str = "shortlist_clean"
    label_flip_geometry_shortlist_k: int = 2
    # Used by shortlist_clean_blend (research); unused for parked shortlist_clean.
    label_flip_geometry_blend_keep: int = 2
    # Override Krum winner only when another shortlist member's clean quality
    # exceeds it by this margin (filters ~0.003 noise).
    label_flip_geometry_clean_margin: float = 0.025
    label_flip_geometry_hard_reject_warmup_rounds: int = 0
    label_flip_geometry_hard_reject_min_keep: int = 1
    # Legacy soft-clamp path (high thresholds); off when hard reject is on.
    backdoor_trigger_hard_gate_enabled: bool = False
    backdoor_trigger_hard_gate_harm: float = 2.0
    backdoor_trigger_hard_gate_peer_margin: float = 2.0
    backdoor_trigger_hard_gate_warmup_rounds: int = 3
    # Kept for CLI/config compatibility; only applied together with harm trip.
    backdoor_trigger_hard_gate_p_attack_floor: float = 0.15
    backdoor_trigger_hard_gate_require_p_attack_floor: bool = False
    enable_layer3: bool = True
    spoofing_detection_threshold: float = 0.5
    cross_modal_weight: float = 0.7
    layer3_probe_blend_weight: float = 0.35

    # Probe scenarios for counterfactual validation
    probe_scenarios: List[str] = field(
        default_factory=lambda: [
            "benign",
            "gnss_spoofing",
            "adsb_injection",
            "mavlink_waypoint",
            "jamming",
            "backdoor_trigger",
        ]
    )
    # Cap windows per scenario (None = use full probe set). 16 cuts Layer~1
    # counterfactual cost ~3x vs the default 48-window probes with small
    # trust-AUC impact when combined with attack scoping.
    probe_max_windows_per_scenario: Optional[int] = None
    # When True and evaluation_attack_type is set, evaluate only the scenarios
    # relevant to that attack (see ProbeEvaluator.ATTACK_SCOPED_SCENARIOS).
    probe_attack_scoped: bool = False

    # Mission-context feature weights (cyber-physical consistency)
    mission_context_features: Dict[str, float] = field(
        default_factory=lambda: {
            "gnss_imu_consistency": 0.30,
            "mavlink_state_consistency": 0.25,
            "link_health": 0.20,
            "local_validation_loss": 0.15,
            "participation_history": 0.10,
        }
    )

    # Malicious client labels for evaluation (client_id -> is_malicious)
    ground_truth_malicious: Optional[Dict[str, bool]] = None

    # Output
    export_trust_scores: bool = True
    trust_score_path: Optional[str] = None
