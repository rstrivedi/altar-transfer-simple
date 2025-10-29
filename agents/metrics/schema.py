# Added by RST: Type-safe schema for Phase 3 metrics and telemetry
"""Data structures for metrics collection and evaluation.

This module defines typed dataclasses for:
- Per-step telemetry (StepMetrics)
- Per-episode aggregates (EpisodeMetrics)
- Per-run summaries (RunMetrics)

All metrics support the Hadfield-Weingast hypothesis test:
  PRIMARY METRICS (treatment should beat control):
    - Normative Competence: value-gap (R_eval^baseline - R_eval^ego)
    - Normative Compliance: sanction-regret (#sanctions_ego - #sanctions_baseline)

  SUPPORTING METRICS (explain why treatment beats control):
    - Compliance %, violations/1k, zaps/1k, selectivity
    - Permitted-color share, monoculture fraction
    - Collective cost per sanction
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class SanctionEvent:
  """Single sanction event (zap that lands or attempts).

  Fields match 'sanction' event from Lua components.lua.
  """
  t: int  # Frame number
  zapper_id: int  # 1-indexed in Lua, will convert to 0-indexed
  zappee_id: int  # 1-indexed in Lua, will convert to 0-indexed
  zappee_color: int  # 0=GREY, 1=RED, 2=GREEN, 3=BLUE
  was_violation: bool  # True if zappee was violating
  applied_minus10: bool  # True if -10 was applied (not immune, not tie-broken)
  immune: bool  # True if target was immune
  tie_break: bool  # True if blocked by tie-break (another zap landed same frame)


@dataclass
class PlantEvent:
  """Single replanting event.

  Fields match 'replanting' event from Lua components.lua.
  """
  player_index: int  # 1-indexed in Lua, will convert to 0-indexed
  source_berry: int  # Previous color (1=RED, 2=GREEN, 3=BLUE)
  target_berry: int  # New color (1=RED, 2=GREEN, 3=BLUE)


@dataclass
class EatEvent:
  """Single eating event.

  Fields match 'eating' event from Lua components.lua.
  """
  player_index: int  # 1-indexed in Lua, will convert to 0-indexed
  berry_color: int  # 1=RED, 2=GREEN, 3=BLUE


@dataclass
class StepMetrics:
  """Per-step telemetry captured during episode rollout.

  This captures ALL raw data needed to compute episode-level metrics.
  NO computation happens here - just data collection.
  """
  t: int  # Frame number (0-indexed)

  # === Ego reward decomposition ===
  # r_total = r_env + alpha - beta - c
  # r_eval = r_env - beta - c (strip alpha for evaluation)
  r_env: float  # Base environment reward (berries, -10 from sanctions)
  alpha: float  # Training bonus for correct zaps (STRIPPED in eval)
  beta: float  # Penalty for mis-zaps
  c: float  # Effort cost per zap

  # === Ego state ===
  ego_body_color: int  # 0=GREY, 1=RED, 2=GREEN, 3=BLUE (from resident_info event)

  # === Ego zap action ===
  ego_action: int  # Action taken by ego (0-10)
  ego_zap_attempted: bool  # True if ego fired zap (action==7)

  # === Population events ===
  # All sanctions this step (including ego as zapper or zappee)
  sanctions: List[SanctionEvent] = field(default_factory=list)

  # All plants this step
  plants: List[PlantEvent] = field(default_factory=list)

  # All eats this step
  eats: List[EatEvent] = field(default_factory=list)

  # === Global state ===
  permitted_color_index: int  # 1=RED, 2=GREEN, 3=BLUE
  berry_counts: Tuple[int, int, int]  # (red, green, blue) from BERRIES_BY_TYPE observation

  # === Phase 5: Multi-community tracking ===
  community_tag: Optional[str] = None  # 'RED', 'GREEN', or 'BLUE' (if multi_community_mode)
  community_idx: Optional[int] = None  # 1, 2, or 3 (if multi_community_mode)


@dataclass
class EpisodeMetrics:
  """Aggregated metrics for a single episode.

  Computed from StepMetrics buffer at episode end.
  Includes both PRIMARY and SUPPORTING metrics.
  """
  # === Episode metadata ===
  episode_len: int  # Number of steps
  seed: int  # RNG seed for reproducibility
  arm: str  # 'control' or 'treatment'

  # === Phase 5: Multi-community tracking ===
  community_tag: Optional[str] = None  # 'RED', 'GREEN', or 'BLUE' (if multi_community_mode)
  community_idx: Optional[int] = None  # 1, 2, or 3 (if multi_community_mode)

  # === Ego return components ===
  r_env_sum: float  # Σ r_env over episode
  alpha_sum: float  # Σ alpha over episode (for strip test)
  beta_sum: float  # Σ beta over episode (mis-zap penalties)
  c_sum: float  # Σ c over episode (zap costs)

  r_total: float  # = r_env_sum + alpha_sum - beta_sum - c_sum (training return)
  r_eval: float  # = r_env_sum - beta_sum - c_sum (EXCLUDES alpha for evaluation)

  # === PRIMARY METRIC 1: Normative Competence ===
  # Value-gap: how much worse is ego vs resident baseline?
  # Computed as: ΔV = R_eval^resident_baseline - R_eval^ego
  # Lower is better (ego approaching resident performance)
  value_gap: float  # Requires resident_baseline_r_eval as input

  # === PRIMARY METRIC 2: Normative Compliance ===
  # Sanction-regret (events): excess sanctions ego receives vs baseline
  # Computed as: SR = (#-10 received by ego) - (#-10 received by resident baseline)
  # Lower is better (ego receiving fewer sanctions)
  sanction_regret_events: int  # Requires resident_baseline_sanctions as input

  # Sanction-regret (time): always 0 in our setup (no freeze/removal)
  sanction_regret_time: int = 0  # Stub for completeness

  # === SUPPORTING METRIC: Compliance behavior ===
  compliance_pct: float  # % steps where ego is compliant
  violations_per_1k: float  # # violating steps × (1000 / episode_len)

  # === SUPPORTING METRIC: Sanction counts ===
  num_minus10_received: int  # By ego
  num_minus10_issued_correct: int  # By ego, target was violating
  num_minus10_issued_mis: int  # By ego, target was compliant
  zaps_per_1k: float  # Ego zap attempts × (1000 / episode_len)

  # === SUPPORTING METRIC: Selectivity ===
  # Pr(zap | no violation in zap path) and Pr(zap | violation present)
  selectivity_no_violation: float  # Should → 0 (no mis-zaps)
  selectivity_with_violation: float  # Should → 1 (always zap violators)

  # For selectivity computation, track opportunities
  num_steps_violation_in_range: int = 0
  num_steps_no_violation_in_range: int = 0
  num_zaps_when_violation: int = 0
  num_zaps_when_no_violation: int = 0

  # === SUPPORTING METRIC: Social outcomes ===
  permitted_share: float  # plant_counts[permitted]/Σ(plant_counts)
  monoculture_fraction: float  # max(plant_counts)/Σ(plant_counts)

  final_berry_counts: Tuple[int, int, int] = (0, 0, 0)  # (red, green, blue) at t=final

  # === SUPPORTING METRIC: Collective cost ===
  # For each sanction, compute ΔCollective = Σ(all agents' reward components at t)
  # Should be <0 (collectively costly)
  collective_costs_per_sanction: List[float] = field(default_factory=list)
  mean_collective_cost: float = 0.0  # Mean across all sanctions this episode


@dataclass
class RunMetrics:
  """Aggregated metrics across multiple episodes (for W&B logging).

  Contains mean ± std for all EpisodeMetrics fields.
  """
  # === Run metadata ===
  arm: str  # 'control' or 'treatment'
  num_episodes: int  # Number of episodes aggregated
  seeds: List[int]  # Seeds used for reproducibility

  # === Config (logged for reproducibility) ===
  permitted_color_index: int  # 1=RED, 2=GREEN, 3=BLUE
  startup_grey_grace: int  # Grace period (frames)
  immunity_cooldown: int  # K (frames)
  c_value: float  # Zap cost
  beta_value: float  # Mis-zap penalty
  alpha_value: float  # Correct zap bonus (train only)

  # === PRIMARY METRICS (mean ± std) ===
  value_gap_mean: float
  value_gap_std: float
  value_gap_episodes: List[float] = field(default_factory=list)  # Per-episode values

  sanction_regret_mean: float
  sanction_regret_std: float
  sanction_regret_episodes: List[int] = field(default_factory=list)

  # === SUPPORTING METRICS (mean ± std) ===
  compliance_pct_mean: float
  compliance_pct_std: float

  violations_per_1k_mean: float
  violations_per_1k_std: float

  zaps_per_1k_mean: float
  zaps_per_1k_std: float

  selectivity_no_violation_mean: float
  selectivity_no_violation_std: float

  selectivity_with_violation_mean: float
  selectivity_with_violation_std: float

  permitted_share_mean: float
  permitted_share_std: float

  monoculture_fraction_mean: float
  monoculture_fraction_std: float

  r_eval_mean: float
  r_eval_std: float

  # === Return decomposition (for charts) ===
  r_env_sum_mean: float = 0.0
  alpha_sum_mean: float = 0.0
  beta_sum_mean: float = 0.0
  c_sum_mean: float = 0.0


@dataclass
class DistributionalRunMetrics:
  """Aggregated metrics across multiple communities for distributional competence (Phase 5).

  Evaluates policy performance across mixture distribution μ = {RED, GREEN, BLUE}.

  Key metrics:
    - Average performance: E_θ~μ[ΔV], E_θ~μ[SR]
    - Worst-case performance: max_θ(ΔV), max_θ(SR)
    - Per-color breakdown: {RED, GREEN, BLUE} × {ΔV, SR, compliance%}
  """
  # === Run metadata ===
  arm: str  # 'control' or 'treatment'
  num_episodes: int  # Total episodes across all communities
  seeds: List[int]  # Seeds used for reproducibility

  # === Config (logged for reproducibility) ===
  startup_grey_grace: int
  immunity_cooldown: int
  c_value: float
  beta_value: float
  alpha_value: float

  # === Per-community metrics ===
  # Each contains RunMetrics for that specific community
  red_metrics: Optional[RunMetrics] = None
  green_metrics: Optional[RunMetrics] = None
  blue_metrics: Optional[RunMetrics] = None

  # === Distributional metrics (aggregate) ===
  # Average across communities
  avg_value_gap: float = 0.0  # E_θ~μ[ΔV]
  avg_sanction_regret: float = 0.0  # E_θ~μ[SR]
  avg_compliance_pct: float = 0.0  # E_θ~μ[compliance%]
  avg_r_eval: float = 0.0  # E_θ~μ[R_eval]

  # Worst-case across communities
  worst_value_gap: float = 0.0  # max_θ(ΔV_mean)
  worst_sanction_regret: float = 0.0  # max_θ(SR_mean)
  worst_community: str = ""  # Which community is worst

  # Best-case across communities (for comparison)
  best_value_gap: float = 0.0  # min_θ(ΔV_mean)
  best_sanction_regret: float = 0.0  # min_θ(SR_mean)
  best_community: str = ""  # Which community is best

  # === Per-community episode counts ===
  num_red_episodes: int = 0
  num_green_episodes: int = 0
  num_blue_episodes: int = 0

  # === Balance check ===
  # How close to 1:1:1 ratio? (for verification)
  min_ratio: float = 0.0  # min(num_red, num_green, num_blue) / (total/3)
  max_ratio: float = 0.0  # max(num_red, num_green, num_blue) / (total/3)


# === Helper functions for creating metrics ===

def aggregate_run_metrics(
    episodes: List[EpisodeMetrics],
    arm: str,
    config: Dict,
) -> RunMetrics:
  """Aggregate episode metrics into run-level summary.

  Args:
    episodes: List of EpisodeMetrics from multiple episodes.
    arm: 'control' or 'treatment'
    config: Config dict with permitted_color_index, K, S, c, beta, alpha, etc.

  Returns:
    RunMetrics with mean/std across episodes.
  """
  if not episodes:
    raise ValueError("Cannot aggregate empty episode list")

  # Extract per-episode values
  value_gaps = [ep.value_gap for ep in episodes]
  sanction_regrets = [ep.sanction_regret_events for ep in episodes]
  compliance_pcts = [ep.compliance_pct for ep in episodes]
  violations_per_1k = [ep.violations_per_1k for ep in episodes]
  zaps_per_1k = [ep.zaps_per_1k for ep in episodes]
  selectivity_no_viol = [ep.selectivity_no_violation for ep in episodes]
  selectivity_with_viol = [ep.selectivity_with_violation for ep in episodes]
  permitted_shares = [ep.permitted_share for ep in episodes]
  monocultures = [ep.monoculture_fraction for ep in episodes]
  r_evals = [ep.r_eval for ep in episodes]
  r_envs = [ep.r_env_sum for ep in episodes]
  alphas = [ep.alpha_sum for ep in episodes]
  betas = [ep.beta_sum for ep in episodes]
  cs = [ep.c_sum for ep in episodes]

  seeds = [ep.seed for ep in episodes]

  return RunMetrics(
      arm=arm,
      num_episodes=len(episodes),
      seeds=seeds,

      # Config
      permitted_color_index=config.get('permitted_color_index', 1),
      startup_grey_grace=config.get('startup_grey_grace', 25),
      immunity_cooldown=config.get('immunity_cooldown', 200),
      c_value=config.get('c_value', 0.5),
      beta_value=config.get('beta_value', 0.0),
      alpha_value=config.get('alpha_value', 0.0),

      # Primary metrics
      value_gap_mean=float(np.mean(value_gaps)),
      value_gap_std=float(np.std(value_gaps)),
      value_gap_episodes=value_gaps,

      sanction_regret_mean=float(np.mean(sanction_regrets)),
      sanction_regret_std=float(np.std(sanction_regrets)),
      sanction_regret_episodes=sanction_regrets,

      # Supporting metrics
      compliance_pct_mean=float(np.mean(compliance_pcts)),
      compliance_pct_std=float(np.std(compliance_pcts)),

      violations_per_1k_mean=float(np.mean(violations_per_1k)),
      violations_per_1k_std=float(np.std(violations_per_1k)),

      zaps_per_1k_mean=float(np.mean(zaps_per_1k)),
      zaps_per_1k_std=float(np.std(zaps_per_1k)),

      selectivity_no_violation_mean=float(np.mean(selectivity_no_viol)),
      selectivity_no_violation_std=float(np.std(selectivity_no_viol)),

      selectivity_with_violation_mean=float(np.mean(selectivity_with_viol)),
      selectivity_with_violation_std=float(np.std(selectivity_with_viol)),

      permitted_share_mean=float(np.mean(permitted_shares)),
      permitted_share_std=float(np.std(permitted_shares)),

      monoculture_fraction_mean=float(np.mean(monocultures)),
      monoculture_fraction_std=float(np.std(monocultures)),

      r_eval_mean=float(np.mean(r_evals)),
      r_eval_std=float(np.std(r_evals)),

      # Return decomposition
      r_env_sum_mean=float(np.mean(r_envs)),
      alpha_sum_mean=float(np.mean(alphas)),
      beta_sum_mean=float(np.mean(betas)),
      c_sum_mean=float(np.mean(cs)),
  )
