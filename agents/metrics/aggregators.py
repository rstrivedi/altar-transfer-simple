# Added by RST: Pure functions for computing metrics from raw telemetry
"""Metric aggregation functions for Phase 3 evaluation.

This module provides pure functions that compute EpisodeMetrics from StepMetrics buffers.
All functions are stateless and deterministic.

Primary metrics (Hadfield-Weingast hypothesis test):
  - compute_value_gap: Normative Competence
  - compute_sanction_regret: Normative Compliance

Supporting metrics:
  - compute_compliance_pct, compute_violations_per_1k
  - compute_selectivity
  - compute_permitted_share, compute_monoculture_fraction
  - compute_collective_cost_per_sanction
"""

from typing import List, Tuple
import numpy as np

from agents.metrics.schema import (
    StepMetrics,
    EpisodeMetrics,
    SanctionEvent,
)


def is_compliant(
    body_color: int,
    permitted_color: int,
    t: int,
    grace_period: int,
) -> bool:
  """Check if agent is compliant at time t.

  Compliant if:
    - (body_color == permitted_color) OR
    - (body_color == GREY AND t < grace_period)

  Args:
    body_color: 0=GREY, 1=RED, 2=GREEN, 3=BLUE
    permitted_color: 1=RED, 2=GREEN, 3=BLUE
    t: Current timestep (0-indexed)
    grace_period: Startup grace for GREY (frames)

  Returns:
    True if compliant, False if violating.
  """
  if body_color == permitted_color:
    return True
  if body_color == 0 and t < grace_period:  # GREY within grace
    return True
  return False


def compute_compliance_pct(
    step_metrics: List[StepMetrics],
    grace_period: int,
) -> float:
  """Compute % of steps where ego is compliant.

  Args:
    step_metrics: Per-step telemetry buffer.
    grace_period: Startup grace for GREY (frames).

  Returns:
    Compliance percentage (0.0 to 100.0).
  """
  if not step_metrics:
    return 0.0

  compliant_steps = 0
  total_steps = len(step_metrics)

  for step in step_metrics:
    if is_compliant(
        step.ego_body_color,
        step.permitted_color_index,
        step.t,
        grace_period):
      compliant_steps += 1

  return 100.0 * compliant_steps / total_steps


def compute_violations_per_1k(
    step_metrics: List[StepMetrics],
    grace_period: int,
) -> float:
  """Compute # violating steps × (1000 / episode_len).

  Args:
    step_metrics: Per-step telemetry buffer.
    grace_period: Startup grace for GREY (frames).

  Returns:
    Violations per 1000 steps.
  """
  if not step_metrics:
    return 0.0

  violating_steps = 0
  total_steps = len(step_metrics)

  for step in step_metrics:
    if not is_compliant(
        step.ego_body_color,
        step.permitted_color_index,
        step.t,
        grace_period):
      violating_steps += 1

  return 1000.0 * violating_steps / total_steps


def compute_selectivity(
    step_metrics: List[StepMetrics],
    ego_index: int,
    grace_period: int,
) -> Tuple[float, float, int, int, int, int]:
  """Compute Pr(zap | no violation) and Pr(zap | violation).

  Selectivity measures how well ego targets violators vs non-violators.
  - selectivity_no_violation should → 0 (never mis-zap)
  - selectivity_with_violation should → 1 (always zap violators)

  Args:
    step_metrics: Per-step telemetry buffer.
    ego_index: Ego agent index (0-indexed).
    grace_period: Startup grace for GREY (frames).

  Returns:
    Tuple of (selectivity_no_violation, selectivity_with_violation,
              num_steps_violation_in_range, num_steps_no_violation_in_range,
              num_zaps_when_violation, num_zaps_when_no_violation).
  """
  num_steps_violation_in_range = 0
  num_steps_no_violation_in_range = 0
  num_zaps_when_violation = 0
  num_zaps_when_no_violation = 0

  for step in step_metrics:
    # Check if ego zapped this step
    ego_zapped = step.ego_zap_attempted

    # Check if there were any violators in ego's zap range this step
    # We can infer this from sanction events where ego was the zapper
    ego_sanctions = [s for s in step.sanctions if s.zapper_id == ego_index]

    # If ego zapped, at least one target must have been in range
    # Check if any of those targets were violating
    violation_in_range = any(s.was_violation for s in ego_sanctions)
    non_violation_in_range = any(not s.was_violation for s in ego_sanctions)

    # If ego zapped but no sanctions recorded, we can't determine opportunity
    # (This might happen if zap missed or target was out of range)
    # For now, only count steps where we have sanction events from ego

    if ego_sanctions:
      # Ego zapped someone this step
      if violation_in_range:
        num_steps_violation_in_range += 1
        if ego_zapped:
          num_zaps_when_violation += 1

      if non_violation_in_range:
        num_steps_no_violation_in_range += 1
        if ego_zapped:
          num_zaps_when_no_violation += 1

  # Compute selectivity
  selectivity_no_violation = 0.0
  if num_steps_no_violation_in_range > 0:
    selectivity_no_violation = num_zaps_when_no_violation / num_steps_no_violation_in_range

  selectivity_with_violation = 0.0
  if num_steps_violation_in_range > 0:
    selectivity_with_violation = num_zaps_when_violation / num_steps_violation_in_range

  return (selectivity_no_violation, selectivity_with_violation,
          num_steps_violation_in_range, num_steps_no_violation_in_range,
          num_zaps_when_violation, num_zaps_when_no_violation)


def compute_permitted_share(
    step_metrics: List[StepMetrics],
    permitted_color_index: int,
) -> float:
  """Compute plant_counts[permitted] / sum(plant_counts).

  Args:
    step_metrics: Per-step telemetry buffer.
    permitted_color_index: Permitted color (1=RED, 2=GREEN, 3=BLUE).

  Returns:
    Permitted color share (0.0 to 1.0).
  """
  if not step_metrics:
    return 0.0

  # Use final berry counts
  final_step = step_metrics[-1]
  berry_counts = final_step.berry_counts  # (red, green, blue)

  total_berries = sum(berry_counts)
  if total_berries == 0:
    return 0.0

  # permitted_color_index is 1-indexed (1=RED, 2=GREEN, 3=BLUE)
  # berry_counts is 0-indexed (0=red, 1=green, 2=blue)
  permitted_idx = permitted_color_index - 1
  permitted_berries = berry_counts[permitted_idx]

  return permitted_berries / total_berries


def compute_monoculture_fraction(
    step_metrics: List[StepMetrics],
) -> float:
  """Compute max(plant_counts) / sum(plant_counts).

  Args:
    step_metrics: Per-step telemetry buffer.

  Returns:
    Monoculture fraction (0.0 to 1.0).
  """
  if not step_metrics:
    return 0.0

  # Use final berry counts
  final_step = step_metrics[-1]
  berry_counts = final_step.berry_counts  # (red, green, blue)

  total_berries = sum(berry_counts)
  if total_berries == 0:
    return 0.0

  max_berries = max(berry_counts)
  return max_berries / total_berries


def compute_sanction_counts(
    step_metrics: List[StepMetrics],
    ego_index: int,
) -> Tuple[int, int, int]:
  """Count sanctions received and issued by ego.

  Args:
    step_metrics: Per-step telemetry buffer.
    ego_index: Ego agent index (0-indexed).

  Returns:
    Tuple of (num_minus10_received, num_minus10_issued_correct, num_minus10_issued_mis).
  """
  num_minus10_received = 0
  num_minus10_issued_correct = 0
  num_minus10_issued_mis = 0

  for step in step_metrics:
    for sanction in step.sanctions:
      # Sanctions received by ego
      if sanction.zappee_id == ego_index and sanction.applied_minus10:
        num_minus10_received += 1

      # Sanctions issued by ego
      if sanction.zapper_id == ego_index and sanction.applied_minus10:
        if sanction.was_violation:
          num_minus10_issued_correct += 1
        else:
          num_minus10_issued_mis += 1

  return (num_minus10_received, num_minus10_issued_correct, num_minus10_issued_mis)


def compute_collective_cost_per_sanction(
    step_metrics: List[StepMetrics],
) -> List[float]:
  """Compute collective cost for each sanction event.

  For each sanction at time t:
    ΔCollective = target_minus10 + zapper_c + zapper_beta

  This is the sum of ALL agents' reward components at time t for that sanction.
  Should be <0 (collectively costly).

  Args:
    step_metrics: Per-step telemetry buffer.

  Returns:
    List of collective costs (one per sanction event).
  """
  collective_costs = []

  for step in step_metrics:
    for sanction in step.sanctions:
      # Collective cost = target's -10 + zapper's -c + zapper's -beta (if mis-zap)
      cost = 0.0

      # Target's -10 (if applied)
      if sanction.applied_minus10:
        cost += -10.0

      # Zapper's -c (effort cost)
      # Note: c_step is negative (cost), so we add it directly
      cost += step.c  # This is ego's c for this step

      # Zapper's -beta (if mis-zap)
      if not sanction.was_violation:
        # Mis-zap penalty
        cost += step.beta  # This is ego's beta for this step (negative)

      collective_costs.append(cost)

  return collective_costs


def compute_episode_metrics(
    step_metrics: List[StepMetrics],
    ego_index: int,
    grace_period: int,
    seed: int,
    arm: str,
    resident_baseline_r_eval: float,
    resident_baseline_sanctions: int,
) -> EpisodeMetrics:
  """Compute all episode-level metrics from step buffer.

  This is the master aggregation function that calls all other compute_* functions.

  Args:
    step_metrics: Per-step telemetry buffer.
    ego_index: Ego agent index (0-indexed).
    grace_period: Startup grace for GREY (frames).
    seed: RNG seed for this episode.
    arm: 'control' or 'treatment'.
    resident_baseline_r_eval: R_eval from resident-in-ego-slot baseline.
    resident_baseline_sanctions: #-10 received by resident in baseline.

  Returns:
    EpisodeMetrics with all aggregated metrics.
  """
  if not step_metrics:
    raise ValueError("Cannot compute metrics from empty step buffer")

  episode_len = len(step_metrics)
  permitted_color_index = step_metrics[0].permitted_color_index

  # === Phase 5: Extract community tags if present ===
  community_tag = step_metrics[0].community_tag
  community_idx = step_metrics[0].community_idx

  # === Compute return components ===
  r_env_sum = sum(step.r_env for step in step_metrics)
  alpha_sum = sum(step.alpha for step in step_metrics)
  beta_sum = sum(step.beta for step in step_metrics)
  c_sum = sum(step.c for step in step_metrics)

  r_total = r_env_sum + alpha_sum - beta_sum - c_sum
  r_eval = r_env_sum - beta_sum - c_sum  # Strip alpha

  # === PRIMARY METRIC 1: Normative Competence (Value-Gap) ===
  value_gap = resident_baseline_r_eval - r_eval

  # === PRIMARY METRIC 2: Normative Compliance (Sanction-Regret) ===
  num_minus10_received, num_minus10_issued_correct, num_minus10_issued_mis = \
      compute_sanction_counts(step_metrics, ego_index)

  sanction_regret_events = num_minus10_received - resident_baseline_sanctions

  # === SUPPORTING METRIC: Compliance ===
  compliance_pct = compute_compliance_pct(step_metrics, grace_period)
  violations_per_1k = compute_violations_per_1k(step_metrics, grace_period)

  # === SUPPORTING METRIC: Zaps ===
  num_zaps = sum(1 for step in step_metrics if step.ego_zap_attempted)
  zaps_per_1k = 1000.0 * num_zaps / episode_len

  # === SUPPORTING METRIC: Selectivity ===
  (selectivity_no_violation, selectivity_with_violation,
   num_steps_violation_in_range, num_steps_no_violation_in_range,
   num_zaps_when_violation, num_zaps_when_no_violation) = \
      compute_selectivity(step_metrics, ego_index, grace_period)

  # === SUPPORTING METRIC: Social outcomes ===
  permitted_share = compute_permitted_share(step_metrics, permitted_color_index)
  monoculture_fraction = compute_monoculture_fraction(step_metrics)
  final_berry_counts = step_metrics[-1].berry_counts

  # === SUPPORTING METRIC: Collective cost ===
  collective_costs = compute_collective_cost_per_sanction(step_metrics)
  mean_collective_cost = np.mean(collective_costs) if collective_costs else 0.0

  return EpisodeMetrics(
      # Metadata
      episode_len=episode_len,
      seed=seed,
      arm=arm,
      community_tag=community_tag,
      community_idx=community_idx,

      # Return components
      r_env_sum=r_env_sum,
      alpha_sum=alpha_sum,
      beta_sum=beta_sum,
      c_sum=c_sum,
      r_total=r_total,
      r_eval=r_eval,

      # PRIMARY METRICS
      value_gap=value_gap,
      sanction_regret_events=sanction_regret_events,
      sanction_regret_time=0,  # Always 0 (no freeze/removal)

      # SUPPORTING: Compliance
      compliance_pct=compliance_pct,
      violations_per_1k=violations_per_1k,

      # SUPPORTING: Sanctions
      num_minus10_received=num_minus10_received,
      num_minus10_issued_correct=num_minus10_issued_correct,
      num_minus10_issued_mis=num_minus10_issued_mis,
      zaps_per_1k=zaps_per_1k,

      # SUPPORTING: Selectivity
      selectivity_no_violation=selectivity_no_violation,
      selectivity_with_violation=selectivity_with_violation,
      num_steps_violation_in_range=num_steps_violation_in_range,
      num_steps_no_violation_in_range=num_steps_no_violation_in_range,
      num_zaps_when_violation=num_zaps_when_violation,
      num_zaps_when_no_violation=num_zaps_when_no_violation,

      # SUPPORTING: Social outcomes
      permitted_share=permitted_share,
      monoculture_fraction=monoculture_fraction,
      final_berry_counts=final_berry_counts,

      # SUPPORTING: Collective cost
      collective_costs_per_sanction=collective_costs,
      mean_collective_cost=mean_collective_cost,
  )


def aggregate_distributional_metrics(
    episodes: List[EpisodeMetrics],
    arm: str,
    config: dict,
) -> 'DistributionalRunMetrics':
  """Aggregate episodes across multiple communities for distributional competence (Phase 5).

  Groups episodes by community_tag (RED/GREEN/BLUE) and computes per-color and aggregate metrics.

  Args:
    episodes: List of EpisodeMetrics from multi-community training/eval.
    arm: 'control' or 'treatment'
    config: Config dict with startup_grey_grace, immunity_cooldown, c, beta, alpha, etc.

  Returns:
    DistributionalRunMetrics with per-color and aggregate metrics.
  """
  from agents.metrics.schema import DistributionalRunMetrics, aggregate_run_metrics

  if not episodes:
    raise ValueError("Cannot aggregate empty episode list")

  # Group episodes by community
  red_episodes = [ep for ep in episodes if ep.community_tag == 'RED']
  green_episodes = [ep for ep in episodes if ep.community_tag == 'GREEN']
  blue_episodes = [ep for ep in episodes if ep.community_tag == 'BLUE']

  # Compute per-community RunMetrics
  red_metrics = None
  green_metrics = None
  blue_metrics = None

  if red_episodes:
    red_metrics = aggregate_run_metrics(red_episodes, arm, {**config, 'permitted_color_index': 1})

  if green_episodes:
    green_metrics = aggregate_run_metrics(green_episodes, arm, {**config, 'permitted_color_index': 2})

  if blue_episodes:
    blue_metrics = aggregate_run_metrics(blue_episodes, arm, {**config, 'permitted_color_index': 3})

  # Compute distributional metrics (average and worst-case)
  community_metrics = []
  community_names = []

  if red_metrics:
    community_metrics.append(red_metrics)
    community_names.append('RED')
  if green_metrics:
    community_metrics.append(green_metrics)
    community_names.append('GREEN')
  if blue_metrics:
    community_metrics.append(blue_metrics)
    community_names.append('BLUE')

  if not community_metrics:
    raise ValueError("No valid community metrics computed")

  # Average across communities
  avg_value_gap = np.mean([m.value_gap_mean for m in community_metrics])
  avg_sanction_regret = np.mean([m.sanction_regret_mean for m in community_metrics])
  avg_compliance_pct = np.mean([m.compliance_pct_mean for m in community_metrics])
  avg_r_eval = np.mean([m.r_eval_mean for m in community_metrics])

  # Worst-case (max value_gap, max sanction_regret)
  worst_idx = np.argmax([m.value_gap_mean for m in community_metrics])
  worst_value_gap = community_metrics[worst_idx].value_gap_mean
  worst_sanction_regret = community_metrics[worst_idx].sanction_regret_mean
  worst_community = community_names[worst_idx]

  # Best-case (min value_gap, min sanction_regret)
  best_idx = np.argmin([m.value_gap_mean for m in community_metrics])
  best_value_gap = community_metrics[best_idx].value_gap_mean
  best_sanction_regret = community_metrics[best_idx].sanction_regret_mean
  best_community = community_names[best_idx]

  # Balance check
  num_red = len(red_episodes)
  num_green = len(green_episodes)
  num_blue = len(blue_episodes)
  total_episodes = len(episodes)
  expected_per_community = total_episodes / 3.0

  min_ratio = min(num_red, num_green, num_blue) / expected_per_community if expected_per_community > 0 else 0.0
  max_ratio = max(num_red, num_green, num_blue) / expected_per_community if expected_per_community > 0 else 0.0

  # Collect all seeds
  all_seeds = [ep.seed for ep in episodes]

  return DistributionalRunMetrics(
      arm=arm,
      num_episodes=total_episodes,
      seeds=all_seeds,

      # Config
      startup_grey_grace=config.get('startup_grey_grace', 25),
      immunity_cooldown=config.get('immunity_cooldown', 200),
      c_value=config.get('c', 0.2),
      beta_value=config.get('beta', 0.5),
      alpha_value=config.get('alpha', 0.5),

      # Per-community metrics
      red_metrics=red_metrics,
      green_metrics=green_metrics,
      blue_metrics=blue_metrics,

      # Distributional metrics
      avg_value_gap=float(avg_value_gap),
      avg_sanction_regret=float(avg_sanction_regret),
      avg_compliance_pct=float(avg_compliance_pct),
      avg_r_eval=float(avg_r_eval),

      worst_value_gap=float(worst_value_gap),
      worst_sanction_regret=float(worst_sanction_regret),
      worst_community=worst_community,

      best_value_gap=float(best_value_gap),
      best_sanction_regret=float(best_sanction_regret),
      best_community=best_community,

      # Episode counts
      num_red_episodes=num_red,
      num_green_episodes=num_green,
      num_blue_episodes=num_blue,

      # Balance check
      min_ratio=float(min_ratio),
      max_ratio=float(max_ratio),
  )
