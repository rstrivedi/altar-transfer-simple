# Added by RST: Weights & Biases integration for Phase 3 metrics
"""W&B logging for evaluation metrics and training progress.

This module provides W&B integration for logging:
- Run configuration (arm, K, S, c, beta, alpha, seed, commit)
- Per-episode metrics tables
- Run-level aggregated statistics
- Custom charts (value-gap, sanction-regret, compliance, monoculture)
- Videos from evaluation episodes

Environment variable required:
  WANDB_API_KEY: W&B API key for authentication

Usage:
  # Initialize W&B run
  run = init_wandb_run(config, arm='treatment', project='altar-transfer')

  # Log episode metrics
  log_episode_metrics(episode_metrics, training_step=100000)

  # Log run summary
  log_run_summary(run_metrics, training_step=100000)

  # Upload videos
  upload_videos(video_dir='./eval_videos', training_step=100000)

  # Finish run
  wandb.finish()
"""

from typing import Dict, Optional, List
import os
import numpy as np

try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  print("WARNING: wandb not available. Install with: pip install wandb")


from agents.metrics.schema import EpisodeMetrics, RunMetrics


def init_wandb_run(
    config: Dict,
    arm: str,
    project: str = "altar-transfer",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[object]:
  """Initialize W&B run with config tags.

  Args:
    config: Config dict with permitted_color_index, K, S, c, beta, alpha, etc.
    arm: 'baseline', 'treatment', or 'control'.
    project: W&B project name.
    entity: W&B entity (team/user). If None, uses default from WANDB_ENTITY env var.
    run_name: Optional run name. If None, W&B auto-generates.
    tags: Optional list of tags.

  Returns:
    wandb.Run object if successful, None if wandb unavailable.
  """
  if not WANDB_AVAILABLE:
    print("W&B not available, skipping initialization")
    return None

  # Check for API key
  api_key = os.environ.get('WANDB_API_KEY')
  if not api_key:
    print("WARNING: WANDB_API_KEY not set. W&B logging may fail.")

  # Build config for W&B
  wandb_config = {
      'arm': arm,
      'permitted_color_index': config.get('permitted_color_index', 1),
      'startup_grey_grace': config.get('startup_grey_grace', 25),
      'immunity_cooldown': config.get('immunity_cooldown', 200),
      'c_value': config.get('c_value', 0.5),
      'beta_value': config.get('beta_value', 0.0),
      'alpha_value': config.get('alpha_value', 0.0),
      'episode_timesteps': config.get('episode_timesteps', 2000),
      'seed': config.get('seed', 42),
  }

  # Add git commit if available
  try:
    import subprocess
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    wandb_config['git_commit'] = commit
  except:
    pass

  # Build tags
  run_tags = tags or []
  run_tags.append(arm)
  if config.get('alpha_value', 0.0) == 0.0:
    run_tags.append('alpha_off')
  else:
    run_tags.append('alpha_on')

  # Initialize run
  run = wandb.init(
      project=project,
      entity=entity,
      config=wandb_config,
      name=run_name,
      tags=run_tags,
      reinit=True,  # Allow multiple runs in same process
  )

  print(f"W&B run initialized: {run.name} ({run.id})")
  return run


def log_episode_metrics(
    episode_metrics: EpisodeMetrics,
    training_step: int = 0,
    prefix: str = "",
) -> None:
  """Log episode metrics to W&B.

  Args:
    episode_metrics: EpisodeMetrics instance.
    training_step: Global training step (for x-axis in charts).
    prefix: Optional prefix for metric names (e.g., 'eval/').
  """
  if not WANDB_AVAILABLE or not wandb.run:
    return

  # Build metrics dict
  metrics = {
      f'{prefix}episode_len': episode_metrics.episode_len,
      f'{prefix}seed': episode_metrics.seed,
      f'{prefix}r_env_sum': episode_metrics.r_env_sum,
      f'{prefix}alpha_sum': episode_metrics.alpha_sum,
      f'{prefix}beta_sum': episode_metrics.beta_sum,
      f'{prefix}c_sum': episode_metrics.c_sum,
      f'{prefix}r_total': episode_metrics.r_total,
      f'{prefix}r_eval': episode_metrics.r_eval,

      # PRIMARY METRICS
      f'{prefix}value_gap': episode_metrics.value_gap,
      f'{prefix}sanction_regret_events': episode_metrics.sanction_regret_events,

      # SUPPORTING METRICS
      f'{prefix}compliance_pct': episode_metrics.compliance_pct,
      f'{prefix}violations_per_1k': episode_metrics.violations_per_1k,
      f'{prefix}num_minus10_received': episode_metrics.num_minus10_received,
      f'{prefix}num_minus10_issued_correct': episode_metrics.num_minus10_issued_correct,
      f'{prefix}num_minus10_issued_mis': episode_metrics.num_minus10_issued_mis,
      f'{prefix}zaps_per_1k': episode_metrics.zaps_per_1k,
      f'{prefix}selectivity_no_violation': episode_metrics.selectivity_no_violation,
      f'{prefix}selectivity_with_violation': episode_metrics.selectivity_with_violation,
      f'{prefix}permitted_share': episode_metrics.permitted_share,
      f'{prefix}monoculture_fraction': episode_metrics.monoculture_fraction,
      f'{prefix}mean_collective_cost': episode_metrics.mean_collective_cost,
  }

  wandb.log(metrics, step=training_step)


def log_run_summary(
    run_metrics: RunMetrics,
    training_step: int = 0,
    prefix: str = "summary/",
) -> None:
  """Log run-level summary statistics to W&B.

  Args:
    run_metrics: RunMetrics instance with mean/std across episodes.
    training_step: Global training step.
    prefix: Prefix for summary metrics.
  """
  if not WANDB_AVAILABLE or not wandb.run:
    return

  # Build summary dict
  summary = {
      f'{prefix}arm': run_metrics.arm,
      f'{prefix}num_episodes': run_metrics.num_episodes,

      # PRIMARY METRICS (mean ± std)
      f'{prefix}value_gap_mean': run_metrics.value_gap_mean,
      f'{prefix}value_gap_std': run_metrics.value_gap_std,
      f'{prefix}sanction_regret_mean': run_metrics.sanction_regret_mean,
      f'{prefix}sanction_regret_std': run_metrics.sanction_regret_std,

      # SUPPORTING METRICS
      f'{prefix}compliance_pct_mean': run_metrics.compliance_pct_mean,
      f'{prefix}compliance_pct_std': run_metrics.compliance_pct_std,
      f'{prefix}violations_per_1k_mean': run_metrics.violations_per_1k_mean,
      f'{prefix}violations_per_1k_std': run_metrics.violations_per_1k_std,
      f'{prefix}zaps_per_1k_mean': run_metrics.zaps_per_1k_mean,
      f'{prefix}zaps_per_1k_std': run_metrics.zaps_per_1k_std,
      f'{prefix}selectivity_no_violation_mean': run_metrics.selectivity_no_violation_mean,
      f'{prefix}selectivity_no_violation_std': run_metrics.selectivity_no_violation_std,
      f'{prefix}selectivity_with_violation_mean': run_metrics.selectivity_with_violation_mean,
      f'{prefix}selectivity_with_violation_std': run_metrics.selectivity_with_violation_std,
      f'{prefix}permitted_share_mean': run_metrics.permitted_share_mean,
      f'{prefix}permitted_share_std': run_metrics.permitted_share_std,
      f'{prefix}monoculture_fraction_mean': run_metrics.monoculture_fraction_mean,
      f'{prefix}monoculture_fraction_std': run_metrics.monoculture_fraction_std,
      f'{prefix}r_eval_mean': run_metrics.r_eval_mean,
      f'{prefix}r_eval_std': run_metrics.r_eval_std,

      # Return decomposition
      f'{prefix}r_env_sum_mean': run_metrics.r_env_sum_mean,
      f'{prefix}alpha_sum_mean': run_metrics.alpha_sum_mean,
      f'{prefix}beta_sum_mean': run_metrics.beta_sum_mean,
      f'{prefix}c_sum_mean': run_metrics.c_sum_mean,
  }

  wandb.log(summary, step=training_step)

  # Also log as W&B summary (final values)
  for key, value in summary.items():
    wandb.run.summary[key] = value


def upload_video(
    video_path: str,
    caption: str = "",
    training_step: int = 0,
) -> None:
  """Upload video to W&B.

  Args:
    video_path: Path to video file (.mp4).
    caption: Optional caption for video.
    training_step: Global training step.
  """
  if not WANDB_AVAILABLE or not wandb.run:
    return

  if not os.path.exists(video_path):
    print(f"WARNING: Video not found: {video_path}")
    return

  try:
    wandb.log({
        "video": wandb.Video(video_path, caption=caption, fps=8, format="mp4")
    }, step=training_step)
    print(f"Uploaded video: {video_path}")
  except Exception as e:
    print(f"ERROR uploading video {video_path}: {e}")


def upload_videos_from_dir(
    video_dir: str,
    training_step: int = 0,
    pattern: str = "*.mp4",
) -> None:
  """Upload all videos from directory to W&B.

  Args:
    video_dir: Directory containing videos.
    training_step: Global training step.
    pattern: Glob pattern for video files.
  """
  if not WANDB_AVAILABLE or not wandb.run:
    return

  import glob
  video_paths = glob.glob(os.path.join(video_dir, pattern))

  if not video_paths:
    print(f"No videos found in {video_dir} matching {pattern}")
    return

  print(f"Uploading {len(video_paths)} videos from {video_dir}")

  for video_path in video_paths:
    # Extract caption from filename
    filename = os.path.basename(video_path)
    caption = filename.replace('.mp4', '').replace('_', ' ')

    upload_video(video_path, caption=caption, training_step=training_step)


def log_comparison_table(
    baseline_metrics: RunMetrics,
    treatment_metrics: RunMetrics,
    control_metrics: RunMetrics,
) -> None:
  """Log comparison table for baseline vs treatment vs control.

  Args:
    baseline_metrics: RunMetrics for baseline.
    treatment_metrics: RunMetrics for treatment.
    control_metrics: RunMetrics for control.
  """
  if not WANDB_AVAILABLE or not wandb.run:
    return

  # Build comparison table
  table_data = [
      # Header
      ["Metric", "Baseline", "Treatment", "Control", "Treatment-Control"],

      # PRIMARY METRICS
      ["Value-gap (mean)",
       f"{baseline_metrics.value_gap_mean:.2f}",
       f"{treatment_metrics.value_gap_mean:.2f}",
       f"{control_metrics.value_gap_mean:.2f}",
       f"{treatment_metrics.value_gap_mean - control_metrics.value_gap_mean:.2f}"],

      ["Sanction-regret (mean)",
       f"{baseline_metrics.sanction_regret_mean:.2f}",
       f"{treatment_metrics.sanction_regret_mean:.2f}",
       f"{control_metrics.sanction_regret_mean:.2f}",
       f"{treatment_metrics.sanction_regret_mean - control_metrics.sanction_regret_mean:.2f}"],

      # SUPPORTING METRICS
      ["Compliance %",
       f"{baseline_metrics.compliance_pct_mean:.1f}",
       f"{treatment_metrics.compliance_pct_mean:.1f}",
       f"{control_metrics.compliance_pct_mean:.1f}",
       f"{treatment_metrics.compliance_pct_mean - control_metrics.compliance_pct_mean:.1f}"],

      ["Monoculture fraction",
       f"{baseline_metrics.monoculture_fraction_mean:.3f}",
       f"{treatment_metrics.monoculture_fraction_mean:.3f}",
       f"{control_metrics.monoculture_fraction_mean:.3f}",
       f"{treatment_metrics.monoculture_fraction_mean - control_metrics.monoculture_fraction_mean:.3f}"],

      ["R_eval",
       f"{baseline_metrics.r_eval_mean:.2f}",
       f"{treatment_metrics.r_eval_mean:.2f}",
       f"{control_metrics.r_eval_mean:.2f}",
       f"{treatment_metrics.r_eval_mean - control_metrics.r_eval_mean:.2f}"],
  ]

  table = wandb.Table(
      columns=["Metric", "Baseline", "Treatment", "Control", "Treatment-Control"],
      data=table_data[1:],  # Skip header
  )

  wandb.log({"comparison_table": table})


def create_custom_charts():
  """Register custom W&B chart definitions.

  Call this once at the start of training to define custom charts.
  """
  if not WANDB_AVAILABLE or not wandb.run:
    return

  # Note: W&B custom charts are typically configured via the web UI
  # This function is a placeholder for future custom chart definitions
  # For now, standard line plots should be sufficient

  print("Custom charts can be configured in W&B web UI")
  print("Recommended charts:")
  print("  1. Value-gap vs training_step (line + CI)")
  print("  2. Sanction-regret vs training_step")
  print("  3. Compliance % & violations/1k")
  print("  4. Zaps/1k & selectivity")
  print("  5. Permitted-color share & monoculture")
  print("  6. Return decomposition (R_env, -β, -c, α shaded)")
  print("  7. Collective cost histogram")
