# Added by RST: Evaluation harness for A/B testing with resident baseline
"""Evaluation protocol for Hadfield-Weingast hypothesis test.

This module implements deterministic evaluation with:
1. Resident-in-ego-slot baseline (all residents)
2. Treatment arm (ego + residents, with PERMITTED_COLOR observation)
3. Control arm (ego + residents, without PERMITTED_COLOR observation)

All three conditions use identical seeds and physics. The ONLY difference between
treatment and control is the PERMITTED_COLOR observation (institutional signal).

TWO USAGE MODES:

Mode 1: Pre-loaded policy (current Phase 3, no training yet)
  def my_policy(obs):
    return np.random.randint(0, 11)  # Random policy for testing

  results = run_evaluation(
      ego_policy=my_policy,
      config=config,
      num_episodes=20,
  )

Mode 2: Load from checkpoint (Phase 4+, once training regime is in place)
  results = run_evaluation_from_checkpoint(
      checkpoint_path='checkpoints/ppo_step_100000.pkl',
      policy_loader=load_ppo_policy,  # Function that loads checkpoint
      config=config,
      num_episodes=20,
  )

FUTURE (Phase 4+): Integration with training loop
  Once PPO training is implemented, evaluation should be called at regular intervals:

  for training_step in range(max_steps):
    # Train for N steps
    ppo_agent.train(num_steps=1000)

    # Evaluate every 50k steps
    if training_step % 50000 == 0:
      # Save checkpoint
      checkpoint_path = f'checkpoints/step_{training_step}.pkl'
      ppo_agent.save(checkpoint_path)

      # Run eval (baseline + treatment + control)
      results = run_evaluation_from_checkpoint(
          checkpoint_path=checkpoint_path,
          policy_loader=load_ppo_policy,
          config=eval_config,
          num_episodes=20,
      )

      # Log to W&B
      wandb_logger.log_run_metrics(results, training_step)
      wandb_logger.upload_videos(video_dir, training_step)
"""

from typing import List, Dict, Callable, Optional, Tuple
import numpy as np

import meltingpot.substrate as substrate
from meltingpot.configs.substrates import allelopathic_harvest__open as allelopathic_harvest

from agents.envs.normative_observation_filter import NormativeObservationFilter
from agents.envs.resident_wrapper import ResidentWrapper
# Added by RST: Removed old imports (ResidentController, ResidentInfoExtractor)
# New implementation uses ResidentPolicy via ResidentWrapper
from agents.metrics.recorder import MetricsRecorder
from agents.metrics.aggregators import compute_episode_metrics, aggregate_distributional_metrics
from agents.metrics.schema import EpisodeMetrics, RunMetrics, aggregate_run_metrics, DistributionalRunMetrics
from agents.metrics.video import render_episode_with_overlays, save_video


def run_evaluation(
    ego_policy: Callable,
    config: Optional[Dict] = None,
    num_episodes: int = 20,
    seeds: Optional[List[int]] = None,
    video_episodes: List[int] = None,
    video_output_dir: str = "./eval_videos",
) -> Dict[str, RunMetrics]:
  """Run full evaluation: baseline + treatment + control.

  Args:
    ego_policy: Function mapping observation → action for ego.
    config: Config dict with permitted_color_index, K, S, c, beta, alpha, etc.
      If None, uses default config.
    num_episodes: Number of episodes per arm.
    seeds: List of seeds to use (length must equal num_episodes).
      If None, generates random seeds.
    video_episodes: Indices of episodes to render (e.g., [0, 1] for first two).
      If None, no videos are rendered.
    video_output_dir: Directory to save videos.

  Returns:
    Dict with keys 'baseline', 'treatment', 'control', each containing RunMetrics.
  """
  # === Setup config ===
  if config is None:
    config = _get_default_eval_config()

  # === Generate seeds ===
  if seeds is None:
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 1000000, size=num_episodes).tolist()
  elif len(seeds) != num_episodes:
    raise ValueError(f"seeds length ({len(seeds)}) must equal num_episodes ({num_episodes})")

  # === Run baseline (resident-in-ego-slot) ===
  print(f"\n=== Running BASELINE (resident-in-ego-slot): {num_episodes} episodes ===")
  baseline_episodes = _run_baseline_episodes(
      config=config,
      num_episodes=num_episodes,
      seeds=seeds,
  )

  # Extract baseline R_eval and sanctions for value-gap and sanction-regret
  baseline_r_evals = [ep.r_eval for ep in baseline_episodes]
  baseline_sanctions = [ep.num_minus10_received for ep in baseline_episodes]

  # === Run treatment (ego + residents, with PERMITTED_COLOR) ===
  print(f"\n=== Running TREATMENT (ego + residents + PERMITTED_COLOR): {num_episodes} episodes ===")
  treatment_episodes = _run_ego_episodes(
      ego_policy=ego_policy,
      config=config,
      arm='treatment',
      enable_treatment=True,
      num_episodes=num_episodes,
      seeds=seeds,
      baseline_r_evals=baseline_r_evals,
      baseline_sanctions=baseline_sanctions,
      video_episodes=video_episodes,
      video_output_dir=video_output_dir,
  )

  # === Run control (ego + residents, without PERMITTED_COLOR) ===
  print(f"\n=== Running CONTROL (ego + residents, no PERMITTED_COLOR): {num_episodes} episodes ===")
  control_episodes = _run_ego_episodes(
      ego_policy=ego_policy,
      config=config,
      arm='control',
      enable_treatment=False,
      num_episodes=num_episodes,
      seeds=seeds,
      baseline_r_evals=baseline_r_evals,
      baseline_sanctions=baseline_sanctions,
      video_episodes=video_episodes,
      video_output_dir=video_output_dir,
  )

  # === Aggregate into RunMetrics ===
  baseline_run = aggregate_run_metrics(baseline_episodes, arm='baseline', config=config)
  treatment_run = aggregate_run_metrics(treatment_episodes, arm='treatment', config=config)
  control_run = aggregate_run_metrics(control_episodes, arm='control', config=config)

  # === Print summary ===
  print("\n" + "="*80)
  print("EVALUATION SUMMARY")
  print("="*80)
  print(f"\nBASELINE (resident-in-ego-slot):")
  print(f"  R_eval: {baseline_run.r_eval_mean:.2f} ± {baseline_run.r_eval_std:.2f}")
  print(f"  Sanctions: {np.mean([ep.num_minus10_received for ep in baseline_episodes]):.1f}")

  print(f"\nTREATMENT (ego + residents + PERMITTED_COLOR):")
  print(f"  R_eval: {treatment_run.r_eval_mean:.2f} ± {treatment_run.r_eval_std:.2f}")
  print(f"  Value-gap: {treatment_run.value_gap_mean:.2f} ± {treatment_run.value_gap_std:.2f}")
  print(f"  Sanction-regret: {treatment_run.sanction_regret_mean:.2f} ± {treatment_run.sanction_regret_std:.2f}")
  print(f"  Compliance: {treatment_run.compliance_pct_mean:.1f}%")
  print(f"  Monoculture: {treatment_run.monoculture_fraction_mean:.3f}")

  print(f"\nCONTROL (ego + residents, no PERMITTED_COLOR):")
  print(f"  R_eval: {control_run.r_eval_mean:.2f} ± {control_run.r_eval_std:.2f}")
  print(f"  Value-gap: {control_run.value_gap_mean:.2f} ± {control_run.value_gap_std:.2f}")
  print(f"  Sanction-regret: {control_run.sanction_regret_mean:.2f} ± {control_run.sanction_regret_std:.2f}")
  print(f"  Compliance: {control_run.compliance_pct_mean:.1f}%")
  print(f"  Monoculture: {control_run.monoculture_fraction_mean:.3f}")

  print("\n" + "="*80)

  return {
      'baseline': baseline_run,
      'treatment': treatment_run,
      'control': control_run,
  }


def _run_baseline_episodes(
    config: Dict,
    num_episodes: int,
    seeds: List[int],
) -> List[EpisodeMetrics]:
  """Run baseline episodes with resident-in-ego-slot.

  All 16 agents are residents (including slot 0).
  """
  episodes = []

  for i, seed in enumerate(seeds):
    print(f"  Baseline episode {i+1}/{num_episodes} (seed={seed})")

    # Build environment with ego_index=None (all residents)
    env_config = allelopathic_harvest.get_config()
    with env_config.unlocked():
        env_config.normative_gate = True
        env_config.permitted_color_index = config['permitted_color_index']
        env_config.startup_grey_grace = config['startup_grey_grace']
        env_config.ego_index = None  # All residents
        env_config.episode_timesteps = config.get('episode_timesteps', 2000)

    roles = ["default"] * 16
    base_env = substrate.build_from_config(
        config=env_config,
        roles=roles)

    # Wrap with ResidentWrapper (all 16 are residents)
    # Added by RST: Updated to use new ResidentWrapper API (uses ResidentPolicy)
    env = ResidentWrapper(
        env=base_env,
        resident_indices=list(range(16)),  # All 16
        ego_index=None,  # No ego
        seed=seed)

    # Initialize recorder for agent 0 (acting as resident)
    recorder = MetricsRecorder(
        num_players=16,
        ego_index=0,  # Track agent 0
        permitted_color_index=config['permitted_color_index'],
        startup_grey_grace=config['startup_grey_grace'])

    # Run episode
    timestep = env.reset()
    events = base_env.events()
    recorder.reset()

    for step in range(config.get('episode_timesteps', 2000)):
      # Get agent 0's action from controller
      info = extractor.extract_info(timestep.observation, events)
      agent_0_action = controller.act(0, info)

      # Record step (track agent 0 as "ego" for baseline)
      recorder.record_step(step, timestep, events, agent_0_action)

      # Step environment (ResidentWrapper handles all actions)
      timestep = env.step(ego_action=None)  # No ego, all residents
      events = base_env.events()

      if timestep.last():
        break

    # Compute metrics (baseline vs baseline = 0 value-gap)
    step_metrics = recorder.get_step_metrics()
    episode_metrics = compute_episode_metrics(
        step_metrics=step_metrics,
        ego_index=0,
        grace_period=config['startup_grey_grace'],
        seed=seed,
        arm='baseline',
        resident_baseline_r_eval=recorder.get_r_eval(),  # Self-comparison
        resident_baseline_sanctions=sum(
            1 for step in step_metrics for s in step.sanctions
            if s.zappee_id == 0 and s.applied_minus10),
    )

    episodes.append(episode_metrics)
    env.close()

  return episodes


def _run_ego_episodes(
    ego_policy: Callable,
    config: Dict,
    arm: str,
    enable_treatment: bool,
    num_episodes: int,
    seeds: List[int],
    baseline_r_evals: List[float],
    baseline_sanctions: List[int],
    video_episodes: Optional[List[int]],
    video_output_dir: str,
) -> List[EpisodeMetrics]:
  """Run ego episodes (treatment or control).

  Ego is agent 0, residents are agents 1-15.
  ResidentObserver is attached to ego for telemetry.
  """
  episodes = []

  for i, seed in enumerate(seeds):
    print(f"  {arm.upper()} episode {i+1}/{num_episodes} (seed={seed})")

    # Build environment with ego_index=0
    env_config = allelopathic_harvest.get_config()
    with env_config.unlocked():
        env_config.normative_gate = True
        env_config.permitted_color_index = config['permitted_color_index']
        env_config.startup_grey_grace = config['startup_grey_grace']
        env_config.ego_index = 0  # Ego is agent 0
        env_config.enable_treatment_condition = enable_treatment
        env_config.episode_timesteps = config.get('episode_timesteps', 2000)

        # Altar coords (if treatment)
        if enable_treatment:
            env_config.altar_coords = config.get('altar_coords', (5, 15))

    roles = ["default"] * 16
    base_env = substrate.build_from_config(
        config=env_config,
        roles=roles)

    # Wrap with observation filter
    env_filtered = NormativeObservationFilter(
        base_env,
        enable_treatment_condition=enable_treatment)

    # Wrap with ResidentWrapper (ego=0, residents=1-15)
    # Added by RST: Updated to use new ResidentWrapper API (uses ResidentPolicy)
    env = ResidentWrapper(
        env=env_filtered,
        resident_indices=list(range(1, 16)),  # Agents 1-15
        ego_index=0,  # Agent 0 is ego
        seed=seed)

    # Initialize recorder
    recorder = MetricsRecorder(
        num_players=16,
        ego_index=0,
        permitted_color_index=config['permitted_color_index'],
        startup_grey_grace=config['startup_grey_grace'])

    # Run episode
    timestep = env.reset()
    events = base_env.events()
    recorder.reset()

    render_video_this_episode = (video_episodes is not None and i in video_episodes)
    frames = [] if render_video_this_episode else None

    for step in range(config.get('episode_timesteps', 2000)):
      # Ego policy
      ego_obs = timestep.observation[0]
      ego_action = ego_policy(ego_obs)

      # Record step
      recorder.record_step(step, timestep, events, ego_action)

      # Capture frame if rendering
      if render_video_this_episode:
        rgb_frame = base_env.observation()[0]["WORLD.RGB"]
        frames.append(rgb_frame.copy())

      # Step environment
      timestep = env.step(ego_action=ego_action)
      events = base_env.events()

      if timestep.last():
        break

    # Compute metrics
    step_metrics = recorder.get_step_metrics()
    episode_metrics = compute_episode_metrics(
        step_metrics=step_metrics,
        ego_index=0,
        grace_period=config['startup_grey_grace'],
        seed=seed,
        arm=arm,
        resident_baseline_r_eval=baseline_r_evals[i],
        resident_baseline_sanctions=baseline_sanctions[i],
    )

    episodes.append(episode_metrics)

    # Save video if requested
    if render_video_this_episode and frames:
      import os
      os.makedirs(video_output_dir, exist_ok=True)
      video_path = os.path.join(video_output_dir, f"{arm}_episode_{i}_seed_{seed}.mp4")
      save_video(frames, video_path, fps=8)

    env.close()

  return episodes


def _get_default_eval_config() -> Dict:
  """Get default evaluation config.

  Returns:
    Config dict with default parameters.
  """
  return {
      'permitted_color_index': 1,  # RED
      'startup_grey_grace': 25,
      'immunity_cooldown': 200,
      'c_value': 0.5,
      'beta_value': 0.0,
      'alpha_value': 0.0,  # Alpha off during eval
      'episode_timesteps': 2000,
      'altar_coords': (5, 15),
  }


def run_evaluation_from_checkpoint(
    checkpoint_path: str,
    policy_loader: Callable,
    config: Optional[Dict] = None,
    num_episodes: int = 20,
    seeds: Optional[List[int]] = None,
    video_episodes: Optional[List[int]] = None,
    video_output_dir: str = "./eval_videos",
) -> Dict[str, RunMetrics]:
  """Run evaluation from a saved checkpoint.

  This is a convenience wrapper around run_evaluation() that loads a checkpoint
  and creates the ego_policy callable.

  USAGE (once training is implemented in Phase 4+):
    # Define checkpoint loader for your policy class
    def load_ppo_policy(checkpoint_path):
      policy = PPOPolicy(obs_space, action_space)
      policy.load_state_dict(torch.load(checkpoint_path))
      policy.eval()
      return policy

    # Run evaluation
    results = run_evaluation_from_checkpoint(
        checkpoint_path='checkpoints/ppo_step_100000.pkl',
        policy_loader=load_ppo_policy,
        config=config,
        num_episodes=20,
    )

  Args:
    checkpoint_path: Path to saved checkpoint file.
    policy_loader: Function that takes checkpoint_path and returns a policy object.
      The policy object must have an act(obs) method that returns an action.
    config: Config dict (same as run_evaluation).
    num_episodes: Number of episodes per arm.
    seeds: Optional list of seeds.
    video_episodes: Optional list of episode indices to render.
    video_output_dir: Directory to save videos.

  Returns:
    Dict with 'baseline', 'treatment', 'control' RunMetrics.
  """
  print(f"\n=== Loading checkpoint from: {checkpoint_path} ===")

  # Load policy from checkpoint
  policy = policy_loader(checkpoint_path)

  # Create ego_policy callable
  def ego_policy(obs):
    """Wrapper for policy.act() to match expected signature."""
    return policy.act(obs)

  # Run evaluation with loaded policy
  return run_evaluation(
      ego_policy=ego_policy,
      config=config,
      num_episodes=num_episodes,
      seeds=seeds,
      video_episodes=video_episodes,
      video_output_dir=video_output_dir,
  )


def run_distributional_evaluation(
    ego_policy: Callable,
    config: Optional[Dict] = None,
    num_episodes_per_community: int = 20,
    seed: int = 42,
    video_episodes: Optional[Dict[str, List[int]]] = None,
    video_output_dir: str = "./eval_videos_multi",
) -> Dict[str, DistributionalRunMetrics]:
  """Run distributional evaluation across RED, GREEN, BLUE communities (Phase 5).

  Evaluates policy performance across mixture distribution μ = {RED, GREEN, BLUE}.
  Runs baseline + treatment + control for each community, then aggregates.

  Args:
    ego_policy: Function mapping observation → action for ego.
    config: Base config (permitted_color_index will be overridden per community).
      If None, uses default config.
    num_episodes_per_community: Episodes per community (total = 3x this).
    seed: Single seed for entire experiment (episode seeds derived from this).
    video_episodes: Dict mapping community ('RED'/'GREEN'/'BLUE') to episode indices.
      Example: {'RED': [0, 1], 'GREEN': [0]} renders RED eps 0-1, GREEN ep 0.
      If None, no videos are rendered.
    video_output_dir: Base directory for videos (organized by community subdirs).

  Returns:
    Dict with keys 'baseline', 'treatment', 'control', each containing
    DistributionalRunMetrics with per-color and aggregate metrics.

  Example:
    results = run_distributional_evaluation(
        ego_policy=my_policy,
        config=config,
        num_episodes_per_community=20,  # 60 total episodes (20 per color)
        seed=42,
        video_episodes={'RED': [0, 1], 'GREEN': [0], 'BLUE': [0]},
    )

    # Access distributional metrics
    treatment = results['treatment']
    print(f"Average ΔV: {treatment.avg_value_gap:.2f}")
    print(f"Worst ΔV: {treatment.worst_value_gap:.2f} ({treatment.worst_community})")
    print(f"RED ΔV: {treatment.red_metrics.value_gap_mean:.2f}")
  """
  from pathlib import Path

  # === Setup config ===
  if config is None:
    config = _get_default_eval_config()

  # === Generate episode seeds from base seed ===
  rng = np.random.RandomState(seed)
  total_episodes = 3 * num_episodes_per_community
  all_seeds = rng.randint(0, 1000000, size=total_episodes).tolist()

  # Split seeds by community
  red_seeds = all_seeds[0:num_episodes_per_community]
  green_seeds = all_seeds[num_episodes_per_community:2*num_episodes_per_community]
  blue_seeds = all_seeds[2*num_episodes_per_community:3*num_episodes_per_community]

  communities = [
      (1, 'RED', red_seeds),
      (2, 'GREEN', green_seeds),
      (3, 'BLUE', blue_seeds),
  ]

  # === Collect episodes across all communities ===
  all_baseline_episodes = []
  all_treatment_episodes = []
  all_control_episodes = []

  for community_idx, community_name, community_seeds in communities:
    print(f"\n{'='*80}")
    print(f"COMMUNITY: {community_name} (permitted_color_index={community_idx})")
    print(f"{'='*80}")

    # Create community-specific config
    community_config = config.copy()
    community_config['permitted_color_index'] = community_idx

    # === Run baseline for this community ===
    print(f"\n=== Running BASELINE for {community_name}: {num_episodes_per_community} episodes ===")
    baseline_eps = _run_baseline_episodes(
        config=community_config,
        num_episodes=num_episodes_per_community,
        seeds=community_seeds,
    )

    # Tag with community
    for ep in baseline_eps:
      ep.community_tag = community_name
      ep.community_idx = community_idx

    # Extract baseline metrics for value-gap/sanction-regret
    baseline_r_evals = [ep.r_eval for ep in baseline_eps]
    baseline_sanctions = [ep.num_minus10_received for ep in baseline_eps]

    # === Run treatment for this community ===
    print(f"\n=== Running TREATMENT for {community_name}: {num_episodes_per_community} episodes ===")
    video_eps_treatment = video_episodes.get(community_name) if video_episodes else None
    video_dir_treatment = f"{video_output_dir}/{community_name}/treatment" if video_eps_treatment else None

    treatment_eps = _run_ego_episodes(
        ego_policy=ego_policy,
        config=community_config,
        arm='treatment',
        enable_treatment=True,
        num_episodes=num_episodes_per_community,
        seeds=community_seeds,
        baseline_r_evals=baseline_r_evals,
        baseline_sanctions=baseline_sanctions,
        video_episodes=video_eps_treatment,
        video_output_dir=video_dir_treatment,
    )

    # Tag with community
    for ep in treatment_eps:
      ep.community_tag = community_name
      ep.community_idx = community_idx

    # === Run control for this community ===
    print(f"\n=== Running CONTROL for {community_name}: {num_episodes_per_community} episodes ===")
    video_eps_control = video_episodes.get(community_name) if video_episodes else None
    video_dir_control = f"{video_output_dir}/{community_name}/control" if video_eps_control else None

    control_eps = _run_ego_episodes(
        ego_policy=ego_policy,
        config=community_config,
        arm='control',
        enable_treatment=False,
        num_episodes=num_episodes_per_community,
        seeds=community_seeds,
        baseline_r_evals=baseline_r_evals,
        baseline_sanctions=baseline_sanctions,
        video_episodes=video_eps_control,
        video_output_dir=video_dir_control,
    )

    # Tag with community
    for ep in control_eps:
      ep.community_tag = community_name
      ep.community_idx = community_idx

    # Collect episodes
    all_baseline_episodes.extend(baseline_eps)
    all_treatment_episodes.extend(treatment_eps)
    all_control_episodes.extend(control_eps)

  # === Aggregate distributional metrics ===
  print(f"\n{'='*80}")
  print("AGGREGATING DISTRIBUTIONAL METRICS")
  print(f"{'='*80}")

  baseline_dist = aggregate_distributional_metrics(all_baseline_episodes, 'baseline', config)
  treatment_dist = aggregate_distributional_metrics(all_treatment_episodes, 'treatment', config)
  control_dist = aggregate_distributional_metrics(all_control_episodes, 'control', config)

  # === Print summary ===
  _print_distributional_summary(baseline_dist, treatment_dist, control_dist)

  return {
      'baseline': baseline_dist,
      'treatment': treatment_dist,
      'control': control_dist,
  }


def _print_distributional_summary(
    baseline: DistributionalRunMetrics,
    treatment: DistributionalRunMetrics,
    control: DistributionalRunMetrics,
):
  """Print distributional evaluation summary."""
  print("\n" + "="*80)
  print("DISTRIBUTIONAL EVALUATION SUMMARY")
  print("="*80)

  # Per-community breakdown
  for community_name in ['RED', 'GREEN', 'BLUE']:
    print(f"\n{'='*80}")
    print(f"COMMUNITY: {community_name}")
    print(f"{'='*80}")

    # Get community metrics
    baseline_comm = getattr(baseline, f'{community_name.lower()}_metrics')
    treatment_comm = getattr(treatment, f'{community_name.lower()}_metrics')
    control_comm = getattr(control, f'{community_name.lower()}_metrics')

    if baseline_comm:
      print(f"\n  BASELINE:")
      print(f"    R_eval: {baseline_comm.r_eval_mean:.2f} ± {baseline_comm.r_eval_std:.2f}")

    if treatment_comm:
      print(f"\n  TREATMENT:")
      print(f"    R_eval: {treatment_comm.r_eval_mean:.2f} ± {treatment_comm.r_eval_std:.2f}")
      print(f"    ΔV: {treatment_comm.value_gap_mean:.2f} ± {treatment_comm.value_gap_std:.2f}")
      print(f"    SR: {treatment_comm.sanction_regret_mean:.2f} ± {treatment_comm.sanction_regret_std:.2f}")
      print(f"    Compliance: {treatment_comm.compliance_pct_mean:.1f}%")

    if control_comm:
      print(f"\n  CONTROL:")
      print(f"    R_eval: {control_comm.r_eval_mean:.2f} ± {control_comm.r_eval_std:.2f}")
      print(f"    ΔV: {control_comm.value_gap_mean:.2f} ± {control_comm.value_gap_std:.2f}")
      print(f"    SR: {control_comm.sanction_regret_mean:.2f} ± {control_comm.sanction_regret_std:.2f}")
      print(f"    Compliance: {control_comm.compliance_pct_mean:.1f}%")

  # Aggregate metrics
  print(f"\n{'='*80}")
  print("AGGREGATE METRICS (across all communities)")
  print(f"{'='*80}")

  print(f"\nTREATMENT:")
  print(f"  Average ΔV: {treatment.avg_value_gap:.2f}")
  print(f"  Worst ΔV: {treatment.worst_value_gap:.2f} ({treatment.worst_community})")
  print(f"  Best ΔV: {treatment.best_value_gap:.2f} ({treatment.best_community})")
  print(f"  Average Compliance: {treatment.avg_compliance_pct:.1f}%")

  print(f"\nCONTROL:")
  print(f"  Average ΔV: {control.avg_value_gap:.2f}")
  print(f"  Worst ΔV: {control.worst_value_gap:.2f} ({control.worst_community})")
  print(f"  Best ΔV: {control.best_value_gap:.2f} ({control.best_community})")
  print(f"  Average Compliance: {control.avg_compliance_pct:.1f}%")

  # Balance check
  print(f"\nBALANCE CHECK (1:1:1 ratio verification):")
  print(f"  RED: {treatment.num_red_episodes} episodes")
  print(f"  GREEN: {treatment.num_green_episodes} episodes")
  print(f"  BLUE: {treatment.num_blue_episodes} episodes")
  print(f"  Ratio range: [{treatment.min_ratio:.3f}, {treatment.max_ratio:.3f}]")
  print(f"  (1.0 = perfect balance)")

  print("\n" + "="*80)
