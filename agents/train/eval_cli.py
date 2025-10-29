# Added by RST: Offline evaluation CLI for Phase 4
"""Offline evaluation CLI using Phase 3 evaluation harness.

This script loads a trained PPO checkpoint and runs the Phase 3 evaluation protocol:
1. Baseline: Resident-in-ego-slot (all 16 agents are residents)
2. Treatment: Ego + residents with PERMITTED_COLOR observation
3. Control: Ego + residents without PERMITTED_COLOR observation

Results are logged to W&B and saved locally.

Usage:
    # Evaluate treatment checkpoint
    python agents/train/eval_cli.py \
        --checkpoint checkpoints/ppo_treatment_step_1000000.zip \
        --arm treatment \
        --config configs/treatment.yaml \
        --output-dir eval_results

    # Evaluate with custom seeds
    python agents/train/eval_cli.py \
        --checkpoint checkpoints/ppo_control_final.zip \
        --arm control \
        --n-episodes 50 \
        --seeds 42,43,44,...
"""

import argparse
import os
from typing import Dict, Optional, Callable
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from agents.metrics.eval_harness import run_evaluation


def load_policy_from_checkpoint(
    checkpoint_path: str,
    vec_normalize_path: Optional[str] = None,
) -> Callable:
    """Load PPO policy from checkpoint.

    Args:
        checkpoint_path: Path to SB3 model checkpoint (.zip)
        vec_normalize_path: Path to VecNormalize stats (.pkl), optional

    Returns:
        Policy function: obs -> action
    """
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = PPO.load(checkpoint_path)

    # Load VecNormalize stats if provided
    # Note: VecNormalize is only used for reward normalization during training
    # For evaluation, we use raw observations and rewards
    if vec_normalize_path is not None and os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize stats from {vec_normalize_path}...")
        # VecNormalize stats are not needed for policy inference in eval
        # They were only used to normalize rewards during training

    def policy_fn(obs):
        """Policy function for evaluation.

        Args:
            obs: Observation dict from dmlab2d environment

        Returns:
            Action (integer 0-10)
        """
        # Convert dmlab2d observation to SB3 format
        # Phase 3 eval_harness passes dmlab2d observations directly
        # We need to convert to the format expected by our Gymnasium wrapper

        # Extract fields
        rgb = obs.get('RGB')  # (88, 88, 3) uint8
        ready_to_shoot = obs.get('READY_TO_SHOOT', 0.0)  # scalar
        permitted_color = obs.get('PERMITTED_COLOR')  # (3,) or None

        # Build observation dict for policy
        policy_obs = {
            'rgb': np.expand_dims(rgb, axis=0),  # Add batch dimension: (1, 88, 88, 3)
            'ready_to_shoot': np.array([[ready_to_shoot]], dtype=np.float32),  # (1, 1)
            'timestep': np.array([[0.0]], dtype=np.float32),  # TODO: track timestep in eval
        }

        if permitted_color is not None:
            policy_obs['permitted_color'] = np.expand_dims(permitted_color.astype(np.float32), axis=0)  # (1, 3)

        # Predict action (deterministic for evaluation)
        action, _states = model.predict(policy_obs, deterministic=True)

        return int(action[0])  # Return scalar action

    return policy_fn


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Configuration dict
    """
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Default eval config
        return {
            'permitted_color_index': 1,  # RED
            'startup_grey_grace': 25,
            'episode_timesteps': 2000,
            'altar_coords': (5, 15),
            'alpha': 0.0,  # Eval: alpha stripped
            'beta': 0.5,
            'c': 0.2,
            'immunity_cooldown': 200,
        }


def evaluate_checkpoint(
    checkpoint_path: str,
    arm: str,
    config: Dict,
    n_episodes: int = 20,
    seeds: Optional[list] = None,
    vec_normalize_path: Optional[str] = None,
    video_episodes: Optional[list] = None,
    video_output_dir: str = './eval_videos',
    output_dir: str = './eval_results',
):
    """Evaluate a trained checkpoint using Phase 3 harness.

    Args:
        checkpoint_path: Path to SB3 checkpoint
        arm: 'treatment' or 'control' (which arm was trained)
        config: Evaluation config dict
        n_episodes: Number of episodes per arm
        seeds: List of seeds for evaluation (optional)
        vec_normalize_path: Path to VecNormalize stats (optional)
        video_episodes: List of episode indices to render (optional)
        video_output_dir: Directory to save videos
        output_dir: Directory to save evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Phase 4 Offline Evaluation: {arm.upper()} Arm")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load policy from checkpoint
    policy_fn = load_policy_from_checkpoint(checkpoint_path, vec_normalize_path)

    # Generate seeds if not provided
    if seeds is None:
        rng = np.random.RandomState(12345)
        seeds = rng.randint(0, 1_000_000, size=n_episodes).tolist()

    # Run evaluation using Phase 3 harness
    print(f"Running evaluation with {n_episodes} episodes...")
    print(f"Seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''}")

    results = run_evaluation(
        ego_policy=policy_fn,
        config=config,
        num_episodes=n_episodes,
        seeds=seeds,
        video_episodes=video_episodes,
        video_output_dir=video_output_dir,
    )

    # Print results summary
    print(f"\n{'='*80}")
    print("Evaluation Results Summary")
    print(f"{'='*80}\n")

    for arm_name, run_metrics in results.items():
        print(f"{arm_name.upper()}:")
        print(f"  Value-gap (ΔV):        {run_metrics.value_gap_mean:.3f} ± {run_metrics.value_gap_std:.3f}")
        print(f"  Sanction-regret (SR):  {run_metrics.sanction_regret_events_mean:.1f} ± {run_metrics.sanction_regret_events_std:.1f}")
        print(f"  Compliance %:          {run_metrics.compliance_pct_mean:.1f} ± {run_metrics.compliance_pct_std:.1f}")
        print(f"  Violations/1k:         {run_metrics.violations_per_1k_mean:.1f} ± {run_metrics.violations_per_1k_std:.1f}")
        print(f"  R_eval:                {run_metrics.r_eval_mean:.1f} ± {run_metrics.r_eval_std:.1f}")
        print()

    # Save results to file
    results_path = os.path.join(output_dir, f'eval_results_{arm}.txt')
    print(f"Saving results to {results_path}")

    with open(results_path, 'w') as f:
        f.write("Phase 4 Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Arm: {arm}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Seeds: {seeds}\n\n")

        for arm_name, run_metrics in results.items():
            f.write(f"{arm_name.upper()}:\n")
            f.write(f"  Value-gap (ΔV):        {run_metrics.value_gap_mean:.3f} ± {run_metrics.value_gap_std:.3f}\n")
            f.write(f"  Sanction-regret (SR):  {run_metrics.sanction_regret_events_mean:.1f} ± {run_metrics.sanction_regret_events_std:.1f}\n")
            f.write(f"  Compliance %:          {run_metrics.compliance_pct_mean:.1f} ± {run_metrics.compliance_pct_std:.1f}\n")
            f.write(f"  Violations/1k:         {run_metrics.violations_per_1k_mean:.1f} ± {run_metrics.violations_per_1k_std:.1f}\n")
            f.write(f"  R_eval:                {run_metrics.r_eval_mean:.1f} ± {run_metrics.r_eval_std:.1f}\n")
            f.write("\n")

    # TODO: Log to W&B if enabled

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO checkpoint')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SB3 checkpoint (.zip)')
    parser.add_argument('--arm', type=str, required=True, choices=['treatment', 'control'],
                        help='Arm that was trained: treatment or control')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (optional)')
    parser.add_argument('--vec-normalize', type=str, default=None,
                        help='Path to VecNormalize stats (.pkl, optional)')
    parser.add_argument('--n-episodes', type=int, default=20,
                        help='Number of episodes per arm (default: 20)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated list of seeds (optional)')
    parser.add_argument('--video-episodes', type=str, default=None,
                        help='Comma-separated list of episode indices to render (optional)')
    parser.add_argument('--video-output-dir', type=str, default='./eval_videos',
                        help='Directory to save videos (default: ./eval_videos)')
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                        help='Directory to save results (default: ./eval_results)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Parse seeds
    seeds = None
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Parse video episodes
    video_episodes = None
    if args.video_episodes is not None:
        video_episodes = [int(e.strip()) for e in args.video_episodes.split(',')]

    # Run evaluation
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        arm=args.arm,
        config=config,
        n_episodes=args.n_episodes,
        seeds=seeds,
        vec_normalize_path=args.vec_normalize,
        video_episodes=video_episodes,
        video_output_dir=args.video_output_dir,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
