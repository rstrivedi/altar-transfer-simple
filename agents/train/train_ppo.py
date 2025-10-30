# Added by RST: Main training script for Phase 4 & 5 PPO
"""Train PPO agent with FiLM-conditioned two-head policy (Phase 4 & 5).

This script trains a single ego agent (agent 0) in presence of 15 scripted residents.
Supports treatment (with PERMITTED_COLOR) and control (without) arms.

Usage:
    # Phase 4: Single-community training
    python agents/train/train_ppo.py --arm treatment --config configs/treatment.yaml

    # Phase 5: Multi-community training (distributional competence)
    python agents/train/train_ppo.py --arm treatment --config configs/treatment_multi.yaml --multi-community

    # Custom hyperparameters
    python agents/train/train_ppo.py --arm treatment --total-timesteps 5000000 --n-envs 32 --multi-community

The script:
1. Creates vectorized environments (treatment or control, single or multi-community)
2. Instantiates FiLM two-head policy
3. Sets up W&B logging, checkpointing, and evaluation callbacks
4. Trains with PPO
5. Saves final model and VecNormalize stats
"""

import argparse
import os
from typing import Dict, Optional
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from agents.envs.sb3_wrapper import make_vec_env_treatment, make_vec_env_control, make_vec_env_multi_community
from agents.train.film_policy import FiLMTwoHeadPolicy


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
        # Default config
        return get_default_config()


def get_default_config() -> Dict:
    """Get default training configuration.

    Returns:
        Default config dict
    """
    return {
        # Environment config
        'env': {
            'permitted_color_index': 1,  # RED
            'startup_grey_grace': 25,
            'episode_timesteps': 2000,
            'altar_coords': (5, 15),
            'alpha': 0.5,  # Train-time bonus
            'beta': 0.5,   # Mis-zap penalty
            'c': 0.2,      # Zap cost
            'immunity_cooldown': 200,
        },
        # Training config
        'training': {
            'total_timesteps': 5_000_000,
            'n_envs': 16,
            'seed': 42,
            'learning_rate': 3e-4,
            'n_steps': 256,
            'batch_size': 2048,
            'n_epochs': 10,
            'gamma': 0.995,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,  # Default (will be overridden by head-wise)
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': False,
            'sde_sample_freq': -1,
        },
        # Policy config
        'policy': {
            'trunk_dim': 256,
            'sanction_hidden_dim': 128,
            'ent_coef_game': 0.01,
            'ent_coef_sanction': 0.02,
            'recurrent': False,  # Set to True for LSTM variant
            'lstm_hidden_size': 256,
        },
        # VecNormalize config
        'vec_normalize': {
            'norm_obs': False,  # Don't normalize RGB pixels
            'norm_reward': True,  # Normalize rewards
            'clip_obs': 10.0,
            'clip_reward': 10.0,
            'gamma': 0.995,
        },
        # Logging config
        'logging': {
            'wandb_project': 'altar-transfer',
            'wandb_entity': None,
            'wandb_run_name': None,
            'tensorboard_log': './logs/tensorboard',
            'log_interval': 10,
        },
        # Checkpointing config
        'checkpointing': {
            'save_freq': 50_000,  # Save every 50k steps
            'checkpoint_dir': './checkpoints',
            'save_vec_normalize': True,
        },
        # Evaluation config
        'evaluation': {
            'eval_freq': 100_000,  # Eval every 100k steps
            'n_eval_episodes': 20,
            'eval_seeds': None,  # Will generate if None
        },
    }


def make_vec_env(
    arm: str,
    config: Dict,
    n_envs: int,
    seed: int,
    enable_telemetry: bool = False,
    multi_community: bool = False,
    include_timestep: bool = False,
):
    """Create vectorized environment (treatment or control).

    Args:
        arm: 'treatment' or 'control'
        config: Environment configuration dict
        n_envs: Number of parallel environments
        seed: Base random seed
        enable_telemetry: Whether to enable MetricsRecorder
        multi_community: If True, use multi-community mode (Phase 5)
        include_timestep: Include normalized timestep in observations (default False)

    Returns:
        Vectorized environment
    """
    # Phase 5: Multi-community mode (independent sampling)
    if multi_community:
        vec_env = make_vec_env_multi_community(
            arm=arm,
            num_envs=n_envs,
            config=config,
            seed=seed,
            enable_telemetry=enable_telemetry,
            include_timestep=include_timestep,
        )
        return vec_env

    # Phase 4: Single-community mode
    # Generate seeds for each environment
    rng = np.random.RandomState(seed)
    env_seeds = rng.randint(0, 1_000_000, size=n_envs).tolist()

    if arm == 'treatment':
        vec_env = make_vec_env_treatment(
            num_envs=n_envs,
            config=config,
            seeds=env_seeds,
            enable_telemetry=enable_telemetry,
            include_timestep=include_timestep,
        )
    elif arm == 'control':
        vec_env = make_vec_env_control(
            num_envs=n_envs,
            config=config,
            seeds=env_seeds,
            enable_telemetry=enable_telemetry,
            include_timestep=include_timestep,
        )
    else:
        raise ValueError(f"arm must be 'treatment' or 'control', got {arm}")

    return vec_env


def train(
    arm: str,
    config: Dict,
    output_dir: str = './outputs',
    verbose: int = 1,
    multi_community: bool = False,
):
    """Train PPO agent.

    Args:
        arm: 'treatment' or 'control'
        config: Configuration dict
        output_dir: Directory to save outputs
        verbose: Verbosity level
        multi_community: If True, use multi-community mode (Phase 5)
    """
    print(f"\n{'='*80}")
    mode = "Phase 5 (Multi-Community)" if multi_community else "Phase 4 (Single-Community)"
    print(f"{mode} Training: {arm.upper()} Arm")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract configs
    env_config = config['env']
    training_config = config['training']
    policy_config = config['policy']
    vec_normalize_config = config['vec_normalize']
    logging_config = config['logging']
    checkpoint_config = config['checkpointing']

    # Create vectorized environment
    mode_str = "multi-community" if multi_community else "single-community"
    print(f"Creating {training_config['n_envs']} vectorized {mode_str} environments...")

    # Get include_timestep from env_config (default False to avoid temporal confounds)
    include_timestep = env_config.get('include_timestep', False)

    vec_env = make_vec_env(
        arm=arm,
        config=env_config,
        n_envs=training_config['n_envs'],
        seed=training_config['seed'],
        enable_telemetry=False,  # Disable telemetry during training for speed
        multi_community=multi_community,
        include_timestep=include_timestep,
    )

    # Wrap with VecNormalize (for reward normalization)
    if vec_normalize_config['norm_reward']:
        print("Wrapping with VecNormalize (reward normalization)...")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=vec_normalize_config['norm_obs'],
            norm_reward=vec_normalize_config['norm_reward'],
            clip_obs=vec_normalize_config['clip_obs'],
            clip_reward=vec_normalize_config['clip_reward'],
            gamma=vec_normalize_config['gamma'],
        )

    # Policy kwargs
    policy_kwargs = {
        'features_extractor_class': None,  # Will use default DictFeaturesExtractor from film_policy
        'trunk_dim': policy_config['trunk_dim'],
        'sanction_hidden_dim': policy_config['sanction_hidden_dim'],
        'ent_coef_game': policy_config['ent_coef_game'],
        'ent_coef_sanction': policy_config['ent_coef_sanction'],
        'net_arch': [],  # Not used (custom architecture in FiLMTwoHeadPolicy)
    }

    # Create PPO model
    print("Creating PPO model with FiLM two-head policy...")
    model = PPO(
        policy=FiLMTwoHeadPolicy,
        env=vec_env,
        learning_rate=training_config['learning_rate'],
        n_steps=training_config['n_steps'],
        batch_size=training_config['batch_size'],
        n_epochs=training_config['n_epochs'],
        gamma=training_config['gamma'],
        gae_lambda=training_config['gae_lambda'],
        clip_range=training_config['clip_range'],
        clip_range_vf=training_config['clip_range_vf'],
        ent_coef=training_config['ent_coef'],  # Default (overridden by head-wise in policy)
        vf_coef=training_config['vf_coef'],
        max_grad_norm=training_config['max_grad_norm'],
        use_sde=training_config['use_sde'],
        sde_sample_freq=training_config['sde_sample_freq'],
        tensorboard_log=logging_config.get('tensorboard_log'),
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=training_config['seed'],
    )

    print(f"\nModel created:")
    print(f"  Policy: {model.policy.__class__.__name__}")
    print(f"  Total timesteps: {training_config['total_timesteps']:,}")
    print(f"  Environments: {training_config['n_envs']}")
    print(f"  Steps per env: {training_config['n_steps']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Gamma: {training_config['gamma']}")

    # TODO: Setup callbacks (W&B logging, checkpointing, evaluation)
    # For now, train without callbacks
    callbacks = None

    # Train
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")

    model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=callbacks,
        log_interval=logging_config.get('log_interval', 10),
        tb_log_name=f"ppo_{arm}",
        reset_num_timesteps=True,
    )

    # Save final model
    final_model_path = os.path.join(output_dir, f'ppo_{arm}_final')
    print(f"\nSaving final model to {final_model_path}...")
    model.save(final_model_path)

    # Save VecNormalize stats
    if checkpoint_config['save_vec_normalize'] and isinstance(vec_env, VecNormalize):
        vec_normalize_path = os.path.join(output_dir, f'vec_normalize_{arm}.pkl')
        print(f"Saving VecNormalize stats to {vec_normalize_path}...")
        vec_env.save(vec_normalize_path)

    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}\n")

    # Close environment
    vec_env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train PPO agent for Phase 4 & 5')

    parser.add_argument('--arm', type=str, required=True, choices=['treatment', 'control'],
                        help='Training arm: treatment or control')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (optional)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs (default: ./outputs)')
    parser.add_argument('--total-timesteps', type=int, default=None,
                        help='Total training timesteps (overrides config)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Number of parallel environments (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--multi-community', action='store_true',
                        help='Enable multi-community mode (Phase 5)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=none, 1=info, 2=debug)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command-line args
    if args.total_timesteps is not None:
        config['training']['total_timesteps'] = args.total_timesteps
    if args.n_envs is not None:
        config['training']['n_envs'] = args.n_envs
    if args.seed is not None:
        config['training']['seed'] = args.seed

    # Train
    train(
        arm=args.arm,
        config=config,
        output_dir=args.output_dir,
        verbose=args.verbose,
        multi_community=args.multi_community,
    )


if __name__ == '__main__':
    main()
