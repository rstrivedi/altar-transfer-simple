# Added by RST: Main training script for Phase 4 & 5 PPO
"""Train PPO agent with FiLM-conditioned two-head policy (Phase 4 & 5).

This script trains a single ego agent (agent 0) in presence of 15 scripted residents.
Supports treatment (with permitted_color from ALTAR) and control (without) arms.

Usage:
    # Basic usage (config file required if not overriding all params)
    python agents/train/train_ppo.py treatment --config configs/treatment.yaml

    # Override key params via command line (cluster-friendly!)
    python agents/train/train_ppo.py treatment \
        --config configs/treatment.yaml \
        --seed 42 \
        --total-timesteps 500000 \
        --wandb-entity username \
        --wandb-project my-project \
        --output-dir ./outputs/run1

    # Multiple runs for cluster (no YAML editing!)
    for seed in 42 43 44; do
        python agents/train/train_ppo.py treatment \
            --config configs/treatment.yaml \
            --seed $seed \
            --wandb-entity username \
            --output-dir ./outputs/treatment_seed${seed}
    done

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

# Import RecurrentPPO from sb3-contrib
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
    RecurrentPPO = None
from stable_baselines3.common.callbacks import CallbackList

from agents.envs.sb3_wrapper import make_vec_env_treatment, make_vec_env_control, make_vec_env_multi_community
from agents.train.film_policy import FiLMTwoHeadPolicy, DictFeaturesExtractor, RecurrentFiLMTwoHeadPolicy, RecurrentDictFeaturesExtractor
from agents.train.callbacks import WandbLoggingCallback, CheckpointCallback, EvalCallback


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
    resume_from: Optional[str] = None,
    resume_vec_normalize: Optional[str] = None,
):
    """Train PPO agent.

    Args:
        arm: 'treatment' or 'control'
        config: Configuration dict
        output_dir: Directory to save outputs
        verbose: Verbosity level
        multi_community: If True, use multi-community mode (Phase 5)
        resume_from: Path to checkpoint .zip file to resume training from
        resume_vec_normalize: Path to VecNormalize .pkl file (auto-detected if not provided)
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
        enable_telemetry=True,  # Enable telemetry for metrics logging
        multi_community=multi_community,
        include_timestep=include_timestep,
    )

    # Wrap with VecNormalize (for reward normalization) or load from checkpoint
    if vec_normalize_config['norm_reward']:
        if resume_from is not None and resume_vec_normalize is not None:
            # Resuming: Load VecNormalize stats from checkpoint
            print(f"Loading VecNormalize stats from {resume_vec_normalize}...")
            vec_env = VecNormalize.load(resume_vec_normalize, vec_env)
            # Keep training mode (don't freeze stats)
            vec_env.training = True
            vec_env.norm_reward = True
        elif resume_from is not None and resume_vec_normalize is None:
            # Auto-detect VecNormalize file based on checkpoint path
            # Checkpoint: checkpoints/ppo_treatment_5000000_steps.zip
            # VecNormalize: checkpoints/vec_normalize_ppo_treatment_5000000_steps.pkl
            checkpoint_dir = os.path.dirname(resume_from)
            checkpoint_name = os.path.basename(resume_from).replace('.zip', '')
            vec_normalize_path = os.path.join(checkpoint_dir, f'vec_normalize_{checkpoint_name}.pkl')

            if os.path.exists(vec_normalize_path):
                print(f"Auto-detected VecNormalize stats at {vec_normalize_path}")
                vec_env = VecNormalize.load(vec_normalize_path, vec_env)
                vec_env.training = True
                vec_env.norm_reward = True
            else:
                print(f"WARNING: Could not find VecNormalize stats at {vec_normalize_path}")
                print("Creating new VecNormalize (this may cause instability when resuming)")
                vec_env = VecNormalize(
                    vec_env,
                    norm_obs=vec_normalize_config['norm_obs'],
                    norm_reward=vec_normalize_config['norm_reward'],
                    clip_obs=vec_normalize_config['clip_obs'],
                    clip_reward=vec_normalize_config['clip_reward'],
                    gamma=vec_normalize_config['gamma'],
                )
        else:
            # Normal training: Create new VecNormalize
            print("Wrapping with VecNormalize (reward normalization)...")
            vec_env = VecNormalize(
                vec_env,
                norm_obs=vec_normalize_config['norm_obs'],
                norm_reward=vec_normalize_config['norm_reward'],
                clip_obs=vec_normalize_config['clip_obs'],
                clip_reward=vec_normalize_config['clip_reward'],
                gamma=vec_normalize_config['gamma'],
            )

    # Check if recurrent mode is enabled
    is_recurrent = policy_config.get('recurrent', False)

    # Load model from checkpoint if resuming, otherwise create new model
    if resume_from is not None:
        print(f"\nResuming training from checkpoint: {resume_from}")

        # Load the appropriate model class based on recurrent flag
        if is_recurrent:
            if not HAS_RECURRENT_PPO:
                raise ImportError(
                    "RecurrentPPO requires sb3-contrib. Install with: pip install sb3-contrib"
                )
            print("Loading RecurrentPPO model from checkpoint...")
            model = RecurrentPPO.load(
                resume_from,
                env=vec_env,
                verbose=verbose,
            )
        else:
            print("Loading PPO model from checkpoint...")
            model = PPO.load(
                resume_from,
                env=vec_env,
                verbose=verbose,
            )

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Training will continue from previous timestep")

    # Configure policy and algorithm based on recurrent flag
    elif is_recurrent:
        # Recurrent mode: Use RecurrentPPO with proper LSTM state management
        if not HAS_RECURRENT_PPO:
            raise ImportError(
                "RecurrentPPO requires sb3-contrib. Install with: pip install sb3-contrib"
            )

        print(f"Creating RecurrentPPO model with proper LSTM state management...")

        # Recurrent policy kwargs
        policy_kwargs = {
            'features_extractor_class': RecurrentDictFeaturesExtractor,
            'features_extractor_kwargs': {
                'lstm_hidden_size': policy_config.get('lstm_hidden_size', 256),
                'trunk_dim': policy_config['trunk_dim'],
            },
            'trunk_dim': policy_config['trunk_dim'],
            'lstm_hidden_size': policy_config.get('lstm_hidden_size', 256),
            'sanction_hidden_dim': policy_config['sanction_hidden_dim'],
            'ent_coef_game': policy_config['ent_coef_game'],
            'ent_coef_sanction': policy_config['ent_coef_sanction'],
        }

        model = RecurrentPPO(
            policy=RecurrentFiLMTwoHeadPolicy,
            env=vec_env,
            learning_rate=training_config['learning_rate'],
            n_steps=training_config['n_steps'],
            batch_size=training_config['batch_size'],
            n_epochs=training_config['n_epochs'],
            gamma=training_config['gamma'],
            gae_lambda=training_config['gae_lambda'],
            clip_range=training_config['clip_range'],
            clip_range_vf=training_config['clip_range_vf'],
            ent_coef=training_config['ent_coef'],
            vf_coef=training_config['vf_coef'],
            max_grad_norm=training_config['max_grad_norm'],
            tensorboard_log=logging_config.get('tensorboard_log'),
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=training_config['seed'],
        )
    else:
        # Non-recurrent mode: Use regular PPO with feedforward policy (no LSTM)
        print(f"Creating PPO model with feedforward FiLM two-head policy (no LSTM)...")

        # Feedforward policy kwargs
        policy_kwargs = {
            'features_extractor_class': DictFeaturesExtractor,
            'features_extractor_kwargs': {
                'trunk_dim': policy_config['trunk_dim'],
            },
            'trunk_dim': policy_config['trunk_dim'],
            'sanction_hidden_dim': policy_config['sanction_hidden_dim'],
            'ent_coef_game': policy_config['ent_coef_game'],
            'ent_coef_sanction': policy_config['ent_coef_sanction'],
            'net_arch': [],  # Not used (custom architecture in FiLMTwoHeadPolicy)
        }

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
            ent_coef=training_config['ent_coef'],
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

    # Setup callbacks (W&B logging, checkpointing, evaluation)
    print(f"\n{'='*80}")
    print("Setting up callbacks...")
    print(f"{'='*80}\n")

    callback_list = []

    # 1. W&B Logging Callback
    wandb_callback = WandbLoggingCallback(
        config=config,
        arm=arm,
        multi_community=multi_community,
        project=logging_config.get('wandb_project', 'altar-transfer'),
        entity=logging_config.get('wandb_entity', None),
        run_name=logging_config.get('wandb_run_name', None),
        tags=logging_config.get('wandb_tags', None),
        log_interval=logging_config.get('log_interval', 2560),
        verbose=verbose,
    )
    callback_list.append(wandb_callback)

    # 2. Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_config['save_freq'],
        checkpoint_dir=checkpoint_config['checkpoint_dir'],
        name_prefix=f'ppo_{arm}',
        save_vec_normalize=checkpoint_config.get('save_vec_normalize', True),
        verbose=verbose,
    )
    callback_list.append(checkpoint_callback)

    # 3. Evaluation Callback (also tracks and saves best models)
    eval_config = config.get('evaluation', {})
    if eval_config.get('eval_freq', 0) > 0:
        eval_callback = EvalCallback(
            eval_freq=eval_config['eval_freq'],
            n_eval_episodes=eval_config.get('n_eval_episodes', 20),
            config=env_config,
            arm=arm,
            multi_community=multi_community,
            eval_seeds=eval_config.get('eval_seeds', None),
            verbose=verbose,
        )
        callback_list.append(eval_callback)

    callbacks = CallbackList(callback_list)

    # Train
    print(f"\n{'='*80}")
    if resume_from is not None:
        print("Resuming training...")
    else:
        print("Starting training...")
    print(f"{'='*80}\n")

    model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=callbacks,
        log_interval=logging_config.get('log_interval', 10),
        tb_log_name=f"ppo_{arm}",
        reset_num_timesteps=(resume_from is None),  # False if resuming, True otherwise
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

    parser.add_argument('arm', type=str, choices=['treatment', 'control'],
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
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity/username (overrides config and WANDB_ENTITY env var)')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name (overrides config, default: altar-transfer)')
    parser.add_argument('--multi-community', action='store_true',
                        help='Enable multi-community mode (Phase 5)')
    parser.add_argument('--permitted-color', type=str, default=None,
                        choices=['red', 'green', 'blue'],
                        help='Permitted color for single-community training (red/green/blue, overrides config)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from checkpoint (path to .zip file)')
    parser.add_argument('--resume-vec-normalize', type=str, default=None,
                        help='VecNormalize stats for resumed training (path to .pkl file, auto-detected if not provided)')
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
    if args.wandb_entity is not None:
        config['logging']['wandb_entity'] = args.wandb_entity
    if args.wandb_project is not None:
        config['logging']['wandb_project'] = args.wandb_project

    # Override permitted color (convert color name to index)
    if args.permitted_color is not None:
        color_to_index = {'red': 1, 'green': 2, 'blue': 3}
        config['env']['permitted_color_index'] = color_to_index[args.permitted_color]
        print(f"Overriding permitted_color_index to {config['env']['permitted_color_index']} ({args.permitted_color.upper()})")

    # Train
    train(
        arm=args.arm,
        config=config,
        output_dir=args.output_dir,
        verbose=args.verbose,
        multi_community=args.multi_community,
        resume_from=args.resume_from,
        resume_vec_normalize=args.resume_vec_normalize,
    )


if __name__ == '__main__':
    main()
