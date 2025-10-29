# Added by RST: Training callbacks for Phase 4
"""Callbacks for W&B logging, checkpointing, and evaluation.

Callbacks:
- WandbLoggingCallback: Log head-wise stats, FiLM diagnostics to W&B
- CheckpointCallback: Save checkpoints periodically and best models
- EvalCallback: Run Phase 3 evaluation harness during training

Usage:
    from agents.train.callbacks import WandbLoggingCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.callbacks import CallbackList

    callbacks = CallbackList([
        WandbLoggingCallback(config=config, arm='treatment'),
        CheckpointCallback(save_freq=50000, checkpoint_dir='./checkpoints'),
        EvalCallback(eval_freq=100000, config=config, arm='treatment'),
    ])

    model.learn(total_timesteps=5000000, callback=callbacks)
"""

import os
from typing import Dict, Optional, Any
import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class WandbLoggingCallback(BaseCallback):
    """Callback for logging training metrics to Weights & Biases.

    Logs:
    - Standard PPO metrics (reward, episode length, value loss, policy loss)
    - Head-wise entropy (game vs sanction)
    - FiLM diagnostics (γ, β norms)
    - Action distribution (zap rate)

    Args:
        config: Configuration dict
        arm: 'treatment' or 'control'
        project: W&B project name (default 'altar-transfer')
        entity: W&B entity (optional)
        run_name: W&B run name (optional, auto-generated if None)
        tags: List of tags for the run
    """

    def __init__(
        self,
        config: Dict,
        arm: str,
        project: str = 'altar-transfer',
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.config = config
        self.arm = arm
        self.project = project
        self.entity = entity
        self.run_name = run_name or f'phase4_{arm}'
        self.tags = tags or [f'phase4', arm]
        self.wandb_run = None

    def _init_callback(self) -> None:
        """Initialize W&B run."""
        try:
            import wandb

            # Check for API key
            api_key = os.environ.get('WANDB_API_KEY')
            if not api_key:
                if self.verbose > 0:
                    print("WARNING: WANDB_API_KEY not set. W&B logging disabled.")
                self.wandb_run = None
                return

            # Build wandb config
            wandb_config = {
                'arm': self.arm,
                'phase': 4,
                **self.config,
            }

            # Initialize run
            self.wandb_run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=wandb_config,
                tags=self.tags,
                reinit=True,
            )

            if self.verbose > 0:
                print(f"W&B run initialized: {self.wandb_run.url}")

        except ImportError:
            if self.verbose > 0:
                print("WARNING: wandb not installed. W&B logging disabled.")
            self.wandb_run = None

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        if self.wandb_run is None:
            return True

        # Log standard metrics (already logged by SB3 to tensorboard)
        # W&B will auto-sync tensorboard logs

        # TODO: Add custom head-wise metrics, FiLM diagnostics
        # This requires accessing policy internals during training

        return True

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.wandb_run is not None:
            self.wandb_run.finish()


class CheckpointCallback(BaseCallback):
    """Callback for saving checkpoints during training.

    Saves:
    - Periodic checkpoints every save_freq steps
    - VecNormalize stats (if applicable)

    Args:
        save_freq: Save checkpoint every N steps
        checkpoint_dir: Directory to save checkpoints
        save_vec_normalize: Whether to save VecNormalize stats
        name_prefix: Prefix for checkpoint filenames
        verbose: Verbosity level
    """

    def __init__(
        self,
        save_freq: int,
        checkpoint_dir: str = './checkpoints',
        save_vec_normalize: bool = True,
        name_prefix: str = 'ppo',
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.save_vec_normalize = save_vec_normalize
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        """Initialize callback."""
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.verbose > 0:
            print(f"Checkpoints will be saved to {self.checkpoint_dir} every {self.save_freq} steps")

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        # Save checkpoint every save_freq steps
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'{self.name_prefix}_step_{self.num_timesteps}'
            )

            if self.verbose > 0:
                print(f"Saving checkpoint to {checkpoint_path}")

            self.model.save(checkpoint_path)

            # Save VecNormalize stats
            if self.save_vec_normalize and isinstance(self.model.env, VecNormalize):
                vec_normalize_path = os.path.join(
                    self.checkpoint_dir,
                    f'vec_normalize_step_{self.num_timesteps}.pkl'
                )
                self.model.env.save(vec_normalize_path)

                if self.verbose > 0:
                    print(f"Saved VecNormalize stats to {vec_normalize_path}")

        return True


class EvalCallback(BaseCallback):
    """Callback for running Phase 3 evaluation harness during training.

    Runs evaluation at regular intervals:
    1. Saves checkpoint
    2. Loads checkpoint with policy_loader
    3. Runs Phase 3 eval_harness (baseline + treatment + control)
    4. Logs metrics to W&B

    Args:
        eval_freq: Evaluate every N steps
        config: Environment configuration dict
        arm: 'treatment' or 'control' (which arm is being trained)
        n_eval_episodes: Number of episodes per arm for evaluation
        eval_seeds: List of seeds for evaluation (optional)
        checkpoint_dir: Directory to save eval checkpoints
        verbose: Verbosity level
    """

    def __init__(
        self,
        eval_freq: int,
        config: Dict,
        arm: str,
        n_eval_episodes: int = 20,
        eval_seeds: Optional[list] = None,
        checkpoint_dir: str = './checkpoints',
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.config = config
        self.arm = arm
        self.n_eval_episodes = n_eval_episodes
        self.eval_seeds = eval_seeds
        self.checkpoint_dir = checkpoint_dir

        # Generate eval seeds if not provided
        if self.eval_seeds is None:
            rng = np.random.RandomState(12345)
            self.eval_seeds = rng.randint(0, 1_000_000, size=n_eval_episodes).tolist()

    def _init_callback(self) -> None:
        """Initialize callback."""
        if self.verbose > 0:
            print(f"Evaluation will run every {self.eval_freq} steps with {self.n_eval_episodes} episodes")

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        # Run evaluation every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"Running evaluation at step {self.num_timesteps}...")
                print(f"{'='*80}\n")

            # Save checkpoint for evaluation
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'ppo_eval_step_{self.num_timesteps}.zip'
            )
            self.model.save(checkpoint_path)

            # TODO: Load checkpoint and run Phase 3 eval_harness
            # This requires:
            # 1. Policy loader function that loads SB3 checkpoint
            # 2. Calling run_evaluation_from_checkpoint from Phase 3
            # 3. Logging results to W&B

            if self.verbose > 0:
                print(f"Evaluation checkpoint saved to {checkpoint_path}")
                print("TODO: Integrate with Phase 3 eval_harness")
                print(f"{'='*80}\n")

        return True


# Helper function for creating callback list

def create_callbacks(
    config: Dict,
    arm: str,
    enable_wandb: bool = True,
    enable_checkpointing: bool = True,
    enable_eval: bool = False,  # Disabled by default (TODO: implement eval integration)
) -> list:
    """Create list of callbacks for training.

    Args:
        config: Configuration dict
        arm: 'treatment' or 'control'
        enable_wandb: Whether to enable W&B logging
        enable_checkpointing: Whether to enable checkpointing
        enable_eval: Whether to enable periodic evaluation (TODO)

    Returns:
        List of callback instances
    """
    callbacks = []

    # W&B logging
    if enable_wandb:
        wandb_callback = WandbLoggingCallback(
            config=config,
            arm=arm,
            project=config.get('logging', {}).get('wandb_project', 'altar-transfer'),
            entity=config.get('logging', {}).get('wandb_entity'),
            run_name=config.get('logging', {}).get('wandb_run_name'),
        )
        callbacks.append(wandb_callback)

    # Checkpointing
    if enable_checkpointing:
        checkpoint_callback = CheckpointCallback(
            save_freq=config.get('checkpointing', {}).get('save_freq', 50_000),
            checkpoint_dir=config.get('checkpointing', {}).get('checkpoint_dir', './checkpoints'),
            save_vec_normalize=config.get('checkpointing', {}).get('save_vec_normalize', True),
            name_prefix=f'ppo_{arm}',
        )
        callbacks.append(checkpoint_callback)

    # Evaluation (TODO: implement Phase 3 integration)
    if enable_eval:
        eval_callback = EvalCallback(
            eval_freq=config.get('evaluation', {}).get('eval_freq', 100_000),
            config=config.get('env', {}),
            arm=arm,
            n_eval_episodes=config.get('evaluation', {}).get('n_eval_episodes', 20),
            eval_seeds=config.get('evaluation', {}).get('eval_seeds'),
        )
        callbacks.append(eval_callback)

    return callbacks
