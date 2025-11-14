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
from collections import defaultdict
import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

# Rich console for pretty printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Edited by RST: Import distributional evaluation for Phase 5
from agents.metrics.eval_harness import run_evaluation, run_distributional_evaluation
from agents.metrics.schema import DistributionalRunMetrics


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
        multi_community: Whether this is Phase 5 multi-community training
        project: W&B project name (default 'altar-transfer')
        entity: W&B entity (optional)
        run_name: W&B run name (optional, auto-generated if None)
        tags: List of tags for the run
        log_interval: Log FiLM diagnostics every N timesteps (default 2560 = 10 rollouts)
        verbose: Verbosity level
    """

    def __init__(
        self,
        config: Dict,
        arm: str,
        multi_community: bool = False,  # Edited by RST: Phase 5 flag
        project: str = 'altar-transfer',
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        log_interval: int = 2560,  # Log FiLM diagnostics every N steps (default: every 10 rollouts with n_steps=256)
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.config = config
        self.arm = arm
        self.multi_community = multi_community  # Edited by RST: Track Phase 5 mode
        self.project = project
        self.entity = entity
        self.log_interval = log_interval
        # Edited by RST: Update run name and tags for Phase 5
        phase = 5 if multi_community else 4
        self.run_name = run_name or f'phase{phase}_{arm}{"_multi" if multi_community else ""}'
        self.tags = tags or [f'phase{phase}', arm] + (['multi-community'] if multi_community else [])
        self.wandb_run = None

        # Console output for nice metrics display
        self.console = Console() if HAS_RICH else None
        self.episode_data = defaultdict(list)  # Track episodes per community
        self.last_report_timestep = 0
        self.report_iteration = 0

        # Added by RST: tqdm progress bar for training
        self.pbar = None

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
            # Edited by RST: Update phase based on multi_community mode
            phase = 5 if self.multi_community else 4
            wandb_config = {
                'arm': self.arm,
                'phase': phase,
                'multi_community': self.multi_community,
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

    def _on_training_start(self) -> None:
        """Initialize tqdm progress bar at training start."""
        # Added by RST: Initialize tqdm progress bar with colors
        if HAS_TQDM:
            # Account for resumed training by subtracting already completed timesteps
            total_timesteps = self.locals.get("total_timesteps", 0)
            already_done = self.model.num_timesteps
            remaining_timesteps = total_timesteps - already_done

            self.pbar = tqdm(
                total=remaining_timesteps,
                desc=f"Training {self.arm}",
                unit="steps",
                colour='green',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        # Added by RST: Update tqdm progress bar
        # _on_step is called once per env.step(), which executes num_envs steps in parallel
        if self.pbar is not None:
            self.pbar.update(self.training_env.num_envs)

        # Track episode completions per community (for multi-community mode)
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info and isinstance(info['episode'], dict):
                # Get community tag if available (Phase 5)
                community_tag = info.get('community_tag', 'single')
                self.episode_data[community_tag].append(info['episode'].copy())

        # Print console report every ~2048 steps (matches PPO reporting interval)
        n_envs = self.training_env.num_envs if hasattr(self, 'training_env') else 8
        report_interval = 2048 * n_envs
        if self.num_timesteps >= self.last_report_timestep + report_interval:
            self._print_console_report()
            self.last_report_timestep = self.num_timesteps
            self.report_iteration += 1
            # Clear episode data after reporting
            self.episode_data.clear()

        if self.wandb_run is None:
            return True

        # Log standard metrics (already logged by SB3 to tensorboard)
        # W&B will auto-sync tensorboard logs

        # Log custom FiLM diagnostics and head-wise metrics every N steps
        if self.num_timesteps % self.log_interval == 0:
            self._log_film_diagnostics()
            self._log_head_wise_stats()
            self._log_action_distribution()

        return True

    def _log_film_diagnostics(self) -> None:
        """Log FiLM parameter norms and gradients.

        This helps diagnose whether the policy is learning to use the institutional signal.
        Expected behavior:
        - Treatment: gamma/beta should deviate from identity (1, 0) as training progresses
        - Control: gamma/beta should stay near identity (signal is zeros, so no learning)
        """
        import torch

        try:
            policy = self.model.policy

            # Check if policy has FiLM modules (FiLMTwoHeadPolicy)
            if not hasattr(policy, 'global_film'):
                return

            # Global FiLM parameter norms
            global_gamma_weights = policy.global_film.gamma_layer.weight
            global_gamma_bias = policy.global_film.gamma_layer.bias
            global_beta_weights = policy.global_film.beta_layer.weight
            global_beta_bias = policy.global_film.beta_layer.bias

            self.wandb_run.log({
                'film/global_gamma_weight_norm': torch.norm(global_gamma_weights).item(),
                'film/global_gamma_bias_norm': torch.norm(global_gamma_bias).item(),
                'film/global_beta_weight_norm': torch.norm(global_beta_weights).item(),
                'film/global_beta_bias_norm': torch.norm(global_beta_bias).item(),
                # Log deviation from identity initialization
                'film/global_gamma_bias_deviation': torch.norm(global_gamma_bias - 1.0).item(),
                'film/global_beta_bias_deviation': torch.norm(global_beta_bias).item(),
            }, step=self.num_timesteps)

            # Local FiLM (sanction head) parameter norms
            if hasattr(policy, 'local_film'):
                local_gamma_weights = policy.local_film.gamma_layer.weight
                local_gamma_bias = policy.local_film.gamma_layer.bias
                local_beta_weights = policy.local_film.beta_layer.weight
                local_beta_bias = policy.local_film.beta_layer.bias

                self.wandb_run.log({
                    'film/local_gamma_weight_norm': torch.norm(local_gamma_weights).item(),
                    'film/local_gamma_bias_norm': torch.norm(local_gamma_bias).item(),
                    'film/local_beta_weight_norm': torch.norm(local_beta_weights).item(),
                    'film/local_beta_bias_norm': torch.norm(local_beta_bias).item(),
                    'film/local_gamma_bias_deviation': torch.norm(local_gamma_bias - 1.0).item(),
                    'film/local_beta_bias_deviation': torch.norm(local_beta_bias).item(),
                }, step=self.num_timesteps)

            # Gradient norms (if gradients exist)
            if global_gamma_weights.grad is not None:
                self.wandb_run.log({
                    'film/global_gamma_grad_norm': torch.norm(global_gamma_weights.grad).item(),
                    'film/global_beta_grad_norm': torch.norm(global_beta_weights.grad).item(),
                }, step=self.num_timesteps)

            if hasattr(policy, 'local_film') and policy.local_film.gamma_layer.weight.grad is not None:
                self.wandb_run.log({
                    'film/local_gamma_grad_norm': torch.norm(policy.local_film.gamma_layer.weight.grad).item(),
                    'film/local_beta_grad_norm': torch.norm(policy.local_film.beta_layer.weight.grad).item(),
                }, step=self.num_timesteps)

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log FiLM diagnostics: {e}")

    def _log_head_wise_stats(self) -> None:
        """Log head-wise entropy and action head statistics.

        This helps diagnose whether game and sanction heads are learning differently.
        """
        import torch

        try:
            policy = self.model.policy

            # Get recent rollout buffer data
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.size() > 0:
                # Sample from buffer
                buffer = self.model.rollout_buffer

                # Get actions from buffer (last N actions)
                if hasattr(buffer, 'actions'):
                    actions = buffer.actions[-100:]  # Last 100 actions

                    # Count zap actions (action index 7)
                    zap_count = (actions == 7).sum().item()
                    total_count = len(actions)
                    zap_rate = zap_count / total_count if total_count > 0 else 0.0

                    self.wandb_run.log({
                        'policy/zap_rate': zap_rate,
                        'policy/zap_count': zap_count,
                    }, step=self.num_timesteps)

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log head-wise stats: {e}")

    def _log_action_distribution(self) -> None:
        """Log action distribution statistics.

        This helps diagnose exploration vs exploitation and action diversity.
        """
        import torch

        try:
            # Get recent actions from rollout buffer
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.size() > 0:
                buffer = self.model.rollout_buffer

                if hasattr(buffer, 'actions'):
                    actions = buffer.actions[-1000:]  # Last 1000 actions

                    # Convert to torch tensor if numpy array
                    if isinstance(actions, np.ndarray):
                        actions = torch.from_numpy(actions)

                    # Compute action histogram
                    action_counts = torch.bincount(actions.flatten().long(), minlength=11)
                    action_probs = action_counts.float() / action_counts.sum()

                    # Log per-action probabilities
                    action_names = ['NOOP', 'FORWARD', 'BACKWARD', 'STEP_LEFT', 'STEP_RIGHT',
                                    'TURN_LEFT', 'TURN_RIGHT', 'ZAP', 'PLANT_RED', 'PLANT_GREEN', 'PLANT_BLUE']

                    for i, name in enumerate(action_names):
                        self.wandb_run.log({
                            f'actions/{name}_prob': action_probs[i].item(),
                        }, step=self.num_timesteps)

                    # Log action entropy (diversity)
                    action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()
                    self.wandb_run.log({
                        'actions/entropy': action_entropy,
                    }, step=self.num_timesteps)

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log action distribution: {e}")

    def _print_console_report(self):
        """Print nice formatted metrics table to console and log to W&B."""
        if not HAS_RICH or self.console is None:
            return

        total_timesteps = self.config['training']['total_timesteps']

        # Create table
        table = Table(
            title=f"Training Progress - Iteration {self.report_iteration} (Steps: {self.num_timesteps:,}/{total_timesteps:,})"
        )

        # Add columns
        table.add_column("Community", style="cyan", no_wrap=True)
        table.add_column("Episodes", style="white")
        table.add_column("Mean Reward ± Std", style="green")
        table.add_column("Collective R", style="yellow")
        table.add_column("Sanctions", style="red")
        table.add_column("Berry Distribution", style="magenta")

        # Process each community
        for community_tag in sorted(self.episode_data.keys()):
            episodes = self.episode_data[community_tag]

            if episodes:
                # Compute metrics
                # Mean reward = ego's natural reward (r - alpha, excluding sanctioning bonuses)
                # r = R_env + alpha - beta - c
                # r - alpha = R_env - beta - c (natural reward)
                episode_rewards = [ep.get('r', 0) - ep.get('alpha', 0) for ep in episodes]
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)

                # Collective reward = sum of all agents' total rewards
                collective_rewards = [ep.get('collective_reward', 0) for ep in episodes]
                mean_collective = np.mean(collective_rewards)

                # Sanction metrics
                times_sanctioned = np.mean([ep.get('times_sanctioned', 0) for ep in episodes])
                times_sanctioned_others = np.mean([ep.get('times_sanctioned_others', 0) for ep in episodes])
                sanction_str = f"Recv:{times_sanctioned:.1f} Give:{times_sanctioned_others:.1f}"

                # Berry metrics
                berries_red = np.mean([ep.get('berries_planted_red', 0) for ep in episodes])
                berries_green = np.mean([ep.get('berries_planted_green', 0) for ep in episodes])
                berries_blue = np.mean([ep.get('berries_planted_blue', 0) for ep in episodes])

                total_berries = berries_red + berries_green + berries_blue
                if total_berries > 0:
                    red_pct = berries_red / total_berries * 100
                    green_pct = berries_green / total_berries * 100
                    blue_pct = berries_blue / total_berries * 100
                    berry_dist = f"R:{red_pct:.0f}% G:{green_pct:.0f}% B:{blue_pct:.0f}%"
                    # Added by RST: Compute monoculture fraction (max berry type / total)
                    monoculture_fraction = max(berries_red, berries_green, berries_blue) / total_berries
                else:
                    berry_dist = "No berries"
                    red_pct = 0.0
                    green_pct = 0.0
                    blue_pct = 0.0
                    monoculture_fraction = 0.0

                # Log to W&B per community
                if self.wandb_run is not None:
                    prefix = f"episode/{community_tag}"
                    self.wandb_run.log({
                        f"{prefix}/mean_reward": mean_reward,
                        f"{prefix}/std_reward": std_reward,
                        f"{prefix}/mean_collective_reward": mean_collective,
                        f"{prefix}/mean_r_eval": np.mean([ep.get('r_eval', 0) for ep in episodes]),
                        f"{prefix}/mean_berries_planted_red": berries_red,
                        f"{prefix}/mean_berries_planted_green": berries_green,
                        f"{prefix}/mean_berries_planted_blue": berries_blue,
                        # Added by RST: Berry distribution percentages
                        f"{prefix}/berry_pct_red": red_pct,
                        f"{prefix}/berry_pct_green": green_pct,
                        f"{prefix}/berry_pct_blue": blue_pct,
                        f"{prefix}/monoculture_fraction": monoculture_fraction,
                        f"{prefix}/mean_berries_consumed_red": np.mean([ep.get('berries_consumed_red', 0) for ep in episodes]),
                        f"{prefix}/mean_berries_consumed_green": np.mean([ep.get('berries_consumed_green', 0) for ep in episodes]),
                        f"{prefix}/mean_berries_consumed_blue": np.mean([ep.get('berries_consumed_blue', 0) for ep in episodes]),
                        f"{prefix}/mean_times_sanctioned": times_sanctioned,
                        f"{prefix}/mean_times_sanctioned_others": times_sanctioned_others,
                        f"{prefix}/episode_count": len(episodes),
                    }, step=self.num_timesteps)

                # Add row to console table
                table.add_row(
                    community_tag,
                    str(len(episodes)),
                    f"{mean_reward:.2f} ± {std_reward:.2f}",
                    f"{mean_collective:.1f}",
                    sanction_str,
                    berry_dist,
                )

        # Print table
        self.console.print(table)

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Added by RST: Close tqdm progress bar
        if self.pbar is not None:
            self.pbar.close()

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


class BestModelCallback(BaseCallback):
    """Callback for saving best model based on evaluation performance.

    Tracks mean episode reward during training and saves the best-performing
    model checkpoint. This ensures you always have the peak-performance model,
    even if training degrades later.

    Args:
        checkpoint_dir: Directory to save best model
        eval_freq: Evaluate and check for best model every N steps (default: 100k)
        n_eval_episodes: Number of episodes for evaluation (default: 10)
        deterministic: Use deterministic policy for evaluation (default: True)
        verbose: Verbosity level
    """

    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        eval_freq: int = 100_000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.best_timestep = 0

    def _init_callback(self) -> None:
        """Initialize callback."""
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.verbose > 0:
            print(f"Best model will be saved to {self.checkpoint_dir}/best_model.zip")
            print(f"Evaluation every {self.eval_freq} steps with {self.n_eval_episodes} episodes")

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        # Evaluate every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            # Collect episode rewards from recent rollouts
            # Use info dicts from VecEnv which contain episode stats
            episode_rewards = []
            episode_lengths = []

            # Get recent episodes from the rollout buffer via callback locals
            # The 'infos' in locals contain episode data when episodes complete
            # We'll use a simpler approach: collect from training env's episode buffer

            # Alternative: Use training env's episode statistics
            # SB3 tracks episode rewards in the VecEnv wrapper
            if hasattr(self.training_env, 'get_attr'):
                # Try to get episode rewards from env attributes
                try:
                    # Get recent episode data from the buffer
                    # This is a heuristic - collect last N completed episodes
                    # from the vectorized environments

                    # Simpler approach: evaluate on training env directly
                    # Run N evaluation episodes
                    for _ in range(self.n_eval_episodes):
                        obs = self.training_env.reset()
                        done = False
                        episode_reward = 0.0
                        episode_length = 0

                        while not np.all(done):
                            # Get action from current policy
                            action, _ = self.model.predict(obs, deterministic=self.deterministic)
                            obs, reward, done, info = self.training_env.step(action)

                            # Accumulate reward (for vectorized env, sum across envs)
                            episode_reward += np.mean(reward)
                            episode_length += 1

                            # Check if any episodes completed
                            for idx, d in enumerate(done):
                                if d and 'episode' in info[idx]:
                                    # Use actual episode reward from info
                                    ep_info = info[idx]['episode']
                                    if 'r' in ep_info:
                                        episode_rewards.append(ep_info['r'])
                                    if 'l' in ep_info:
                                        episode_lengths.append(ep_info['l'])

                            # Break if we have enough episodes
                            if len(episode_rewards) >= self.n_eval_episodes:
                                break

                        if len(episode_rewards) >= self.n_eval_episodes:
                            break

                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not evaluate model: {e}")
                    return True

            # Compute mean reward
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)

                if self.verbose > 0:
                    print(f"\n{'='*60}")
                    print(f"Eval at step {self.num_timesteps}:")
                    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
                    print(f"  Best so far: {self.best_mean_reward:.2f} (at step {self.best_timestep})")

                # Check if this is the best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.best_timestep = self.num_timesteps

                    if self.verbose > 0:
                        print(f"  🏆 New best model! Saving...")
                    print(f"{'='*60}\n")

                    # Save best model
                    best_model_path = os.path.join(self.checkpoint_dir, 'best_model')
                    self.model.save(best_model_path)

                    # Save VecNormalize stats
                    if isinstance(self.model.env, VecNormalize):
                        vec_normalize_path = os.path.join(self.checkpoint_dir, 'best_vec_normalize.pkl')
                        self.model.env.save(vec_normalize_path)

                    # Save metadata about best model
                    metadata_path = os.path.join(self.checkpoint_dir, 'best_model_info.txt')
                    with open(metadata_path, 'w') as f:
                        f.write(f"Best model checkpoint\n")
                        f.write(f"Timestep: {self.best_timestep}\n")
                        f.write(f"Mean reward: {self.best_mean_reward:.4f}\n")
                        f.write(f"Std reward: {std_reward:.4f}\n")
                        f.write(f"Num eval episodes: {len(episode_rewards)}\n")

                elif self.verbose > 0:
                    print(f"{'='*60}\n")

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
        multi_community: bool = False,  # Edited by RST: Phase 5 flag
        n_eval_episodes: int = 20,
        eval_seeds: Optional[list] = None,
        checkpoint_dir: str = './checkpoints',
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.config = config
        self.arm = arm
        self.multi_community = multi_community  # Edited by RST: Track Phase 5 mode
        self.n_eval_episodes = n_eval_episodes
        self.eval_seeds = eval_seeds
        self.checkpoint_dir = checkpoint_dir

        # Generate eval seeds if not provided
        if self.eval_seeds is None:
            rng = np.random.RandomState(12345)
            # Edited by RST: Phase 5 needs 3x episodes (one per community)
            n_seeds = n_eval_episodes * 3 if multi_community else n_eval_episodes
            self.eval_seeds = rng.randint(0, 1_000_000, size=n_seeds).tolist()

        # Added by RST: Track best models according to different criteria
        self.best_reward = -np.inf
        self.best_reward_step = 0
        self.best_value_gap = np.inf  # Lower is better
        self.best_value_gap_step = 0
        self.best_sanction_regret = np.inf  # Lower is better
        self.best_sanction_regret_step = 0
        self.best_combined = np.inf  # Lower is better (value_gap + sanction_regret)
        self.best_combined_step = 0

    def _init_callback(self) -> None:
        """Initialize callback."""
        if self.verbose > 0:
            # Edited by RST: Update message for Phase 5
            mode = "Phase 5 (distributional)" if self.multi_community else "Phase 4"
            episodes_desc = f"{self.n_eval_episodes} episodes per community" if self.multi_community else f"{self.n_eval_episodes} episodes"
            print(f"{mode} evaluation will run every {self.eval_freq} steps with {episodes_desc}")

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        # Run evaluation every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n{'='*80}")
                mode = "Phase 5 Distributional" if self.multi_community else "Phase 4"
                print(f"Running {mode} evaluation at step {self.num_timesteps}...")
                print(f"{'='*80}\n")

            # Save checkpoint for evaluation
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'ppo_eval_step_{self.num_timesteps}.zip'
            )
            self.model.save(checkpoint_path)

            # Save VecNormalize stats if present
            vec_normalize_path = None
            if isinstance(self.model.env, VecNormalize):
                vec_normalize_path = os.path.join(
                    self.checkpoint_dir,
                    f'vec_normalize_eval_step_{self.num_timesteps}.pkl'
                )
                self.model.env.save(vec_normalize_path)

            # Edited by RST: Implement Phase 4/5 evaluation integration
            try:
                # Create policy wrapper for eval harness
                def policy_fn(obs):
                    """Policy wrapper for eval harness."""
                    # Get action from policy (deterministic for evaluation)
                    action, _ = self.model.predict(obs, deterministic=True)
                    return action

                # Run evaluation based on mode
                if self.multi_community:
                    # Phase 5: Distributional evaluation across RED/GREEN/BLUE
                    results = run_distributional_evaluation(
                        ego_policy=policy_fn,
                        config=self.config,
                        num_episodes_per_community=self.n_eval_episodes,
                        seed=self.eval_seeds[0] if isinstance(self.eval_seeds, list) else 12345,
                    )

                    # Extract metrics
                    baseline_dist = results['baseline']
                    treatment_dist = results.get('treatment')
                    control_dist = results.get('control')

                    # Get the trained arm metrics
                    ego_dist = treatment_dist if self.arm == 'treatment' else control_dist

                    # Log to W&B
                    self._log_distributional_metrics(ego_dist, baseline_dist)

                else:
                    # Phase 4: Single-community evaluation
                    results = run_evaluation(
                        ego_policy=policy_fn,
                        config=self.config,
                        num_episodes=self.n_eval_episodes,
                        seeds=self.eval_seeds,
                    )

                    # Log to W&B
                    self._log_single_community_metrics(results)

            except Exception as e:
                if self.verbose > 0:
                    print(f"Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()

            if self.verbose > 0:
                print(f"{'='*80}\n")

        return True

    def _check_and_save_best_models(
        self,
        reward: float,
        value_gap: float,
        sanction_regret: float,
        combined_score: float,
    ):
        """Check if current model is best according to any criterion and save.

        Added by RST: Multi-criteria best model tracking.

        Args:
            reward: Mean evaluation reward (higher is better)
            value_gap: Mean value gap (lower is better)
            sanction_regret: Mean sanction regret (lower is better)
            combined_score: Combined score = value_gap + sanction_regret (lower is better)
        """
        # Track which models were updated
        updated = []

        # 1. Check best reward (highest)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_reward_step = self.num_timesteps
            self._save_best_model(
                'reward',
                reward,
                value_gap,
                sanction_regret,
                f"Highest evaluation reward: {reward:.4f}"
            )
            updated.append('reward')

        # 2. Check best value gap (lowest)
        if value_gap < self.best_value_gap:
            self.best_value_gap = value_gap
            self.best_value_gap_step = self.num_timesteps
            self._save_best_model(
                'value_gap',
                reward,
                value_gap,
                sanction_regret,
                f"Lowest value gap: {value_gap:.4f}"
            )
            updated.append('value_gap')

        # 3. Check best sanction regret (lowest)
        if sanction_regret < self.best_sanction_regret:
            self.best_sanction_regret = sanction_regret
            self.best_sanction_regret_step = self.num_timesteps
            self._save_best_model(
                'sanction_regret',
                reward,
                value_gap,
                sanction_regret,
                f"Lowest sanction regret: {sanction_regret:.4f}"
            )
            updated.append('sanction_regret')

        # 4. Check best combined (lowest)
        if combined_score < self.best_combined:
            self.best_combined = combined_score
            self.best_combined_step = self.num_timesteps
            self._save_best_model(
                'combined',
                reward,
                value_gap,
                sanction_regret,
                f"Lowest combined score: {combined_score:.4f} (ΔV + SR)"
            )
            updated.append('combined')

        # Print summary of updates
        if updated and self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🏆 New best model(s) at step {self.num_timesteps}:")
            for criterion in updated:
                print(f"  • {criterion}")
            print(f"{'='*60}\n")

    def _save_best_model(
        self,
        criterion: str,
        reward: float,
        value_gap: float,
        sanction_regret: float,
        description: str,
    ):
        """Save best model checkpoint for a specific criterion.

        Args:
            criterion: One of 'reward', 'value_gap', 'sanction_regret', 'combined'
            reward: Current evaluation reward
            value_gap: Current value gap
            sanction_regret: Current sanction regret
            description: Human-readable description of why this is best
        """
        # Save model
        model_path = os.path.join(self.checkpoint_dir, f'best_model_{criterion}')
        self.model.save(model_path)

        # Save VecNormalize stats if present
        if isinstance(self.model.env, VecNormalize):
            vec_normalize_path = os.path.join(self.checkpoint_dir, f'best_vec_normalize_{criterion}.pkl')
            self.model.env.save(vec_normalize_path)

        # Save metadata
        metadata_path = os.path.join(self.checkpoint_dir, f'best_model_{criterion}_info.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Best model checkpoint: {criterion}\n")
            f.write(f"Timestep: {self.num_timesteps}\n")
            f.write(f"Description: {description}\n")
            f.write(f"\nMetrics at this checkpoint:\n")
            f.write(f"  Evaluation reward: {reward:.4f}\n")
            f.write(f"  Value gap: {value_gap:.4f}\n")
            f.write(f"  Sanction regret: {sanction_regret:.4f}\n")
            f.write(f"  Combined (ΔV + SR): {value_gap + sanction_regret:.4f}\n")

    def _log_distributional_metrics(
        self,
        ego_dist: DistributionalRunMetrics,
        baseline_dist: DistributionalRunMetrics,
    ):
        """Log Phase 5 distributional metrics to W&B.

        Added by RST: Phase 5 per-color and distributional logging.
        """
        try:
            import wandb

            if wandb.run is None:
                return

            # === Per-community metrics ===
            for color in ['red', 'green', 'blue']:
                ego_metrics = getattr(ego_dist, f'{color}_metrics')
                baseline_metrics = getattr(baseline_dist, f'{color}_metrics')

                if ego_metrics is None or baseline_metrics is None:
                    continue

                color_upper = color.upper()
                wandb.log({
                    f'eval/{color}/value_gap_mean': ego_metrics.value_gap_mean,
                    f'eval/{color}/value_gap_median': ego_metrics.value_gap_median,
                    f'eval/{color}/value_gap_std': ego_metrics.value_gap_std,
                    f'eval/{color}/sanction_regret_mean': ego_metrics.sanction_regret_mean,
                    f'eval/{color}/sanction_regret_median': ego_metrics.sanction_regret_median,
                    f'eval/{color}/correct_sanction_rate': ego_metrics.correct_sanction_rate,
                    f'eval/{color}/violation_rate': ego_metrics.violation_rate,
                    f'eval/{color}/immunity_rate': ego_metrics.immunity_rate,
                    f'eval/{color}/baseline_value_gap': baseline_metrics.value_gap_mean,
                    f'eval/{color}/baseline_sanction_regret': baseline_metrics.sanction_regret_mean,
                }, step=self.num_timesteps)

            # === Distributional summary metrics ===
            wandb.log({
                'eval/dist/avg_value_gap': ego_dist.avg_value_gap,
                'eval/dist/avg_sanction_regret': ego_dist.avg_sanction_regret,
                'eval/dist/worst_value_gap': ego_dist.worst_value_gap,
                'eval/dist/worst_community': ego_dist.worst_community,
                'eval/dist/best_value_gap': ego_dist.best_value_gap,
                'eval/dist/best_community': ego_dist.best_community,
                'eval/dist/balance_check': ego_dist.balance_check_ratio,
                'eval/dist/baseline_avg_value_gap': baseline_dist.avg_value_gap,
                'eval/dist/baseline_worst_value_gap': baseline_dist.worst_value_gap,
            }, step=self.num_timesteps)

            if self.verbose > 0:
                print(f"\n=== Phase 5 Distributional Evaluation Results ===")
                print(f"Average ΔV: {ego_dist.avg_value_gap:.3f}")
                print(f"Worst ΔV: {ego_dist.worst_value_gap:.3f} ({ego_dist.worst_community})")
                print(f"Best ΔV: {ego_dist.best_value_gap:.3f} ({ego_dist.best_community})")
                print(f"Balance check: {ego_dist.balance_check_ratio:.3f}")

        except ImportError:
            pass  # W&B not available

        # Added by RST: Track and save best models based on different criteria
        # Use distributional averages for Phase 5
        avg_reward = ego_dist.avg_r_eval
        avg_value_gap = ego_dist.avg_value_gap
        avg_sanction_regret = ego_dist.avg_sanction_regret
        combined_score = avg_value_gap + avg_sanction_regret

        self._check_and_save_best_models(avg_reward, avg_value_gap, avg_sanction_regret, combined_score)

    def _log_single_community_metrics(self, results: Dict):
        """Log Phase 4 single-community metrics to W&B.

        Added by RST: Phase 4 single-community logging.
        """
        try:
            import wandb

            if wandb.run is None:
                return

            # Extract metrics from results dict
            baseline_metrics = results.get('baseline')
            ego_metrics = results.get('treatment' if self.arm == 'treatment' else 'control')

            if ego_metrics is None:
                return

            # Log main metrics
            wandb.log({
                'eval/value_gap_mean': ego_metrics.value_gap_mean,
                'eval/value_gap_median': ego_metrics.value_gap_median,
                'eval/sanction_regret_mean': ego_metrics.sanction_regret_mean,
                'eval/correct_sanction_rate': ego_metrics.correct_sanction_rate,
                'eval/violation_rate': ego_metrics.violation_rate,
            }, step=self.num_timesteps)

            if baseline_metrics is not None:
                wandb.log({
                    'eval/baseline_value_gap': baseline_metrics.value_gap_mean,
                    'eval/baseline_sanction_regret': baseline_metrics.sanction_regret_mean,
                }, step=self.num_timesteps)

            if self.verbose > 0:
                print(f"\n=== Phase 4 Evaluation Results ===")
                print(f"Value Gap: {ego_metrics.value_gap_mean:.3f}")
                print(f"Sanction Regret: {ego_metrics.sanction_regret_mean:.3f}")
                print(f"Correct Sanction Rate: {ego_metrics.correct_sanction_rate:.2%}")

        except ImportError:
            pass  # W&B not available

        # Added by RST: Track and save best models based on different criteria
        # Use single-community metrics for Phase 4
        if ego_metrics is not None:
            reward = ego_metrics.r_eval_mean
            value_gap = ego_metrics.value_gap_mean
            sanction_regret = ego_metrics.sanction_regret_mean
            combined_score = value_gap + sanction_regret

            self._check_and_save_best_models(reward, value_gap, sanction_regret, combined_score)


# Helper function for creating callback list

def create_callbacks(
    config: Dict,
    arm: str,
    multi_community: bool = False,  # Edited by RST: Phase 5 flag
    enable_wandb: bool = True,
    enable_checkpointing: bool = True,
    enable_eval: bool = False,
) -> list:
    """Create list of callbacks for training (Phase 4 or Phase 5).

    Args:
        config: Configuration dict
        arm: 'treatment' or 'control'
        multi_community: Whether to use Phase 5 multi-community mode
        enable_wandb: Whether to enable W&B logging
        enable_checkpointing: Whether to enable checkpointing
        enable_eval: Whether to enable periodic evaluation

    Returns:
        List of callback instances
    """
    callbacks = []

    # W&B logging
    if enable_wandb:
        wandb_callback = WandbLoggingCallback(
            config=config,
            arm=arm,
            multi_community=multi_community,  # Edited by RST: Pass Phase 5 flag
            project=config.get('logging', {}).get('wandb_project', 'altar-transfer'),
            entity=config.get('logging', {}).get('wandb_entity'),
            run_name=config.get('logging', {}).get('wandb_run_name'),
            log_interval=config.get('logging', {}).get('log_interval', 2560),  # FiLM diagnostics logging frequency
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

    # Evaluation
    if enable_eval:
        eval_callback = EvalCallback(
            eval_freq=config.get('evaluation', {}).get('eval_freq', 100_000),
            config=config.get('env', {}),
            arm=arm,
            multi_community=multi_community,  # Edited by RST: Pass Phase 5 flag
            n_eval_episodes=config.get('evaluation', {}).get('n_eval_episodes', 20),
            eval_seeds=config.get('evaluation', {}).get('eval_seeds'),
        )
        callbacks.append(eval_callback)

    return callbacks
