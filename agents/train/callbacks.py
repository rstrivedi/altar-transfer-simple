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

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
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
