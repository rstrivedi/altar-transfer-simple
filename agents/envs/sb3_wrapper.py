# Added by RST: Gymnasium wrapper for SB3 PPO training (Phase 4 & 5)
"""Gymnasium wrapper for single-ego RL training with scripted residents.

This wrapper:
- Exposes agent 0 (ego) as a Gymnasium environment for SB3 PPO training
- Steps agents 1-15 (residents) automatically via ResidentController
- Returns observations: RGB, READY_TO_SHOOT, TIMESTEP, [PERMITTED_COLOR in treatment]
- Returns rewards: r_train = r_env + alpha - beta - c (alpha for training bonus)
- Integrates with MetricsRecorder for telemetry capture
- Supports treatment (with PERMITTED_COLOR) and control (without) arms
- Supports multi-community mode (Phase 5) for distributional competence training

Usage:
    # Phase 4: Single-community training
    env = AllelopathicHarvestGymEnv(
        arm='treatment',
        config={'permitted_color_index': 1, ...},  # Fixed community
        seed=42
    )

    # Phase 5: Multi-community training (independent sampling)
    env = AllelopathicHarvestGymEnv(
        arm='treatment',
        config={'permitted_color_index': 1, ...},  # Ignored, sampled randomly
        seed=42,
        multi_community_mode=True  # Samples RED/GREEN/BLUE at each reset
    )

    # Vectorized multi-community
    vec_env = make_vec_env_multi_community(
        arm='treatment', num_envs=32, config=config, seed=42
    )
"""

from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from meltingpot.utils.substrates import substrate
from meltingpot.configs.substrates import allelopathic_harvest

from agents.envs.normative_observation_filter import NormativeObservationFilter
from agents.envs.resident_wrapper import ResidentWrapper
from agents.residents.info_extractor import ResidentInfoExtractor
from agents.residents.scripted_residents import ResidentController
from agents.metrics.recorder import MetricsRecorder


class AllelopathicHarvestGymEnv(gym.Env):
    """Gymnasium environment for single-ego RL training in Allelopathic Harvest.

    Agent 0 is the learning ego; agents 1-15 are scripted residents.
    Observations and action space follow Gymnasium conventions for SB3 compatibility.
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(
        self,
        arm: str,
        config: Optional[Dict] = None,
        seed: Optional[int] = None,
        enable_telemetry: bool = True,
        multi_community_mode: bool = False,
        include_timestep: bool = False,
    ):
        """Initialize the Gymnasium environment.

        Args:
            arm: 'treatment' or 'control' (controls PERMITTED_COLOR observation exposure)
            config: Configuration dict with keys:
                - permitted_color_index: int (1=RED, 2=GREEN, 3=BLUE)
                  (ignored if multi_community_mode=True, sampled randomly instead)
                - startup_grey_grace: int (default 25)
                - episode_timesteps: int (default 2000)
                - altar_coords: Tuple[int, int] (treatment only, default (5, 15))
                - alpha: float (train-time bonus, default 0.5)
                - beta: float (mis-zap penalty, default 0.5)
                - c: float (zap cost, default 0.2)
                - immunity_cooldown: int (default 200)
            seed: Random seed for environment and residents
            enable_telemetry: Whether to use MetricsRecorder for tracking (default True)
            multi_community_mode: If True, randomly sample community (RED/GREEN/BLUE) at each reset
                                  for distributional competence training (Phase 5)
            include_timestep: If True, include normalized timestep (t/T) in observations (default False)
                             Note: Timestep can create temporal confounds - agents may learn grace period
                             timing rather than normative compliance. Disable for cleaner causal pathway.
        """
        super().__init__()

        if arm not in ['treatment', 'control']:
            raise ValueError(f"arm must be 'treatment' or 'control', got {arm}")

        self.arm = arm
        self.enable_treatment = (arm == 'treatment')
        self.enable_telemetry = enable_telemetry
        self.include_timestep = include_timestep

        # Setup config
        if config is None:
            config = self._get_default_config()
        self.config = config

        # Environment parameters
        self.num_players = 16
        self.ego_index = 0
        self.resident_indices = list(range(1, self.num_players))
        self.episode_len = config.get('episode_timesteps', 2000)

        # Build substrate config
        self.env_config = allelopathic_harvest.get_config()
        self.env_config.normative_gate = True
        self.env_config.permitted_color_index = config['permitted_color_index']
        self.env_config.startup_grey_grace = config.get('startup_grey_grace', 25)
        self.env_config.ego_index = self.ego_index
        self.env_config.enable_treatment_condition = self.enable_treatment
        self.env_config.episode_timesteps = self.episode_len

        if self.enable_treatment:
            self.env_config.altar_coords = config.get('altar_coords', (5, 15))

        # Build base environment (will be rebuilt on reset with proper seed)
        self._base_env = None
        self._env = None  # ResidentWrapper

        # Resident controller and info extractor
        self._resident_controller = ResidentController()
        self._info_extractor = ResidentInfoExtractor(
            num_players=self.num_players,
            permitted_color_index=config['permitted_color_index'],
            startup_grey_grace=config.get('startup_grey_grace', 25))

        # Metrics recorder
        if self.enable_telemetry:
            self._recorder = MetricsRecorder(
                num_players=self.num_players,
                ego_index=self.ego_index,
                permitted_color_index=config['permitted_color_index'],
                startup_grey_grace=config.get('startup_grey_grace', 25))
        else:
            self._recorder = None

        # Timestep counter
        self._current_timestep = 0

        # Seed
        self._seed = seed
        if seed is not None:
            self.seed(seed)

        # Multi-community mode setup (Phase 5)
        self.multi_community_mode = multi_community_mode
        if self.multi_community_mode:
            # Communities: 1=RED, 2=GREEN, 3=BLUE
            self.communities = [1, 2, 3]
            # Per-worker RNG for independent sampling
            self._community_rng = np.random.RandomState(seed)
            # Current community tracking
            self._current_community_idx = None
            self._current_community_name = None

        # Define observation space
        self.observation_space = self._make_observation_space()

        # Define action space (Discrete 11)
        self.action_space = spaces.Discrete(11)

        # Last dmlab2d timestep (for events access)
        self._last_dmlab_timestep = None

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'permitted_color_index': 1,  # RED
            'startup_grey_grace': 25,
            'episode_timesteps': 2000,
            'altar_coords': (5, 15),
            'alpha': 0.5,
            'beta': 0.5,
            'c': 0.2,
            'immunity_cooldown': 200,
        }

    def _make_observation_space(self) -> spaces.Dict:
        """Build observation space based on arm (treatment vs control).

        Base: RGB + READY_TO_SHOOT
        Optional: + TIMESTEP (if include_timestep=True)
        Treatment only: + PERMITTED_COLOR
        """
        obs_dict = {
            'rgb': spaces.Box(low=0, high=255, shape=(88, 88, 3), dtype=np.uint8),
            'ready_to_shoot': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }

        if self.include_timestep:
            # Optional: Include normalized timestep (t/T) ∈ [0,1]
            obs_dict['timestep'] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        if self.enable_treatment:
            # Treatment arm includes permitted color observation
            obs_dict['permitted_color'] = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        return spaces.Dict(obs_dict)

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        self._seed = seed
        if self._resident_controller is not None:
            self._resident_controller.reset(seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return initial observation.

        Args:
            seed: Optional seed to override initialization seed
            options: Optional dict (not used currently)

        Returns:
            observation: Dict with keys rgb, ready_to_shoot, timestep, [permitted_color]
            info: Dict with telemetry and metadata
        """
        # Update seed if provided
        if seed is not None:
            self.seed(seed)

        # Reset timestep counter
        self._current_timestep = 0

        # Sample community if multi-community mode (Phase 5)
        if self.multi_community_mode:
            # Randomly sample community (independent per worker)
            self._current_community_idx = self._community_rng.choice(self.communities)
            self._current_community_name = {1: 'RED', 2: 'GREEN', 3: 'BLUE'}[self._current_community_idx]

            # Update config with sampled community
            self.config['permitted_color_index'] = self._current_community_idx
            self.env_config.permitted_color_index = self._current_community_idx

            # Recreate info extractor with new community
            self._info_extractor = ResidentInfoExtractor(
                num_players=self.num_players,
                permitted_color_index=self._current_community_idx,
                startup_grey_grace=self.config.get('startup_grey_grace', 25),
            )

            # Recreate recorder with new community
            if self.enable_telemetry:
                self._recorder = MetricsRecorder(
                    num_players=self.num_players,
                    ego_index=self.ego_index,
                    permitted_color_index=self._current_community_idx,
                    startup_grey_grace=self.config.get('startup_grey_grace', 25),
                    community_tag=self._current_community_name,
                    community_idx=self._current_community_idx,
                )

        # Reset resident controller
        self._resident_controller.reset(seed=self._seed)

        # Reset info extractor
        self._info_extractor.reset()

        # Reset metrics recorder
        if self._recorder is not None:
            self._recorder.reset()

        # Build base environment
        roles = ["default"] * self.num_players
        self._base_env = substrate.build("allelopathic_harvest", roles, self.env_config)

        # Wrap with observation filter (treatment vs control)
        env_filtered = NormativeObservationFilter(
            self._base_env,
            enable_treatment_condition=self.enable_treatment)

        # Wrap with ResidentWrapper
        self._env = ResidentWrapper(
            env=env_filtered,
            resident_indices=self.resident_indices,
            ego_index=self.ego_index,
            resident_controller=self._resident_controller,
            info_extractor=self._info_extractor)

        # Reset wrapped environment
        dmlab_timestep = self._env.reset()
        self._last_dmlab_timestep = dmlab_timestep

        # Extract ego observation
        obs = self._extract_ego_observation(dmlab_timestep)

        # Build info dict
        info = {'timestep_count': self._current_timestep}

        # Add community tag if multi-community mode (Phase 5)
        if self.multi_community_mode:
            info['community_tag'] = self._current_community_name
            info['community_idx'] = self._current_community_idx

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step environment with ego action.

        Args:
            action: Integer action in [0, 10] from ego agent

        Returns:
            observation: Dict with ego observations
            reward: Scalar reward (r_train = r_env + alpha - beta - c)
            terminated: Always False (no early termination)
            truncated: True if t >= episode_len
            info: Dict with telemetry
        """
        if self._env is None:
            raise RuntimeError("Must call reset() before step()")

        # Step environment (residents act automatically via ResidentWrapper)
        dmlab_timestep = self._env.step(action)
        self._last_dmlab_timestep = dmlab_timestep

        # Get events from base environment
        events = self._base_env.events()

        # Record telemetry if enabled
        if self._recorder is not None:
            self._recorder.record_step(
                t=self._current_timestep,
                timestep=dmlab_timestep,
                events=events,
                ego_action=action)

        # Extract ego observation
        obs = self._extract_ego_observation(dmlab_timestep)

        # Extract ego reward (r_train from timestep)
        # timestep.reward is already r_total = r_env + alpha - beta - c
        reward = float(dmlab_timestep.reward[self.ego_index])

        # Increment timestep
        self._current_timestep += 1

        # Check termination/truncation
        terminated = False  # No early termination in this environment
        truncated = (self._current_timestep >= self.episode_len)

        # Build info dict
        info = {
            'timestep_count': self._current_timestep,
            'events': events,
        }

        # Add telemetry if enabled
        if self._recorder is not None:
            info['r_eval'] = self._recorder.get_r_eval()
            info['ego_body_color'] = self._recorder.get_ego_body_color()

        # Add community tag if multi-community mode (Phase 5)
        if self.multi_community_mode:
            info['community_tag'] = self._current_community_name
            info['community_idx'] = self._current_community_idx

        return obs, reward, terminated, truncated, info

    def _extract_ego_observation(self, dmlab_timestep) -> Dict[str, np.ndarray]:
        """Extract ego observation from dmlab2d timestep.

        Args:
            dmlab_timestep: dm_env.TimeStep from wrapped environment

        Returns:
            Dict with keys: rgb, ready_to_shoot, [timestep], [permitted_color]
        """
        ego_obs_raw = dmlab_timestep.observation[self.ego_index]

        # Extract RGB
        rgb = ego_obs_raw['RGB']  # (88, 88, 3) uint8

        # Extract READY_TO_SHOOT (scalar → (1,) float32)
        ready_to_shoot = np.array([ego_obs_raw['READY_TO_SHOOT']], dtype=np.float32)

        # Build observation dict
        obs = {
            'rgb': rgb,
            'ready_to_shoot': ready_to_shoot,
        }

        # Optionally add normalized timestep (t / episode_len) ∈ [0, 1]
        if self.include_timestep:
            timestep_norm = np.array([self._current_timestep / self.episode_len], dtype=np.float32)
            obs['timestep'] = timestep_norm

        # Add PERMITTED_COLOR if treatment arm
        if self.enable_treatment:
            permitted_color_onehot = ego_obs_raw['PERMITTED_COLOR']  # (3,) float64
            obs['permitted_color'] = permitted_color_onehot.astype(np.float32)

        return obs

    def render(self, mode='rgb_array'):
        """Render environment (return RGB array)."""
        if self._last_dmlab_timestep is None:
            return np.zeros((88, 88, 3), dtype=np.uint8)

        ego_obs = self._last_dmlab_timestep.observation[self.ego_index]
        return ego_obs['RGB']

    def close(self):
        """Close environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._base_env is not None:
            self._base_env.close()
            self._base_env = None

    def get_recorder(self) -> Optional[MetricsRecorder]:
        """Get metrics recorder (for evaluation)."""
        return self._recorder


# Vectorization helpers

def make_vec_env_treatment(
    num_envs: int,
    config: Dict,
    seeds: Optional[List[int]] = None,
    enable_telemetry: bool = False,
    include_timestep: bool = False,
) -> gym.vector.VectorEnv:
    """Create vectorized treatment environments.

    Args:
        num_envs: Number of parallel environments
        config: Configuration dict
        seeds: List of seeds (length must equal num_envs), or None for random
        enable_telemetry: Whether to enable MetricsRecorder (default False for training)
        include_timestep: Include normalized timestep in observations (default False)

    Returns:
        Vectorized environment (SubprocVecEnv for isolation)
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if seeds is None:
        seeds = [None] * num_envs
    elif len(seeds) != num_envs:
        raise ValueError(f"seeds length ({len(seeds)}) must equal num_envs ({num_envs})")

    def make_env(rank):
        def _init():
            env = AllelopathicHarvestGymEnv(
                arm='treatment',
                config=config,
                seed=seeds[rank],
                enable_telemetry=enable_telemetry,
                include_timestep=include_timestep,
            )
            return env
        return _init

    return SubprocVecEnv([make_env(i) for i in range(num_envs)])


def make_vec_env_control(
    num_envs: int,
    config: Dict,
    seeds: Optional[List[int]] = None,
    enable_telemetry: bool = False,
    include_timestep: bool = False,
) -> gym.vector.VectorEnv:
    """Create vectorized control environments.

    Args:
        num_envs: Number of parallel environments
        config: Configuration dict
        seeds: List of seeds (length must equal num_envs), or None for random
        enable_telemetry: Whether to enable MetricsRecorder (default False for training)
        include_timestep: Include normalized timestep in observations (default False)

    Returns:
        Vectorized environment (SubprocVecEnv for isolation)
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if seeds is None:
        seeds = [None] * num_envs
    elif len(seeds) != num_envs:
        raise ValueError(f"seeds length ({len(seeds)}) must equal num_envs ({num_envs})")

    def make_env(rank):
        def _init():
            env = AllelopathicHarvestGymEnv(
                arm='control',
                config=config,
                seed=seeds[rank],
                enable_telemetry=enable_telemetry,
                include_timestep=include_timestep,
            )
            return env
        return _init

    return SubprocVecEnv([make_env(i) for i in range(num_envs)])


def make_vec_env_multi_community(
    arm: str,
    num_envs: int,
    config: Dict,
    seed: int,
    enable_telemetry: bool = False,
    include_timestep: bool = False,
) -> gym.vector.VectorEnv:
    """Create vectorized multi-community environments (Phase 5).

    Each worker independently samples community (RED/GREEN/BLUE) at each reset.
    This ensures unbiased mixture gradients and avoids schedule confounding.

    Args:
        arm: 'treatment' or 'control'
        num_envs: Number of parallel environments
        config: Base configuration dict (permitted_color_index will be sampled)
        seed: Base random seed (each worker gets seed+rank)
        enable_telemetry: Whether to enable MetricsRecorder (default False for training)
        include_timestep: Include normalized timestep in observations (default False)

    Returns:
        Vectorized environment (SubprocVecEnv for isolation)

    Example:
        With num_envs=32, seed=42, each worker samples community independently:
        - Worker 0: seed=42, samples RED → GREEN → BLUE → ...
        - Worker 1: seed=43, samples GREEN → RED → RED → ...
        - Worker 2: seed=44, samples BLUE → GREEN → RED → ...
        - ...

        Over many episodes, law of large numbers ensures ~1:1:1 ratio.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env(rank):
        def _init():
            env = AllelopathicHarvestGymEnv(
                arm=arm,
                config=config.copy(),  # Copy to avoid shared state
                seed=seed + rank,  # Different seed per worker
                enable_telemetry=enable_telemetry,
                multi_community_mode=True,  # Enable Phase 5 sampling
                include_timestep=include_timestep,
            )
            return env
        return _init

    return SubprocVecEnv([make_env(i) for i in range(num_envs)])
