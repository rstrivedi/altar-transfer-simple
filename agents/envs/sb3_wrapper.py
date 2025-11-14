# Added by RST: Gymnasium wrapper for SB3 PPO training (Phase 4 & 5)
"""Gymnasium wrapper for single-ego RL training with scripted residents.

This wrapper:
- Exposes agent 0 (ego) as a Gymnasium environment for SB3 PPO training
- Steps agents 1-15 (residents) automatically via ResidentPolicy (observation-based)
- Returns observations: RGB, READY_TO_SHOOT, TIMESTEP, [permitted_color in treatment]
- Returns rewards: r_train = r_env + alpha - beta - c (alpha for training bonus)
- Integrates with MetricsRecorder for telemetry capture
- Supports treatment (with ALTAR→permitted_color) and control (without) arms
- Supports multi-community mode (Phase 5) for distributional competence training

Note: Substrate provides ALTAR observation (scalar altar color index), which we
convert to one-hot 'permitted_color' for the policy.

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

import meltingpot.substrate as substrate
from meltingpot.configs.substrates import allelopathic_harvest_normative__open as allelopathic_harvest

from agents.envs.normative_observation_filter import NormativeObservationFilter
from agents.envs.resident_wrapper import ResidentWrapper
# Added by RST: Removed old imports (ResidentController, ResidentInfoExtractor)
# New implementation uses ResidentPolicy via ResidentWrapper
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
            arm: 'treatment' or 'control' (controls permitted_color observation exposure)
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

        # Don't create ConfigDict in __init__ - will create fresh in reset() to avoid pickle issues
        self.env_config = None

        # Build base environment (will be rebuilt on reset with proper seed)
        self._base_env = None
        self._env = None  # ResidentWrapper

        # Added by RST: Removed old ResidentController and ResidentInfoExtractor
        # New implementation uses ResidentPolicy directly via ResidentWrapper

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

    def _build_env_config(self, permitted_color_index: int):
        """Build substrate config with given permitted_color_index.

        Args:
            permitted_color_index: 1=RED, 2=GREEN, 3=BLUE

        Returns:
            ConfigDict for substrate
        """
        env_config = allelopathic_harvest.get_config()
        with env_config.unlocked():
            env_config.normative_gate = True
            env_config.permitted_color_index = permitted_color_index
            env_config.startup_grey_grace = self.config.get('startup_grey_grace', 25)
            env_config.ego_index = self.ego_index
            env_config.enable_treatment_condition = self.enable_treatment
            env_config.episode_timesteps = self.episode_len

            if self.enable_treatment:
                env_config.altar_coords = self.config.get('altar_coords', (5, 15))

        return env_config

    def _make_observation_space(self) -> spaces.Dict:
        """Build observation space based on arm (treatment vs control).

        Base: RGB + READY_TO_SHOOT
        Optional: + TIMESTEP (if include_timestep=True)
        Treatment only: + permitted_color (converted from ALTAR observation)
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
        # Seed is passed to substrate and ResidentWrapper during reset()
        # No need to do anything here

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

        # Build/sample env config (ConfigDict doesn't survive pickling, so create fresh each reset)
        if self.multi_community_mode:
            # Multi-community: Randomly sample community (independent per worker)
            # Cast to int to avoid numpy.int64 vs int type mismatch with ConfigDict
            self._current_community_idx = int(self._community_rng.choice(self.communities))
            self._current_community_name = {1: 'RED', 2: 'GREEN', 3: 'BLUE'}[self._current_community_idx]

            # Build fresh config for this community
            self.env_config = self._build_env_config(self._current_community_idx)
            # Update wrapper config tracking
            self.config['permitted_color_index'] = self._current_community_idx

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
        else:
            # Single-community: Build config fresh (avoid pickle corruption)
            self.env_config = self._build_env_config(self.config['permitted_color_index'])

        # Added by RST: Removed reset calls for ResidentController and ResidentInfoExtractor
        # New implementation doesn't need these

        # Reset metrics recorder
        if self._recorder is not None:
            self._recorder.reset()

        # Build base environment (following altar-transfer/agents/sa/wrappers/environment_wrapper.py)
        roles = ["default"] * self.num_players

        # Build substrate using direct module API (not build_from_config - that breaks with pickling)
        from meltingpot.utils.substrates import substrate as mp_substrate_utils

        # Call the substrate module's build function directly to get lab2d_settings
        substrate_definition = allelopathic_harvest.build(roles=roles, config=self.env_config)

        # Build the substrate from lab2d_settings
        self._base_env = mp_substrate_utils.build_substrate(
            lab2d_settings=substrate_definition,
            individual_observations=self.env_config.individual_observation_names,
            global_observations=self.env_config.global_observation_names,
            action_table=self.env_config.action_set,
        )

        # Wrap with observation filter (treatment vs control)
        # Added by RST: Pass ego_index so filter only affects ego, not residents
        env_filtered = NormativeObservationFilter(
            self._base_env,
            enable_treatment_condition=self.enable_treatment,
            ego_index=self.ego_index)

        # Wrap with ResidentWrapper
        # Added by RST: Updated to use new ResidentWrapper API (uses ResidentPolicy)
        self._env = ResidentWrapper(
            env=env_filtered,
            resident_indices=self.resident_indices,
            ego_index=self.ego_index,
            seed=self._seed)

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
        # dmlab2d events are tuples: (event_name, event_data_list)
        # event_data_list is a flat list: [b'dict', b'key1', value1, b'key2', value2, ...]
        # Convert to dict format with 'name' key for recorder
        raw_events = self._base_env.events()
        events = []
        for event in raw_events:
            # # Debug: Print first 3 sanction events to see raw format
            # is_sanction = (isinstance(event, tuple) and len(event) > 0 and event[0] == 'sanction') or \
            #               (isinstance(event, dict) and event.get('name') == 'sanction')
            # if is_sanction and self._debug_sanction_count < 3:
            #     print(f"DEBUG sb3_wrapper t={self._current_timestep}: Raw sanction event: type={type(event)}, len={len(event) if isinstance(event, tuple) else 'N/A'}")
            #     print(f"  Full event: {event}")
            #     self._debug_sanction_count += 1

            if isinstance(event, tuple) and len(event) >= 2:
                # Convert (event_name, data_list) to {'name': event_name, **data}
                event_dict = {'name': event[0]}

                # Parse flat list format: [b'dict', b'key1', val1, b'key2', val2, ...]
                if isinstance(event[1], list) and len(event[1]) > 0:
                    data_list = event[1]
                    # Skip first element (b'dict' type marker)
                    # Then parse pairs: [b'key', value, b'key', value, ...]
                    i = 1
                    while i < len(data_list) - 1:
                        key = data_list[i]
                        value = data_list[i + 1]
                        # Decode bytes to string for key
                        if isinstance(key, bytes):
                            key = key.decode('utf-8')
                        # Extract scalar from numpy array
                        if hasattr(value, 'item'):
                            value = value.item()
                        event_dict[key] = value
                        i += 2
                elif isinstance(event[1], dict):
                    # Fallback for dict format (if it ever happens)
                    event_dict.update(event[1])

                events.append(event_dict)
            elif isinstance(event, dict):
                # Already in dict format
                events.append(event)

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

            # Add episode summary when episode ends (for SB3 callbacks and logging)
            if truncated:
                # Get berry counts from WORLD observations
                world_obs = dmlab_timestep.observation[0]  # WORLD observations are usually in agent 0
                berries_by_type = world_obs.get('WORLD.BERRIES_BY_TYPE', np.zeros(3))

                # Get episode summary from recorder (includes sanction metrics)
                episode_summary = self._recorder.get_episode_summary()

                # Override berry counts with values from WORLD observations (more reliable)
                episode_summary['berries_planted_red'] = int(berries_by_type[0]) if len(berries_by_type) > 0 else 0
                episode_summary['berries_planted_green'] = int(berries_by_type[1]) if len(berries_by_type) > 1 else 0
                episode_summary['berries_planted_blue'] = int(berries_by_type[2]) if len(berries_by_type) > 2 else 0

                info['episode'] = episode_summary

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

        # Edited by RST: Add ALTAR observation (permitted color) if treatment arm
        # ALTAR is scalar (1=RED, 2=GREEN, 3=BLUE in Lua 1-indexed), convert to one-hot
        if self.enable_treatment:
            altar_color_index = int(ego_obs_raw['ALTAR'])  # Scalar: 1, 2, or 3 (Lua 1-indexed)
            # Convert to one-hot (3,) float32: [1,0,0]=RED, [0,1,0]=GREEN, [0,0,1]=BLUE
            permitted_color_onehot = np.zeros(3, dtype=np.float32)
            permitted_color_onehot[altar_color_index - 1] = 1.0  # -1 for Python 0-indexing
            obs['permitted_color'] = permitted_color_onehot

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
