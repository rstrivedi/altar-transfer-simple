# Added by RST: Observation filter wrapper for control vs treatment manipulation
"""Wrapper that filters PERMITTED_COLOR observation based on treatment condition.

In treatment condition, ego sees the PERMITTED_COLOR observation (institutional signal).
In control condition, ego does not see PERMITTED_COLOR (must infer rule from sanctions).
"""

import copy
from typing import Mapping, Sequence

import dm_env
import numpy as np


_ALTAR_OBS = "ALTAR"  # Edited by RST: Use ALTAR (substrate provides this, not PERMITTED_COLOR)


class NormativeObservationFilter:
  """Wrapper that filters institutional observation based on treatment condition.

  When enable_treatment_condition=True, keeps ALTAR observation for ego.
  When enable_treatment_condition=False, removes ALTAR observation from ego only.

  IMPORTANT: Residents (non-ego agents) ALWAYS see ALTAR observation regardless of
  treatment condition. This ensures residents can enforce the correct norm in both
  arms and across all communities in multi-community training.
  """

  def __init__(self, env, enable_treatment_condition: bool = False, ego_index: int = 0):
    """Initializes the wrapper.

    Args:
      env: dmlab2d.Environment to wrap.
      enable_treatment_condition: If True, keep ALTAR observation for ego
        (treatment). If False, remove ALTAR observation from ego only (control).
      ego_index: Index of ego agent (0-indexed). Only this agent's observations
        are filtered in control condition. All other agents always see ALTAR.
    """
    self._env = env
    self._enable_treatment = enable_treatment_condition
    self._ego_index = ego_index

  def _filter_observation(self, obs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Remove ALTAR from observation if in control condition.

    Args:
      obs: Single player's observation dictionary.

    Returns:
      Filtered observation dictionary.
    """
    if self._enable_treatment:
      # Treatment: keep all observations including ALTAR
      return obs
    else:
      # Control: remove ALTAR observation
      filtered_obs = {k: v for k, v in obs.items() if k != _ALTAR_OBS}
      return filtered_obs

  def _get_timestep(self, input_timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Returns timestep with filtered observations.

    Args:
      input_timestep: input_timestep before filtering.

    Returns:
      Timestep with observations filtered based on treatment condition.

    Added by RST: Only filter ego observations in control. Residents always
    see ALTAR so they can enforce the correct norm in both arms.
    """
    filtered_observations = []
    for agent_idx, obs in enumerate(input_timestep.observation):
      if agent_idx == self._ego_index:
        # Ego agent: apply filter based on treatment condition
        filtered_observations.append(self._filter_observation(obs))
      else:
        # Resident agents: always keep all observations (including ALTAR)
        filtered_observations.append(obs)

    return dm_env.TimeStep(
        step_type=input_timestep.step_type,
        reward=input_timestep.reward,
        discount=input_timestep.discount,
        observation=filtered_observations)

  def reset(self, *args, **kwargs) -> dm_env.TimeStep:
    """See base class."""
    timestep = self._env.reset(*args, **kwargs)
    return self._get_timestep(timestep)

  def step(
      self, actions: Sequence[Mapping[str, np.ndarray]]) -> dm_env.TimeStep:
    """See base class."""
    timestep = self._env.step(actions)
    return self._get_timestep(timestep)

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """Returns observation spec with ALTAR removed from ego if in control.

    Added by RST: Only remove ALTAR from ego's spec in control condition.
    Residents always keep ALTAR in their spec.
    """
    observation_spec = copy.deepcopy(self._env.observation_spec())

    if not self._enable_treatment:
      # Control condition: remove ALTAR from ego's spec only
      if self._ego_index < len(observation_spec):
        ego_spec = observation_spec[self._ego_index]
        if _ALTAR_OBS in ego_spec:
          del ego_spec[_ALTAR_OBS]

    return observation_spec

  def action_spec(self):
    """Forward to wrapped environment."""
    return self._env.action_spec()

  def close(self):
    """Forward to wrapped environment."""
    return self._env.close()

  def __getattr__(self, name):
    """Forward all other attributes to wrapped environment."""
    return getattr(self._env, name)
