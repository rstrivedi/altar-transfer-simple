# Added by RST: Observation filter wrapper for control vs treatment manipulation
"""Wrapper that filters PERMITTED_COLOR observation based on treatment condition.

In treatment condition, ego sees the PERMITTED_COLOR observation (institutional signal).
In control condition, ego does not see PERMITTED_COLOR (must infer rule from sanctions).
"""

import copy
from typing import Mapping, Sequence

import dm_env
import numpy as np


_PERMITTED_COLOR_OBS = "PERMITTED_COLOR"


class NormativeObservationFilter:
  """Wrapper that filters institutional observation based on treatment condition.

  When enable_treatment_condition=True, keeps PERMITTED_COLOR observation.
  When enable_treatment_condition=False, removes PERMITTED_COLOR observation.
  """

  def __init__(self, env, enable_treatment_condition: bool = False):
    """Initializes the wrapper.

    Args:
      env: dmlab2d.Environment to wrap.
      enable_treatment_condition: If True, keep PERMITTED_COLOR observation
        (treatment). If False, remove PERMITTED_COLOR observation (control).
    """
    self._env = env
    self._enable_treatment = enable_treatment_condition

  def _filter_observation(self, obs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Remove PERMITTED_COLOR from observation if in control condition.

    Args:
      obs: Single player's observation dictionary.

    Returns:
      Filtered observation dictionary.
    """
    if self._enable_treatment:
      # Treatment: keep all observations including PERMITTED_COLOR
      return obs
    else:
      # Control: remove PERMITTED_COLOR observation
      filtered_obs = {k: v for k, v in obs.items() if k != _PERMITTED_COLOR_OBS}
      return filtered_obs

  def _get_timestep(self, input_timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Returns timestep with filtered observations.

    Args:
      input_timestep: input_timestep before filtering.

    Returns:
      Timestep with observations filtered based on treatment condition.
    """
    filtered_observations = [
        self._filter_observation(obs) for obs in input_timestep.observation
    ]
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
    """Returns observation spec with PERMITTED_COLOR removed if in control."""
    observation_spec = copy.deepcopy(self._env.observation_spec())

    if not self._enable_treatment:
      # Control condition: remove PERMITTED_COLOR from spec
      for obs_spec in observation_spec:
        if _PERMITTED_COLOR_OBS in obs_spec:
          del obs_spec[_PERMITTED_COLOR_OBS]

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
