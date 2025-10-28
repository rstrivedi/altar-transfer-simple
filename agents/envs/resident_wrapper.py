# Added by RST: Wrapper to integrate scripted residents with environment
"""Wrapper that automatically acts scripted residents while allowing ego control.

In training mode (ego_index=0), agent 0 is controlled by the learner while
agents 1-15 act according to the scripted ResidentController.

In baseline mode (ego_index=None), all agents act according to ResidentController.
"""

from typing import List, Optional, Sequence, Mapping

import dm_env
import numpy as np

from agents.residents.info_extractor import ResidentInfoExtractor
from agents.residents.scripted_residents import ResidentController


class ResidentWrapper:
  """Wrapper that manages resident agent actions automatically.

  This wrapper intercepts environment interactions and fills in actions for
  resident agents using the ResidentController. In training mode, only ego's
  action comes from the external agent; residents act automatically.
  """

  def __init__(
      self,
      env,
      resident_indices: List[int],
      ego_index: Optional[int],
      resident_controller: ResidentController,
      info_extractor: ResidentInfoExtractor):
    """Initialize the ResidentWrapper.

    Args:
      env: Base dmlab2d environment to wrap.
      resident_indices: List of agent indices that are residents (0-indexed).
      ego_index: Index of ego agent, or None if all agents are residents.
      resident_controller: ResidentController instance for scripted behavior.
      info_extractor: ResidentInfoExtractor instance for parsing state.
    """
    self._env = env
    self._resident_indices = resident_indices
    self._ego_index = ego_index
    self._resident_controller = resident_controller
    self._info_extractor = info_extractor

    # Determine number of players from environment
    self._num_players = len(env.observation_spec())

    # Validate indices
    if ego_index is not None:
      if ego_index not in range(self._num_players):
        raise ValueError(f"ego_index {ego_index} out of range [0, {self._num_players})")
      if ego_index in resident_indices:
        raise ValueError(f"ego_index {ego_index} cannot be in resident_indices")

    for idx in resident_indices:
      if idx not in range(self._num_players):
        raise ValueError(f"resident_index {idx} out of range [0, {self._num_players})")

    # Check coverage
    if ego_index is None:
      # Baseline mode: all agents must be residents
      if len(resident_indices) != self._num_players:
        raise ValueError(
            f"In baseline mode (ego_index=None), all {self._num_players} agents "
            f"must be residents, but got {len(resident_indices)}")
    else:
      # Training mode: ego + residents must cover all agents
      if len(resident_indices) + 1 != self._num_players:
        raise ValueError(
            f"Expected {self._num_players - 1} residents + 1 ego, "
            f"but got {len(resident_indices)} residents")

    # Store last timestep for accessing observations
    self._last_timestep = None

  def reset(self, *args, **kwargs) -> dm_env.TimeStep:
    """Reset environment and resident controller.

    Args:
      *args: Forwarded to base environment.
      **kwargs: Forwarded to base environment.

    Returns:
      Initial timestep from environment.
    """
    # Reset info extractor
    self._info_extractor.reset()

    # Reset environment
    timestep = self._env.reset(*args, **kwargs)

    # Store timestep for future step() calls
    self._last_timestep = timestep

    return timestep

  def step(self, ego_action: Optional[int] = None) -> dm_env.TimeStep:
    """Step environment with ego action + automatically generated resident actions.

    Args:
      ego_action: Action for ego agent (required if ego_index is not None).
        If ego_index is None (baseline mode), this should be None.

    Returns:
      Timestep from environment after all agents act.
    """
    if self._last_timestep is None:
      raise RuntimeError("Must call reset() before step()")

    # Validate ego_action
    if self._ego_index is None:
      # Baseline mode: no ego, all residents
      if ego_action is not None:
        raise ValueError("ego_action must be None in baseline mode (ego_index=None)")
    else:
      # Training mode: ego_action required
      if ego_action is None:
        raise ValueError(f"ego_action required for ego_index={self._ego_index}")

    # Get observations from last timestep and current events
    observations = self._last_timestep.observation
    events = self._env.events()

    # Extract info for residents
    info = self._info_extractor.extract_info(observations, events)

    # Build action list for all agents
    actions = []
    for agent_id in range(self._num_players):
      if agent_id == self._ego_index:
        # Ego agent: use provided action
        actions.append(ego_action)
      elif agent_id in self._resident_indices:
        # Resident agent: use controller
        resident_action = self._resident_controller.act(agent_id, info)
        actions.append(resident_action)
      else:
        # Should not happen if validation is correct
        raise ValueError(f"Agent {agent_id} is neither ego nor resident")

    # Step environment with all actions
    timestep = self._env.step(actions)

    # Store timestep for next step() call
    self._last_timestep = timestep

    return timestep

  def observation_spec(self):
    """Forward to wrapped environment."""
    return self._env.observation_spec()

  def action_spec(self):
    """Forward to wrapped environment."""
    return self._env.action_spec()

  def close(self):
    """Forward to wrapped environment."""
    return self._env.close()

  def __getattr__(self, name):
    """Forward all other attributes to wrapped environment."""
    return getattr(self._env, name)
