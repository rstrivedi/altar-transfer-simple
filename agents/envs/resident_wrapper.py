# Added by RST: Wrapper to integrate policy-based residents with environment
"""Wrapper that automatically acts scripted residents while allowing ego control.

This is the NEW implementation using policy.Policy interface.

In training mode (ego_index=0), agent 0 is controlled by the learner while
agents 1-15 act according to ResidentPolicy.

In baseline mode (ego_index=None), all agents act according to ResidentPolicy.
"""

from typing import Dict, List, Optional

import dm_env
import numpy as np

from agents.residents.resident_policy import ResidentPolicy


class ResidentWrapper:
  """Wrapper that manages resident agent actions automatically.

  This wrapper intercepts environment interactions and fills in actions for
  resident agents using ResidentPolicy instances. In training mode, only ego's
  action comes from the external agent; residents act automatically.

  NEW: Uses policy.Policy interface, NO event parsing.
  """

  def __init__(
      self,
      env,
      resident_indices: List[int],
      ego_index: Optional[int],
      seed: Optional[int] = None):
    """Initialize the ResidentWrapper.

    Args:
      env: Base dmlab2d environment to wrap.
      resident_indices: List of agent indices that are residents (0-indexed).
      ego_index: Index of ego agent, or None if all agents are residents.
      seed: Random seed for resident policies.
    """
    self._env = env
    self._resident_indices = resident_indices
    self._ego_index = ego_index

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

    # Create one ResidentPolicy instance per resident agent
    # Added by RST: Create policy instances with different seeds for diversity
    self._resident_policies: Dict[int, ResidentPolicy] = {}
    self._resident_states: Dict[int, any] = {}

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    for agent_id in resident_indices:
      # Use different seed for each resident for behavioral diversity
      policy_seed = rng.randint(0, 2**31 - 1)
      policy = ResidentPolicy(seed=policy_seed)
      self._resident_policies[agent_id] = policy
      self._resident_states[agent_id] = policy.initial_state()

    # Store last timestep for accessing observations
    self._last_timestep = None

    # Edited by RST: Store last actions for baseline tracking
    self._last_actions = None

  def reset(self, *args, **kwargs) -> dm_env.TimeStep:
    """Reset environment and resident policies.

    Args:
      *args: Forwarded to base environment.
      **kwargs: Forwarded to base environment.

    Returns:
      Initial timestep from environment.
    """
    # Reset environment
    timestep = self._env.reset(*args, **kwargs)

    # Reset all resident policy states
    # Added by RST: Reset policy state on episode start
    for agent_id, policy in self._resident_policies.items():
      self._resident_states[agent_id] = policy.initial_state()

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

    # Build action list for all agents
    # Added by RST: Get actions from policies using observations ONLY
    actions = []
    for agent_id in range(self._num_players):
      if agent_id == self._ego_index:
        # Ego agent: use provided action
        actions.append(ego_action)
      elif agent_id in self._resident_indices:
        # Resident agent: use policy
        # Extract per-agent timestep from multi-agent observation
        agent_timestep = self._extract_agent_timestep(self._last_timestep, agent_id)

        # Get action from policy
        policy = self._resident_policies[agent_id]
        prev_state = self._resident_states[agent_id]
        action, new_state = policy.step(agent_timestep, prev_state)

        # Update state
        self._resident_states[agent_id] = new_state
        actions.append(action)
      else:
        # Should not happen if validation is correct
        raise ValueError(f"Agent {agent_id} is neither ego nor resident")

    # Step environment with all actions
    timestep = self._env.step(actions)

    # Store timestep for next step() call
    self._last_timestep = timestep

    # Edited by RST: Store actions for baseline tracking
    self._last_actions = actions

    return timestep

  def _extract_agent_timestep(self, timestep: dm_env.TimeStep, agent_id: int) -> dm_env.TimeStep:
    """Extract per-agent timestep from multi-agent observation.

    MeltingPot environments return observations as a list (one per agent).
    This extracts the observation for a specific agent and wraps it in a
    dm_env.TimeStep for the policy.

    Args:
      timestep: Multi-agent timestep from environment.
      agent_id: Agent index (0-indexed).

    Returns:
      Per-agent timestep with single observation dict.
    """
    # Added by RST: Extract agent's observation from list
    observations = timestep.observation
    if isinstance(observations, list):
      agent_obs = observations[agent_id]
    else:
      # Single observation dict (shouldn't happen in multi-agent env)
      agent_obs = observations

    # Create per-agent timestep
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward,
        discount=timestep.discount,
        observation=agent_obs
    )

  def observation_spec(self):
    """Forward to wrapped environment."""
    return self._env.observation_spec()

  def action_spec(self):
    """Forward to wrapped environment."""
    return self._env.action_spec()

  def get_last_action(self, agent_id: int) -> Optional[int]:
    """Get the last action taken by a specific agent.

    Added by RST: For baseline tracking where we need to know resident actions.

    Args:
      agent_id: Agent index (0-indexed).

    Returns:
      Last action taken by agent, or None if no step has occurred yet.
    """
    if self._last_actions is None:
      return None
    if agent_id < 0 or agent_id >= len(self._last_actions):
      return None
    return self._last_actions[agent_id]

  def close(self):
    """Close all policies and wrapped environment."""
    # Added by RST: Close all policies
    for policy in self._resident_policies.values():
      policy.close()
    return self._env.close()

  def __getattr__(self, name):
    """Forward all other attributes to wrapped environment."""
    return getattr(self._env, name)
