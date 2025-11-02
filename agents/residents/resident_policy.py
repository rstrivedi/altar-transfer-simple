"""ResidentPolicy - Normative sanctioning resident agents.

Added by RST: New implementation using policy.Policy interface.

This policy implements resident agents that:
1. Enforce the altar norm by sanctioning violators
2. Plant altar color berries to achieve 95%+ monoculture per episode
3. Explore to find berries and earn rewards

Priority order:
  P1: Sanction violators (grey or wrong color, not immune)
  P2: Plant altar color (30% of time, heuristic)
  P3: Explore/move (default, to find berries)
"""

from typing import Any, Optional, Tuple

import dm_env
import numpy as np
from meltingpot.utils.policies import policy


class ResidentPolicy(policy.Policy):
  """Resident policy for normative sanctioning substrate.

  Implements equilibrium play: always compliant, sanctions violators,
  achieves 95%+ monoculture per episode.

  Works for both single-environment and multi-environment training by
  reading ALTAR observation each step.
  """

  def __init__(self, seed: Optional[int] = None):
    """Initialize resident policy.

    Args:
      seed: Random seed for reproducibility.
    """
    self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    self._step_count = 0

    # Action definitions (from MeltingPot substrate)
    # Added by RST: Action indices for allelopathic harvest
    self._NOOP = 0
    self._FORWARD = 1
    self._BACKWARD = 2
    self._STEP_LEFT = 3
    self._STEP_RIGHT = 4
    self._TURN_LEFT = 5
    self._TURN_RIGHT = 6
    self._FIRE_ZAP = 7
    self._FIRE_RED = 8
    self._FIRE_GREEN = 9
    self._FIRE_BLUE = 10

    # Plant actions by altar color
    # Added by RST: Mapping from altar color ID to plant action
    self._plant_actions = {
        1: self._FIRE_RED,    # altar color 1 = red
        2: self._FIRE_GREEN,  # altar color 2 = green
        3: self._FIRE_BLUE,   # altar color 3 = blue
    }

    # Parameters
    # Added by RST: Policy parameters for sanctioning and planting
    self._grace_period = 50  # frames before sanctioning starts
    self._plant_probability = 0.9  # plant 90% of time when opportunity exists

  def initial_state(self) -> Any:
    """Return initial policy state (stateless policy)."""
    return None

  def step(self, timestep: dm_env.TimeStep, prev_state: Any) -> Tuple[int, Any]:
    """Select action based on observations.

    Priority:
      1. Sanction violators (if ready, in range, violating, not immune)
      2. Plant altar color (if planting opportunity, heuristic: step % 10 < 3)
      3. Explore (default, to find berries)

    Args:
      timestep: Current environment timestep with observations.
      prev_state: Previous policy state (unused, stateless policy).

    Returns:
      Tuple of (action, new_state).
    """
    obs = timestep.observation
    self._step_count += 1

    # === Extract Observations ===
    # Added by RST: Extract all observations needed for decision making

    # Altar color (1=red, 2=green, 3=blue)
    altar_color = self._extract_scalar(obs.get('ALTAR', 1))

    # Ready to shoot (0.0 or 1.0)
    ready_to_shoot = self._extract_scalar(obs.get('READY_TO_SHOOT', 0.0))

    # Agent colors (int tensor, 0=grey, 1=red, 2=green, 3=blue)
    agent_colors = self._extract_tensor(obs.get('AGENT_COLORS', np.array([])))

    # Immunity status (int tensor, 0=not immune, 1=immune)
    immunity_status = self._extract_tensor(obs.get('IMMUNITY_STATUS', np.array([])))

    # Who is in range to zap (int tensor, 0=not in range, 1=in range)
    ids_in_range = self._extract_tensor(obs.get('AVATAR_IDS_IN_RANGE_TO_ZAP', np.array([])))

    # This agent's player index (1-indexed in Lua, converted to int)
    player_index = int(self._extract_scalar(obs.get('PLAYER_INDEX', 1)))

    # === P1: SANCTIONING ===
    # Added by RST: Check if we should sanction a violator
    if ready_to_shoot > 0.5 and self._step_count >= self._grace_period:
      # Edited by RST: First check if there's a violator in range AND no compliant agents in the way
      # ZAP hits ALL agents in the beam line, so we can only fire if all targets are violators
      num_players = len(agent_colors)
      has_violator_in_range = False
      has_compliant_in_range = False

      for lua_idx in range(1, num_players + 1):
        # Skip self
        if lua_idx == player_index:
          continue

        # Convert to 0-indexed array
        array_idx = lua_idx - 1

        # Check if in range
        if array_idx >= len(ids_in_range) or ids_in_range[array_idx] == 0:
          continue

        # Check if immune (already sanctioned)
        is_immune = (array_idx < len(immunity_status) and immunity_status[array_idx] == 1)

        # Check if violating (grey OR wrong color)
        color = agent_colors[array_idx]
        is_violating = (color == 0) or (color != altar_color)

        # Track if there's a violator and/or compliant agent in range
        if not is_immune:
          if is_violating:
            has_violator_in_range = True
          else:
            has_compliant_in_range = True

      # Only fire ZAP if there's a violator in range AND no compliant agents in the way
      if has_violator_in_range and not has_compliant_in_range:
        return self._FIRE_ZAP, None

    # === P2: PLANTING ===
    # Added by RST: Plant altar color when planting opportunity exists
    # Heuristic from NormativeMonoculturePolicy: step % 10 < 3 (30% of time)
    can_plant = self._check_can_plant()
    if can_plant and self._rng.random() < self._plant_probability:
      return self._plant_actions[altar_color], None

    # === P3: EXPLORE ===
    # Added by RST: Move around to find berries (default action)
    return self._explore_action(), None

  def _extract_scalar(self, value: Any) -> float:
    """Extract scalar value from observation.

    Handles both raw scalars and numpy arrays.

    Args:
      value: Observation value (scalar, array, or tensor).

    Returns:
      Float scalar value.
    """
    if isinstance(value, np.ndarray):
      return float(value.item() if value.size == 1 else value.flat[0])
    return float(value)

  def _extract_tensor(self, value: Any) -> np.ndarray:
    """Extract tensor from observation.

    Handles both numpy arrays and raw values.

    Args:
      value: Observation value (array or scalar).

    Returns:
      Numpy array.
    """
    if isinstance(value, np.ndarray):
      return value
    return np.array(value)

  def _check_can_plant(self) -> bool:
    """Check if planting opportunity exists.

    Uses random probability to maintain body color compliance.
    Plant 70% of the time randomly distributed (not in bursts).

    Returns:
      True if should attempt planting, False otherwise.
    """
    # Added by RST: Heuristic planting - 70% rate for high compliance
    # Edited by RST: Changed from cyclical (% 10 < 7) to random for smooth interleaved planting
    return self._rng.random() < 0.70

  def _explore_action(self) -> int:
    """Select exploration action to find berries.

    Uses weighted random selection favoring forward movement with
    occasional turns and side steps.

    Returns:
      Action index for movement/exploration.
    """
    # Added by RST: Exploration movement pattern
    # Mix of forward movement, turning, and side stepping
    action_probs = [
        (self._FORWARD, 0.5),      # Move forward most often
        (self._TURN_LEFT, 0.15),   # Turn occasionally
        (self._TURN_RIGHT, 0.15),
        (self._STEP_LEFT, 0.1),    # Side step less often
        (self._STEP_RIGHT, 0.1),
    ]

    rand = self._rng.random()
    cumsum = 0.0
    for action, prob in action_probs:
      cumsum += prob
      if rand < cumsum:
        return action

    # Default to forward if something went wrong
    return self._FORWARD

  def close(self) -> None:
    """Cleanup resources (none needed for this policy)."""
    pass
