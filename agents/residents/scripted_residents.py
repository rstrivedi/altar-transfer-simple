# Added by RST: Scripted resident agents for normative allelopathic harvest
"""Deterministic scripted resident controller implementing equilibrium play.

Residents enforce the posted rule, harvest/replant permitted color, and patrol.
Policy priority: P1 (zap) > P2 (plant) > P3 (harvest) > P4 (patrol)
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np

from agents.residents import config as cfg


class ResidentController:
  """Deterministic scripted controller for resident agents."""

  def __init__(self):
    """Initialize the controller."""
    self._rng = None
    self._patrol_state = {}  # Per-resident patrol state

  def reset(self, seed: int = cfg.DEFAULT_SEED):
    """Reset controller for new episode.

    Args:
      seed: Random seed for patrol behavior.
    """
    self._rng = np.random.RandomState(seed)
    self._patrol_state = {}

  def act(self, resident_id: int, info: Dict) -> int:
    """Select action for a resident agent.

    Args:
      resident_id: Agent ID (0-indexed).
      info: Information dict from ResidentInfoExtractor.

    Returns:
      Action index (0-10).
    """
    # Get resident-specific info
    resident_info = info['residents'][resident_id]
    world_step = info['world_step']
    permitted_color_index = info['permitted_color_index']
    startup_grey_grace = info['startup_grey_grace']

    # P1: Enforce (Zap violators)
    zap_action = self._try_zap(
        resident_info, permitted_color_index, world_step, startup_grey_grace)
    if zap_action is not None:
      return zap_action

    # P2: Replant permitted color
    plant_action = self._try_plant(resident_info, permitted_color_index)
    if plant_action is not None:
      return plant_action

    # P3: Harvest (move toward ripe berry)
    harvest_action = self._try_harvest(resident_info)
    if harvest_action is not None:
      return harvest_action

    # P4: Patrol (fallback)
    return self._patrol(resident_id)

  def _is_violation(
      self, body_color: int, permitted: int, world_step: int, grace: int) -> bool:
    """Check if an agent is violating the norm.

    Args:
      body_color: Agent's body color (0=GREY, 1=RED, 2=GREEN, 3=BLUE).
      permitted: Permitted color index (1, 2, or 3).
      world_step: Current world step.
      grace: Grace period for grey agents (frames).

    Returns:
      True if violating, False if compliant.
    """
    # Violation: (color != permitted) OR (grey AND past grace period)
    if body_color != permitted:
      if body_color == cfg.COLOR_GREY and world_step < grace:
        return False  # Grey agent within grace period
      return True  # Wrong color or grey past grace
    return False  # Correct color

  def _is_eligible(
      self, target: Dict, permitted: int, world_step: int, grace: int) -> bool:
    """Check if target is eligible for sanctioning.

    Args:
      target: Target agent info dict with 'body_color' and 'immune_ticks_remaining'.
      permitted: Permitted color index.
      world_step: Current world step.
      grace: Grace period for grey agents.

    Returns:
      True if eligible (violating AND not immune), False otherwise.
    """
    is_violating = self._is_violation(
        target['body_color'], permitted, world_step, grace)
    is_not_immune = target['immune_ticks_remaining'] == 0
    return is_violating and is_not_immune

  def _try_zap(
      self, resident_info: Dict, permitted: int, world_step: int, grace: int
  ) -> Optional[int]:
    """P1: Try to zap eligible violators.

    Args:
      resident_info: Resident's info dict.
      permitted: Permitted color index.
      world_step: Current world step.
      grace: Grace period for grey agents.

    Returns:
      ACTION_FIRE_ZAP if should zap, None otherwise.
    """
    # Check if zap is off cooldown
    if resident_info['zap_cooldown_remaining'] > 0:
      return None

    # Get nearby agents
    nearby_agents = resident_info['nearby_agents']

    # Filter for eligible targets
    eligible_targets = [
        agent for agent in nearby_agents
        if self._is_eligible(agent, permitted, world_step, grace)
    ]

    if not eligible_targets:
      return None

    # Select nearest target (tie-break: lowest agent_id)
    # Calculate distance for each target
    targets_with_distance = []
    for target in eligible_targets:
      rel_x, rel_y = target['rel_pos']
      distance = math.sqrt(rel_x ** 2 + rel_y ** 2)
      targets_with_distance.append((distance, target['agent_id'], target))

    # Sort by distance, then agent_id
    targets_with_distance.sort(key=lambda x: (x[0], x[1]))

    # Select first (nearest, lowest id)
    selected_target = targets_with_distance[0][2]

    return cfg.ACTION_FIRE_ZAP

  def _try_plant(self, resident_info: Dict, permitted: int) -> Optional[int]:
    """P2: Try to plant permitted color on unripe patch.

    Args:
      resident_info: Resident's info dict.
      permitted: Permitted color index.

    Returns:
      Plant action if should plant, None otherwise.
    """
    if resident_info['standing_on_unripe']:
      return cfg.PLANT_ACTION_MAP[permitted]
    return None

  def _try_harvest(self, resident_info: Dict) -> Optional[int]:
    """P3: Try to move toward nearest ripe berry.

    Args:
      resident_info: Resident's info dict.

    Returns:
      Movement action if should harvest, None otherwise.
    """
    nearby_berries = resident_info.get('nearby_berries', [])

    if not nearby_berries:
      return None

    # Select nearest berry
    nearest_berry = min(nearby_berries, key=lambda b: b['distance'])
    rel_x, rel_y = nearest_berry['rel_pos']

    # Greedy pathfinding: turn toward berry, then move forward
    # Determine desired direction (angle to berry)
    angle_to_berry = math.atan2(rel_y, rel_x)  # Radians

    # Convert to cardinal direction (0=right, 90=up, 180=left, 270=down)
    # Assume agent's forward direction is along +x axis initially
    # (This is a simplification; ideally we'd track agent orientation)
    # For now, use a heuristic: move in direction of largest displacement

    abs_x = abs(rel_x)
    abs_y = abs(rel_y)

    # If berry is very close, just move forward
    if abs_x <= 0.5 and abs_y <= 0.5:
      return cfg.ACTION_FORWARD

    # Otherwise, use greedy approach:
    # If more horizontal displacement, turn left/right
    # If more vertical displacement, move forward/backward
    # This is a simple heuristic; not optimal but reasonable
    if abs_x > abs_y:
      # Horizontal displacement dominant
      if rel_x > 0:
        # Berry is to the right, turn right or move forward depending on current orientation
        # For simplicity, alternate between turning and moving
        return cfg.ACTION_TURN_RIGHT
      else:
        # Berry is to the left
        return cfg.ACTION_TURN_LEFT
    else:
      # Vertical displacement dominant
      if rel_y > 0:
        # Berry is up, move forward
        return cfg.ACTION_FORWARD
      else:
        # Berry is down, turn around (turn twice) or move backward
        # For simplicity, just turn
        return cfg.ACTION_TURN_LEFT

  def _patrol(self, resident_id: int) -> int:
    """P4: Patrol with persistence.

    Args:
      resident_id: Agent ID.

    Returns:
      Patrol action (FORWARD, TURN_LEFT, or TURN_RIGHT).
    """
    # Get or initialize patrol state for this resident
    if resident_id not in self._patrol_state:
      self._patrol_state[resident_id] = {
          'action': self._random_patrol_action(),
          'frames_remaining': cfg.PATROL_PERSISTENCE
      }

    state = self._patrol_state[resident_id]

    # Decrement frames
    state['frames_remaining'] -= 1

    # If persistence expired, pick new action
    if state['frames_remaining'] <= 0:
      state['action'] = self._random_patrol_action()
      state['frames_remaining'] = cfg.PATROL_PERSISTENCE

    return state['action']

  def _random_patrol_action(self) -> int:
    """Pick a random patrol action.

    Returns:
      Random action from PATROL_DIRECTIONS.
    """
    direction = self._rng.choice(cfg.PATROL_DIRECTIONS)
    if direction == 'FORWARD':
      return cfg.ACTION_FORWARD
    elif direction == 'TURN_LEFT':
      return cfg.ACTION_TURN_LEFT
    elif direction == 'TURN_RIGHT':
      return cfg.ACTION_TURN_RIGHT
    else:
      raise ValueError(f"Unknown patrol direction: {direction}")
