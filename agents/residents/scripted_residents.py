# Added by RST: Scripted resident agents for normative allelopathic harvest
"""Deterministic scripted resident controller implementing equilibrium play.

Residents enforce the posted rule, harvest/replant permitted color, and patrol.
Policy priority: P1 (zap) > P2 (harvest) > P3 (plant) > P4 (patrol)

Enforcement is HIGHEST priority - residents interrupt harvest/plant to zap violators.
Zapping strategy (hybrid approach):
- If violator in zap range → fire immediately
- If violator could be zapped (geometrically possible) → turn toward them
- This balances responsiveness (turn toward targets) with speed (only 1 turn/step)
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
    self._last_plant_step = {}  # Per-resident last plant step for frequency control
    self._step_count = 0  # Global step counter

  def reset(self, seed: int = cfg.DEFAULT_SEED):
    """Reset controller for new episode.

    Args:
      seed: Random seed for patrol behavior.
    """
    self._rng = np.random.RandomState(seed)
    self._patrol_state = {}
    self._last_plant_step = {}
    self._step_count = 0

  def act(self, resident_id: int, info: Dict) -> int:
    """Select action for a resident agent.

    Args:
      resident_id: Agent ID (0-indexed).
      info: Information dict from ResidentInfoExtractor.

    Returns:
      Action index (0-10).
    """
    self._step_count += 1

    # Get resident-specific info
    resident_info = info['residents'][resident_id]
    world_step = info['world_step']
    permitted_color_index = info['permitted_color_index']
    startup_grey_grace = info['startup_grey_grace']

    # P1: Enforce (Zap violators - HIGHEST PRIORITY)
    zap_action = self._try_zap(
        resident_info, permitted_color_index, world_step, startup_grey_grace)
    if zap_action is not None:
      return zap_action

    # P2: Harvest (move toward ripe berry)
    harvest_action = self._try_harvest(resident_info, resident_id)
    if harvest_action is not None:
      return harvest_action

    # P3: Replant permitted color (with frequency control)
    plant_action = self._try_plant(resident_info, permitted_color_index, resident_id)
    if plant_action is not None:
      return plant_action

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
    """P3: Try to zap eligible violators.

    Strategy: If violator in zap range → fire. If violator could be zapped
    (geometrically possible) → turn toward them. This is faster than original
    full pursuit but more active than pure opportunistic.

    Args:
      resident_info: Resident's info dict.
      permitted: Permitted color index.
      world_step: Current world step.
      grace: Grace period for grey agents.

    Returns:
      ACTION_FIRE_ZAP or turn action if targeting, None otherwise.
    """
    # Check if zap is off cooldown
    if resident_info['zap_cooldown_remaining'] > 0:
      return None

    # Get nearby agents
    nearby_agents = resident_info['nearby_agents']


    # First pass: check if anyone is in zap range RIGHT NOW
    for agent in nearby_agents:
      if agent.get('in_zap_range', False):
        is_elig = self._is_eligible(agent, permitted, world_step, grace)
        if is_elig:
          return cfg.ACTION_FIRE_ZAP

    # Second pass: turn toward nearest zappable violator
    eligible_zappable = [
        agent for agent in nearby_agents
        if agent.get('could_zap', False) and self._is_eligible(agent, permitted, world_step, grace)
    ]

    if not eligible_zappable:
      return None

    # Select nearest (tie-break: lowest agent_id)
    nearest = min(eligible_zappable, key=lambda a: (
        math.sqrt(a['rel_pos'][0]**2 + a['rel_pos'][1]**2),
        a['agent_id']
    ))


    # Turn toward them (one step)
    return self._turn_toward(nearest['rel_pos'], resident_info['orientation'])

  def _try_plant(self, resident_info: Dict, permitted: int, resident_id: int) -> Optional[int]:
    """P2: Try to plant permitted color on nearby unripe berries.

    Frequency control: Only attempts planting every PLANT_FREQUENCY steps.
    On non-plant steps, returns None to allow fall-through to zapping.

    Args:
      resident_info: Resident's info dict.
      permitted: Permitted color index.
      resident_id: Agent ID for tracking last plant step.

    Returns:
      Plant action or movement action on plant steps, None on non-plant steps.
    """
    # Check frequency control FIRST - return None on non-plant steps
    last_plant = self._last_plant_step.get(resident_id, -999)
    if self._step_count - last_plant < cfg.PLANT_FREQUENCY:
      # Not time to plant yet - return None to allow fall-through to zapping
      return None

    # On plant steps, try to plant or move toward berries
    nearby_unripe_berries = resident_info.get('nearby_unripe_berries', [])

    if not nearby_unripe_berries:
      return None

    # Select nearest unripe berry
    nearest = min(nearby_unripe_berries, key=lambda b: b['distance'])

    # If close enough to plant (beam length = 3)
    if nearest['distance'] < 3.0:
      self._last_plant_step[resident_id] = self._step_count
      return cfg.PLANT_ACTION_MAP[permitted]

    # Move forward (simple movement, orientation changes through patrol)
    return cfg.ACTION_FORWARD

  def _try_harvest(self, resident_info: Dict, resident_id: int) -> Optional[int]:
    """P1: Try to move toward nearest ripe berry.

    No frequency control - always moves toward ripe berries if visible.
    This creates constant movement that brings residents into contact with violators.

    Args:
      resident_info: Resident's info dict.
      resident_id: Agent ID (unused, kept for API consistency).

    Returns:
      Movement action if ripe berries nearby, None otherwise.
    """
    nearby_berries = resident_info.get('nearby_ripe_berries', [])

    if not nearby_berries:
      return None

    # Always move toward nearest ripe berry (harvesting happens automatically)
    # Simple forward movement - orientation changes through patrol/turns
    return cfg.ACTION_FORWARD

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

  def _turn_toward(self, rel_pos: Tuple[float, float], orientation: str) -> int:
    """Compute turn action to face toward target.

    Args:
      rel_pos: Relative position (rel_x, rel_y) to target.
      orientation: Current facing direction ('N', 'E', 'S', 'W').

    Returns:
      TURN_LEFT or TURN_RIGHT action.
    """
    rel_x, rel_y = rel_pos
    abs_x = abs(rel_x)
    abs_y = abs(rel_y)

    # Determine desired direction based on largest displacement
    if abs_x > abs_y:
      desired_dir = 'S' if rel_x > 0 else 'N'
    else:
      desired_dir = 'E' if rel_y > 0 else 'W'

    # Already facing desired direction - just return a turn (shouldn't happen in _try_zap logic)
    if orientation == desired_dir:
      return cfg.ACTION_TURN_RIGHT

    # Compute shortest turn direction
    dirs = ['N', 'E', 'S', 'W']
    current_idx = dirs.index(orientation)
    desired_idx = dirs.index(desired_dir)

    cw_dist = (desired_idx - current_idx) % 4
    ccw_dist = (current_idx - desired_idx) % 4

    return cfg.ACTION_TURN_RIGHT if cw_dist <= ccw_dist else cfg.ACTION_TURN_LEFT

