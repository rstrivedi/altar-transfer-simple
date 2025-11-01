# Added by RST: Extract resident info from substrate state
"""Extract information needed by resident agents from environment state.

Residents need:
- world_step, permitted_color_index, startup_grey_grace (from config/tracking)
- Per-resident: pos, zap_cooldown_remaining, nearby_agents, berry info

Uses ResidentObserver component events for privileged information access.
"""

from typing import Dict, List
import numpy as np

from agents.residents import config as resident_config


class ResidentInfoExtractor:
  """Extracts resident-info from environment observations and events."""

  def __init__(self, num_players: int, permitted_color_index: int, startup_grey_grace: int):
    """Initialize the info extractor.

    Args:
      num_players: Total number of players in the environment.
      permitted_color_index: The permitted color (1=RED, 2=GREEN, 3=BLUE).
      startup_grey_grace: Grace period for grey agents (frames).
    """
    self._num_players = num_players
    self._permitted_color_index = permitted_color_index
    self._startup_grey_grace = startup_grey_grace

    # Track state across steps
    self._world_step = 0

  def reset(self):
    """Reset state for new episode."""
    self._world_step = 0

  def extract_info(self, observations: List[Dict], events: List[Dict]) -> Dict:
    """Extract resident info from observations and events.

    Args:
      observations: List of per-agent observations from env.observation().
      events: List of events from env.events().

    Returns:
      Dictionary with resident info for all agents.
    """
    # Build info for each resident
    info = {
        'world_step': self._world_step,
        'permitted_color_index': self._permitted_color_index,
        'startup_grey_grace': self._startup_grey_grace,
        'residents': {}
    }

    # Initialize resident info from observations
    for agent_id in range(self._num_players):
      obs = observations[agent_id]

      # Extract zap cooldown from READY_TO_SHOOT
      # READY_TO_SHOOT = 1.0 when ready, 0.0 when on cooldown
      ready_to_shoot_raw = obs.get('READY_TO_SHOOT', 1.0)
      # Handle both scalar and array cases
      ready_to_shoot = float(np.asarray(ready_to_shoot_raw).item())
      if ready_to_shoot > 0.5:
        zap_cooldown_remaining = 0
      else:
        # Not ready, approximate full cooldown
        zap_cooldown_remaining = resident_config.ZAP_COOLDOWN

      info['residents'][agent_id] = {
          'pos': (0, 0),  # Will be updated from events if available
          'orientation': 'N',  # Default to North
          'zap_cooldown_remaining': zap_cooldown_remaining,
          'nearby_agents': [],
          'nearby_ripe_berries': [],
          'nearby_unripe_berries': [],
          'has_ripe_berry_in_radius': False,
          'has_unripe_berry_in_range': False,
          'standing_on_unripe': False,
      }

    # Parse events from ResidentObserver
    self._parse_resident_observer_events(events, info)

    self._world_step += 1
    return info

  def _parse_resident_observer_events(self, events: List[Dict], info: Dict):
    """Parse resident_info, nearby_agent, nearby_ripe_berry, and nearby_unripe_berry events.

    Args:
      events: List of events from env.events().
      info: Info dictionary to update.
    """
    # Collect events by observer
    nearby_agents_by_observer = {}
    nearby_ripe_berries_by_observer = {}
    nearby_unripe_berries_by_observer = {}

    for event in events:
      event_name = event.get('name', '')

      if event_name == 'resident_info':
        # Main resident info event
        player_index = event.get('player_index')
        if player_index is None:
          continue

        # Convert from 1-indexed (Lua) to 0-indexed (Python)
        agent_id = player_index - 1
        if agent_id < 0 or agent_id >= self._num_players:
          continue

        # Update berry info
        info['residents'][agent_id]['has_ripe_berry_in_radius'] = bool(event.get('has_ripe_berry', 0))
        info['residents'][agent_id]['has_unripe_berry_in_range'] = bool(event.get('has_unripe_berry', 0))
        info['residents'][agent_id]['standing_on_unripe'] = bool(event.get('standing_on_unripe', 0))

        # Update orientation (N, E, S, W)
        orientation = event.get('self_orientation', 'N')
        if isinstance(orientation, bytes):
          orientation = orientation.decode('utf-8')
        info['residents'][agent_id]['orientation'] = orientation

      elif event_name == 'nearby_agent':
        # Nearby agent info event
        observer_index = event.get('observer_index')
        if observer_index is None:
          continue

        # Convert from 1-indexed (Lua) to 0-indexed (Python)
        observer_id = observer_index - 1
        if observer_id < 0 or observer_id >= self._num_players:
          continue

        agent_id = int(event.get('agent_id', 0)) - 1  # Convert to 0-indexed int
        rel_x = float(event.get('rel_x', 0))
        rel_y = float(event.get('rel_y', 0))
        body_color = int(event.get('body_color', 0))  # Convert to Python int
        immune_ticks = int(event.get('immune_ticks', 0))  # Convert to Python int

        # Store in temporary structure
        if observer_id not in nearby_agents_by_observer:
          nearby_agents_by_observer[observer_id] = []

        in_zap_range = bool(event.get('in_zap_range', 0))  # Convert to Python bool
        could_zap = bool(event.get('could_zap', 0))  # Could zap if facing this direction

        nearby_agents_by_observer[observer_id].append({
            'agent_id': agent_id,
            'rel_pos': (rel_x, rel_y),
            'body_color': body_color,
            'immune_ticks_remaining': immune_ticks,
            'in_zap_range': in_zap_range,
            'could_zap': could_zap
        })

      elif event_name == 'nearby_ripe_berry':
        # Nearby ripe berry event
        observer_index = event.get('observer_index')
        if observer_index is None:
          continue

        # Convert from 1-indexed (Lua) to 0-indexed (Python)
        observer_id = observer_index - 1
        if observer_id < 0 or observer_id >= self._num_players:
          continue

        rel_x = event.get('rel_x', 0)
        rel_y = event.get('rel_y', 0)
        distance = event.get('distance', 0)
        color_id = event.get('color_id', 0)

        # Store in temporary structure
        if observer_id not in nearby_ripe_berries_by_observer:
          nearby_ripe_berries_by_observer[observer_id] = []

        nearby_ripe_berries_by_observer[observer_id].append({
            'rel_pos': (rel_x, rel_y),
            'distance': distance,
            'color_id': color_id
        })

      elif event_name == 'nearby_unripe_berry':
        # Nearby unripe berry event
        observer_index = event.get('observer_index')
        if observer_index is None:
          continue

        # Convert from 1-indexed (Lua) to 0-indexed (Python)
        observer_id = observer_index - 1
        if observer_id < 0 or observer_id >= self._num_players:
          continue

        rel_x = event.get('rel_x', 0)
        rel_y = event.get('rel_y', 0)
        distance = event.get('distance', 0)
        color_id = event.get('color_id', 0)

        # Store in temporary structure
        if observer_id not in nearby_unripe_berries_by_observer:
          nearby_unripe_berries_by_observer[observer_id] = []

        nearby_unripe_berries_by_observer[observer_id].append({
            'rel_pos': (rel_x, rel_y),
            'distance': distance,
            'color_id': color_id
        })

    # Update nearby_agents in info
    for observer_id, nearby_agents in nearby_agents_by_observer.items():
      if observer_id in info['residents']:
        info['residents'][observer_id]['nearby_agents'] = nearby_agents

    # Update nearby_ripe_berries in info
    for observer_id, nearby_ripe_berries in nearby_ripe_berries_by_observer.items():
      if observer_id in info['residents']:
        info['residents'][observer_id]['nearby_ripe_berries'] = nearby_ripe_berries

    # Update nearby_unripe_berries in info
    for observer_id, nearby_unripe_berries in nearby_unripe_berries_by_observer.items():
      if observer_id in info['residents']:
        info['residents'][observer_id]['nearby_unripe_berries'] = nearby_unripe_berries
