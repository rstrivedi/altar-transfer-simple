# Added by RST: Logging infrastructure for normative reward components
"""Logger for tracking normative reward components (α, β, c) and computing R_eval.

The NormativeRewardTracker in Lua emits 'reward_component' events with:
- component: 'alpha', 'beta', or 'c'
- player_index: which player (1-indexed in Lua)
- value: amount added/subtracted

This logger collects those events and computes:
- R_env: raw environment reward (eating, getting sanctioned, etc.)
- α (alpha): training bonus for correct zaps (stripped at eval)
- β (beta): penalty for mis-zaps
- c: effort cost per zap
- R_eval = R_env - β - c (excludes α)
"""

from typing import Dict, List, Optional

import numpy as np


class NormativeMetricsLogger:
  """Logger for tracking normative reward components per episode."""

  def __init__(self, num_players: int):
    """Initialize the logger.

    Args:
      num_players: Number of players in the environment.
    """
    self._num_players = num_players
    self.reset()

  def reset(self):
    """Reset all counters for a new episode."""
    # Track cumulative sums per player
    self._alpha_sum = np.zeros(self._num_players, dtype=np.float64)
    self._beta_sum = np.zeros(self._num_players, dtype=np.float64)
    self._c_sum = np.zeros(self._num_players, dtype=np.float64)

    # Track event history (for debugging/analysis)
    self._alpha_events = [[] for _ in range(self._num_players)]
    self._beta_events = [[] for _ in range(self._num_players)]
    self._c_events = [[] for _ in range(self._num_players)]

  def log_reward_component(self, component: str, player_index: int, value: float):
    """Log a reward component event.

    Args:
      component: 'alpha', 'beta', or 'c'
      player_index: Player index (1-indexed from Lua)
      value: Amount of reward component
    """
    # Convert from 1-indexed (Lua) to 0-indexed (Python)
    py_idx = player_index - 1

    if py_idx < 0 or py_idx >= self._num_players:
      raise ValueError(f"Invalid player_index: {player_index} (num_players={self._num_players})")

    if component == 'alpha':
      self._alpha_sum[py_idx] += value
      self._alpha_events[py_idx].append(value)
    elif component == 'beta':
      self._beta_sum[py_idx] += value
      self._beta_events[py_idx].append(value)
    elif component == 'c':
      self._c_sum[py_idx] += value
      self._c_events[py_idx].append(value)
    else:
      raise ValueError(f"Unknown reward component: {component}")

  def process_events(self, events: List[Dict]):
    """Process a list of events from the environment.

    Args:
      events: List of event dictionaries from env.events()
    """
    for event in events:
      if event.get('name') == 'reward_component':
        component = event.get('component')
        player_index = event.get('player_index')
        value = event.get('value')

        if component and player_index is not None and value is not None:
          self.log_reward_component(component, player_index, value)

  def get_alpha_sum(self, player_idx: Optional[int] = None):
    """Get cumulative alpha (correct zap bonus) for player(s).

    Args:
      player_idx: If specified, return for that player (0-indexed).
        If None, return array for all players.

    Returns:
      Alpha sum(s).
    """
    if player_idx is not None:
      return self._alpha_sum[player_idx]
    return self._alpha_sum.copy()

  def get_beta_sum(self, player_idx: Optional[int] = None):
    """Get cumulative beta (mis-zap penalty) for player(s).

    Args:
      player_idx: If specified, return for that player (0-indexed).
        If None, return array for all players.

    Returns:
      Beta sum(s).
    """
    if player_idx is not None:
      return self._beta_sum[player_idx]
    return self._beta_sum.copy()

  def get_c_sum(self, player_idx: Optional[int] = None):
    """Get cumulative c (effort cost) for player(s).

    Args:
      player_idx: If specified, return for that player (0-indexed).
        If None, return array for all players.

    Returns:
      C sum(s).
    """
    if player_idx is not None:
      return self._c_sum[player_idx]
    return self._c_sum.copy()

  def compute_r_eval(self, r_total: np.ndarray) -> np.ndarray:
    """Compute R_eval = R_total - alpha (strip training bonus).

    Args:
      r_total: Total reward from environment (includes alpha, beta, c).
        Shape: (num_players,)

    Returns:
      R_eval for each player (training bonus α stripped out).
        R_eval = R_total - α = (R_env + α - β - c) - α = R_env - β - c
    """
    return r_total - self._alpha_sum

  def get_episode_summary(self) -> Dict:
    """Get summary of reward components for the episode.

    Returns:
      Dictionary with per-player and aggregate statistics.
    """
    return {
        'alpha_sum': self._alpha_sum.copy(),
        'beta_sum': self._beta_sum.copy(),
        'c_sum': self._c_sum.copy(),
        'alpha_total': np.sum(self._alpha_sum),
        'beta_total': np.sum(self._beta_sum),
        'c_total': np.sum(self._c_sum),
        'alpha_events_count': [len(events) for events in self._alpha_events],
        'beta_events_count': [len(events) for events in self._beta_events],
        'c_events_count': [len(events) for events in self._c_events],
    }
