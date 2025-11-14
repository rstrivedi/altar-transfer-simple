# Added by RST: Real-time telemetry capture for Phase 3 metrics
"""MetricsRecorder for capturing per-step telemetry during episode rollout.

This module provides real-time event and state capture from the environment wrapper.
It builds StepMetrics buffers that aggregators.py can process into EpisodeMetrics.

Usage:
  recorder = MetricsRecorder(num_players=16, ego_index=0, permitted_color_index=1)
  recorder.reset()

  for step in range(episode_len):
    timestep = env.step(actions)
    events = env.events()
    recorder.record_step(step, timestep, events, ego_action)

  episode_metrics = recorder.get_episode_metrics(resident_baseline_r_eval, resident_baseline_sanctions)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from agents.metrics.schema import (
    StepMetrics,
    EpisodeMetrics,
    SanctionEvent,
    PlantEvent,
    EatEvent,
)


class MetricsRecorder:
  """Real-time telemetry recorder for episode rollout.

  Captures per-step events and observations, builds StepMetrics buffer.
  NO aggregation happens here - just data collection.
  """

  def __init__(
      self,
      num_players: int,
      ego_index: int,
      permitted_color_index: int,
      startup_grey_grace: int = 25,
      community_tag: Optional[str] = None,
      community_idx: Optional[int] = None,
  ):
    """Initialize metrics recorder.

    Args:
      num_players: Total number of agents in environment.
      ego_index: Index of ego agent (0-indexed).
      permitted_color_index: Permitted color (1=RED, 2=GREEN, 3=BLUE).
      startup_grey_grace: Grace period for GREY (frames).
      community_tag: Community name ('RED'/'GREEN'/'BLUE') for Phase 5.
      community_idx: Community index (1/2/3) for Phase 5.
    """
    self._num_players = num_players
    self._ego_index = ego_index
    self._permitted_color_index = permitted_color_index
    self._startup_grey_grace = startup_grey_grace

    # Phase 5: Multi-community tracking
    self._community_tag = community_tag
    self._community_idx = community_idx

    # Per-step buffer
    self._step_metrics: List[StepMetrics] = []

    # Cumulative reward component trackers (per player)
    self._alpha_sum = np.zeros(num_players, dtype=np.float64)
    self._beta_sum = np.zeros(num_players, dtype=np.float64)
    self._c_sum = np.zeros(num_players, dtype=np.float64)

    # Cumulative env reward (r_total from timestep, includes alpha/beta/c)
    self._r_total_sum = np.zeros(num_players, dtype=np.float64)

    # Track ego body color (from resident_info events)
    self._ego_body_color = 0  # Start GREY

  def reset(self):
    """Reset for new episode."""
    self._step_metrics = []
    self._alpha_sum.fill(0)
    self._beta_sum.fill(0)
    self._c_sum.fill(0)
    self._r_total_sum.fill(0)
    self._ego_body_color = 0  # Reset to GREY

  def record_step(
      self,
      t: int,
      timestep,
      events: List[Dict],
      ego_action: int,
  ):
    """Record telemetry for one step.

    Args:
      t: Frame number (0-indexed).
      timestep: dm_env.TimeStep from env.step().
      events: List of event dicts from env.events().
      ego_action: Action taken by ego agent (0-10).
    """
    # === Parse events ===
    sanctions = []
    plants = []
    eats = []

    # Reward component deltas this step (per player)
    alpha_step = np.zeros(self._num_players, dtype=np.float64)
    beta_step = np.zeros(self._num_players, dtype=np.float64)
    c_step = np.zeros(self._num_players, dtype=np.float64)

    # # Debug: Print all event types and violation status at t=100, 200, 300
    # if t in [100, 200, 300]:
    #   event_names = set([event.get('name', 'UNKNOWN') for event in events])
    #   print(f"DEBUG t={t}: Event types: {event_names}")
    #   print(f"DEBUG t={t}: Ego body color: {self._ego_body_color}, Permitted: {self._permitted_color_index}")
    #   print(f"DEBUG t={t}: Ego violating: {self._ego_body_color != self._permitted_color_index and self._ego_body_color != 0}")
    #   print(f"DEBUG t={t}: In grace period: {t < self._startup_grey_grace}")

    # # Debug: Track zap actions
    # if ego_action == 7:  # FIRE_ZAP
    #   print(f"DEBUG: Ego fired ZAP at t={t}, ego_color={self._ego_body_color}")

    # # Debug: Count sanction events at t=100, 200, 300
    # if t in [100, 200, 300]:
    #   sanction_count = sum(1 for e in events if e.get('name') == 'sanction')
    #   print(f"DEBUG t={t}: Sanction events this step: {sanction_count}")

    for event in events:
      event_name = event.get('name', '')

      # # Debug sanction events (print first 3 sanctions ever to verify parsing)
      # if event_name == 'sanction':
      #   total_sanctions_so_far = sum(len(sm.sanctions) for sm in self._step_metrics)
      #   if total_sanctions_so_far < 3:
      #     print(f"DEBUG: Sanction event #{total_sanctions_so_far} at t={t}: {event}")

      # === reward_component events ===
      if event_name == 'reward_component':
        player_id_lua = event.get('player_id')  # 1-indexed
        reward_type = event.get('type')  # 'alpha', 'beta', or 'c'
        value = event.get('value', 0.0)

        if player_id_lua is None or reward_type is None:
          continue

        player_id = player_id_lua - 1  # Convert to 0-indexed

        if reward_type == 'alpha':
          alpha_step[player_id] += value
          self._alpha_sum[player_id] += value
        elif reward_type == 'beta':
          beta_step[player_id] += value
          self._beta_sum[player_id] += value
        elif reward_type == 'c':
          c_step[player_id] += value
          self._c_sum[player_id] += value

      # === sanction events ===
      elif event_name == 'sanction':
        zapper_id_lua = event.get('zapper_id')  # 1-indexed
        zappee_id_lua = event.get('zappee_id')  # 1-indexed
        zappee_color = event.get('zappee_color', 0)
        # Lua sends booleans as 1/0, convert to Python bool
        was_violation = bool(event.get('was_violation', 0))
        applied_minus10 = bool(event.get('applied_minus10', 0))
        immune = bool(event.get('immune', 0))
        tie_break = bool(event.get('tie_break', 0))

        if zapper_id_lua is None or zappee_id_lua is None:
          continue

        sanction = SanctionEvent(
            t=t,
            zapper_id=zapper_id_lua - 1,  # Convert to 0-indexed
            zappee_id=zappee_id_lua - 1,
            zappee_color=zappee_color,
            was_violation=was_violation,
            applied_minus10=applied_minus10,
            immune=immune,
            tie_break=tie_break,
        )
        sanctions.append(sanction)

      # === replanting events ===
      elif event_name == 'replanting':
        player_idx_lua = event.get('player_index')  # 1-indexed
        source_berry = event.get('source_berry', 0)
        target_berry = event.get('target_berry', 0)

        if player_idx_lua is None:
          continue

        plant = PlantEvent(
            player_index=player_idx_lua - 1,  # Convert to 0-indexed
            source_berry=source_berry,
            target_berry=target_berry,
        )
        plants.append(plant)

      # === eating events ===
      elif event_name == 'eating':
        player_idx_lua = event.get('player_index')  # 1-indexed
        berry_color = event.get('berry_color', 0)

        if player_idx_lua is None:
          continue

        eat = EatEvent(
            player_index=player_idx_lua - 1,  # Convert to 0-indexed
            berry_color=berry_color,
        )
        eats.append(eat)

      # === resident_info events (to get ego_body_color) ===
      elif event_name == 'resident_info':
        player_idx_lua = event.get('player_index')  # 1-indexed
        self_body_color = event.get('self_body_color', 0)

        if player_idx_lua is None:
          continue

        player_idx = player_idx_lua - 1

        # If this is ego's resident_info event, update ego body color
        if player_idx == self._ego_index:
          self._ego_body_color = self_body_color

    # === Reconstruct alpha/beta/c from sanction events ===
    # Added by RST: Lua doesn't emit reward_component events, so we reconstruct
    # alpha/beta/c from sanction events. We know:
    # - If applied_minus10 == True: rewards were applied
    # - Zapper gets: c=-0.5, alpha=+5.0 (if violation) OR beta=-5.0 (if not)
    # - Zappee gets: -10.0 penalty (already in timestep.reward)

    # Handle ego agent sanctions (keep separate for clarity)
    for sanction in sanctions:
      # Only count sanctions that actually applied rewards
      if not sanction.applied_minus10:
        continue  # Fizzled (grace period/immune/tie-break), no rewards applied

      # Check if ego was the zapper (convert to int for comparison)
      if int(sanction.zapper_id) == self._ego_index:
        # Ego paid zap cost
        c_step[self._ego_index] += -0.5
        self._c_sum[self._ego_index] += -0.5

        # Ego got alpha (correct sanction) or beta (incorrect sanction)
        if sanction.was_violation:
          # Correct sanction: alpha bonus
          alpha_step[self._ego_index] += 5.0
          self._alpha_sum[self._ego_index] += 5.0
        else:
          # Incorrect sanction: beta penalty
          beta_step[self._ego_index] += -5.0
          self._beta_sum[self._ego_index] += -5.0

    # Handle other agents' sanctions (for collective reward computation)
    for sanction in sanctions:
      # Only count sanctions that actually applied rewards
      if not sanction.applied_minus10:
        continue

      zapper_idx = int(sanction.zapper_id)  # Convert to int for numpy indexing

      # Skip ego (already handled above)
      if zapper_idx == self._ego_index:
        continue

      # Other agent paid zap cost
      c_step[zapper_idx] += -0.5
      self._c_sum[zapper_idx] += -0.5

      # Other agent got alpha or beta
      if sanction.was_violation:
        alpha_step[zapper_idx] += 5.0
        self._alpha_sum[zapper_idx] += 5.0
      else:
        beta_step[zapper_idx] += -5.0
        self._beta_sum[zapper_idx] += -5.0

    # === Get timestep rewards ===
    # timestep.reward is r_total = r_env + alpha - beta - c (array for all agents)
    # Added by RST: Track ALL agents' rewards for collective reward calculation
    for player_idx in range(self._num_players):
      self._r_total_sum[player_idx] += timestep.reward[player_idx]

    # Get ego's reward for this step
    r_total_step = timestep.reward[self._ego_index]

    # Compute r_env_step = r_total_step - alpha_step + beta_step + c_step
    # Because: r_total = r_env + alpha - beta - c
    # So: r_env = r_total - alpha + beta + c
    r_env_step = (r_total_step - alpha_step[self._ego_index] +
                  beta_step[self._ego_index] + c_step[self._ego_index])

    # === Get berry counts from observations ===
    # BERRIES_BY_TYPE is a global observation, should be in ego's observation
    ego_obs = timestep.observation[self._ego_index]
    berry_counts_array = ego_obs.get('BERRIES_BY_TYPE', np.array([0, 0, 0]))
    berry_counts = tuple(int(x) for x in berry_counts_array)  # (red, green, blue)

    # === Ego zap tracking ===
    ego_zap_attempted = (ego_action == 7)  # FIRE_ZAP action

    # === Build StepMetrics ===
    step_metric = StepMetrics(
        t=t,
        r_env=r_env_step,
        alpha=alpha_step[self._ego_index],
        beta=beta_step[self._ego_index],
        c=c_step[self._ego_index],
        ego_body_color=self._ego_body_color,
        ego_action=ego_action,
        ego_zap_attempted=ego_zap_attempted,
        sanctions=sanctions,
        plants=plants,
        eats=eats,
        permitted_color_index=self._permitted_color_index,
        berry_counts=berry_counts,
        community_tag=self._community_tag,
        community_idx=self._community_idx,
    )

    self._step_metrics.append(step_metric)

  def get_step_metrics(self) -> List[StepMetrics]:
    """Get raw per-step metrics buffer.

    Returns:
      List of StepMetrics for all steps so far.
    """
    return self._step_metrics

  def get_cumulative_sums(self) -> Dict[str, np.ndarray]:
    """Get cumulative reward component sums (for debugging).

    Returns:
      Dict with 'alpha_sum', 'beta_sum', 'c_sum', 'r_total_sum' arrays.
    """
    return {
        'alpha_sum': self._alpha_sum.copy(),
        'beta_sum': self._beta_sum.copy(),
        'c_sum': self._c_sum.copy(),
        'r_total_sum': self._r_total_sum.copy(),
    }

  def get_r_eval(self) -> float:
    """Get ego's R_eval = R_env - beta - c (strips alpha).

    Returns:
      R_eval for ego agent.
    """
    # R_total = R_env + alpha - beta - c
    # R_eval = R_env - beta - c = R_total - alpha
    return self._r_total_sum[self._ego_index] - self._alpha_sum[self._ego_index]

  def get_ego_body_color(self) -> int:
    """Get ego's current body color (0=GREY, 1=RED, 2=GREEN, 3=BLUE).

    Returns:
      Ego body color (from most recent resident_info event).
    """
    return self._ego_body_color

  def get_episode_summary(self) -> Dict[str, float]:
    """Get episode summary metrics for logging.

    Returns:
      Dict with episode-level metrics.
    """
    # Count berry plants/eats and sanctions from step metrics
    berries_planted_red = 0
    berries_planted_green = 0
    berries_planted_blue = 0
    berries_consumed_red = 0
    berries_consumed_green = 0
    berries_consumed_blue = 0
    times_sanctioned = 0
    times_sanctioned_others = 0

    for step in self._step_metrics:
      # Berry plants by ego (PlantEvent has player_index, target_berry)
      for plant in step.plants:
        if plant.player_index == self._ego_index:
          color = plant.target_berry
          if color == 1:
            berries_planted_red += 1
          elif color == 2:
            berries_planted_green += 1
          elif color == 3:
            berries_planted_blue += 1

      # Berry consumption by ego (EatEvent has player_index, berry_color)
      for eat in step.eats:
        if eat.player_index == self._ego_index:
          color = eat.berry_color
          if color == 1:
            berries_consumed_red += 1
          elif color == 2:
            berries_consumed_green += 1
          elif color == 3:
            berries_consumed_blue += 1

      # Sanctions (SanctionEvent has zapper_id, zappee_id)
      for sanction in step.sanctions:
        if sanction.zappee_id == self._ego_index:
          times_sanctioned += 1
        if sanction.zapper_id == self._ego_index:
          times_sanctioned_others += 1

    # Episode totals
    r_total = self._r_total_sum[self._ego_index]
    r_eval = self.get_r_eval()
    # Collective reward = sum of natural rewards (r_total - alpha) for ALL agents
    # Alpha is a training bonus, not part of actual social welfare
    collective_reward = (self._r_total_sum - self._alpha_sum).sum()
    alpha_total = self._alpha_sum[self._ego_index]

    return {
      'r': float(r_total),
      'l': len(self._step_metrics),
      'r_eval': float(r_eval),
      'collective_reward': float(collective_reward),
      'alpha': float(alpha_total),  # Added by RST: For displaying r - alpha
      'berries_planted_red': berries_planted_red,
      'berries_planted_green': berries_planted_green,
      'berries_planted_blue': berries_planted_blue,
      'berries_consumed_red': berries_consumed_red,
      'berries_consumed_green': berries_consumed_green,
      'berries_consumed_blue': berries_consumed_blue,
      'times_sanctioned': times_sanctioned,
      'times_sanctioned_others': times_sanctioned_others,
    }
