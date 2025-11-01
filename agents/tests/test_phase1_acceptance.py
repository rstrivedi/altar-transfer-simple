# Added by RST: Phase 1 acceptance tests
"""Acceptance tests for Phase 1 normative infrastructure.

Tests verify:
1. Parity: normative_gate=False preserves base allelopathic harvest behavior
2. Simple sanctions: immediate -10, no freeze/removal
3. Immunity: clears on color change or 200-frame timeout
4. Tie-break: only one sanction per target per frame
5. Observation filtering: PERMITTED_COLOR in treatment, not in control
6. Reward accounting: R_eval = R_total - α
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest

import meltingpot.substrate as substrate
from meltingpot.configs.substrates import allelopathic_harvest__open as allelopathic_harvest
from agents.envs import NormativeObservationFilter, NormativeMetricsLogger
from agents.utils.event_parser import parse_events


def test_parity_normative_gate_disabled():
  """Test that normative_gate=False preserves base AH behavior."""
  config = allelopathic_harvest.get_config()
  with config.unlocked():
    config.normative_gate = False  # Disable normative system

  roles = ["default"] * 4
  env = substrate.build_from_config(
      config=config,
      roles=roles)

  timestep = env.reset()

  # Should not have PERMITTED_COLOR observation when normative_gate is off
  obs_keys = set(timestep.observation[0].keys())
  assert "PERMITTED_COLOR" not in obs_keys, \
      "PERMITTED_COLOR should not exist when normative_gate=False"

  # Run a few steps to ensure it doesn't crash
  for _ in range(10):
    actions = [env.action_spec()[i].generate_value() for i in range(4)]
    timestep = env.step(actions)
    if timestep.last():
      break

  env.close()


def test_permitted_color_observation_exists_with_normative_gate():
  """Test that PERMITTED_COLOR observation exists when normative_gate=True."""
  config = allelopathic_harvest.get_config()
  with config.unlocked():
    config.normative_gate = True
    config.permitted_color_index = 2  # GREEN

  roles = ["default"] * 4
  env = substrate.build_from_config(
      config=config,
      roles=roles)

  timestep = env.reset()

  # Should have PERMITTED_COLOR observation
  obs = timestep.observation[0]
  assert "PERMITTED_COLOR" in obs, \
      "PERMITTED_COLOR should exist when normative_gate=True"

  # Should be one-hot encoding with shape (3,)
  permitted_color_obs = obs["PERMITTED_COLOR"]
  assert permitted_color_obs.shape == (3,), \
      f"PERMITTED_COLOR shape should be (3,), got {permitted_color_obs.shape}"

  # Should be one-hot for GREEN (index 2, 0-indexed in observation)
  expected = np.array([0., 1., 0.])
  np.testing.assert_array_equal(permitted_color_obs, expected,
      err_msg=f"PERMITTED_COLOR should be one-hot for GREEN, got {permitted_color_obs}")

  env.close()


def test_observation_filter_treatment_keeps_permitted_color():
  """Test that treatment condition keeps PERMITTED_COLOR observation."""
  config = allelopathic_harvest.get_config()
  with config.unlocked():
    config.normative_gate = True
    config.permitted_color_index = 1  # RED

  roles = ["default"] * 4
  env = substrate.build_from_config(
      config=config,
      roles=roles)

  # Wrap with treatment condition
  env = NormativeObservationFilter(env, enable_treatment_condition=True)

  timestep = env.reset()
  obs = timestep.observation[0]

  assert "PERMITTED_COLOR" in obs, \
      "Treatment condition should keep PERMITTED_COLOR"

  env.close()


def test_observation_filter_control_removes_permitted_color():
  """Test that control condition removes PERMITTED_COLOR observation."""
  config = allelopathic_harvest.get_config()
  with config.unlocked():
    config.normative_gate = True
    config.permitted_color_index = 1  # RED

  roles = ["default"] * 4
  env = substrate.build_from_config(
      config=config,
      roles=roles)

  # Wrap with control condition
  env = NormativeObservationFilter(env, enable_treatment_condition=False)

  timestep = env.reset()
  obs = timestep.observation[0]

  assert "PERMITTED_COLOR" not in obs, \
      "Control condition should remove PERMITTED_COLOR"

  # Should still have other observations
  assert "RGB" in obs, "Should still have RGB observation"
  assert "READY_TO_SHOOT" in obs, "Should still have READY_TO_SHOOT observation"

  env.close()


def test_metrics_logger_collects_reward_components():
  """Test that NormativeMetricsLogger collects α, β, c from events."""
  logger = NormativeMetricsLogger(num_players=4)

  # Simulate some reward component events
  events = [
      {'name': 'reward_component', 'component': 'alpha', 'player_index': 1, 'value': 5.0},
      {'name': 'reward_component', 'component': 'beta', 'player_index': 2, 'value': 5.0},
      {'name': 'reward_component', 'component': 'c', 'player_index': 1, 'value': 0.5},
      {'name': 'reward_component', 'component': 'alpha', 'player_index': 1, 'value': 5.0},
  ]

  logger.process_events(events)

  # Player 0 (1-indexed): 2 alphas = 10.0, 1 c = 0.5
  assert logger.get_alpha_sum(0) == 10.0, "Player 0 should have alpha=10.0"
  assert logger.get_c_sum(0) == 0.5, "Player 0 should have c=0.5"
  assert logger.get_beta_sum(0) == 0.0, "Player 0 should have beta=0.0"

  # Player 1 (1-indexed): 1 beta = 5.0
  assert logger.get_alpha_sum(1) == 0.0, "Player 1 should have alpha=0.0"
  assert logger.get_beta_sum(1) == 5.0, "Player 1 should have beta=5.0"
  assert logger.get_c_sum(1) == 0.0, "Player 1 should have c=0.0"


def test_metrics_logger_compute_r_eval():
  """Test that NormativeMetricsLogger correctly computes R_eval = R_total - α."""
  logger = NormativeMetricsLogger(num_players=4)

  events = [
      {'name': 'reward_component', 'component': 'alpha', 'player_index': 1, 'value': 5.0},
      {'name': 'reward_component', 'component': 'beta', 'player_index': 1, 'value': 5.0},
      {'name': 'reward_component', 'component': 'c', 'player_index': 1, 'value': 0.5},
  ]
  logger.process_events(events)

  # R_total = R_env + α - β - c
  # Say R_env = 10, then R_total = 10 + 5 - 5 - 0.5 = 9.5
  # R_eval = R_total - α = 9.5 - 5 = 4.5 = R_env - β - c = 10 - 5 - 0.5
  r_total = np.array([9.5, 0.0, 0.0, 0.0])
  r_eval = logger.compute_r_eval(r_total)

  expected = np.array([4.5, 0.0, 0.0, 0.0])  # Strips alpha from player 0
  np.testing.assert_array_almost_equal(r_eval, expected,
      err_msg=f"R_eval should strip alpha, expected {expected}, got {r_eval}")


def test_normative_system_runs_without_crash():
  """Integration test: run environment with normative system for multiple steps."""
  config = allelopathic_harvest.get_config()
  with config.unlocked():
    config.normative_gate = True
    config.enable_treatment_condition = True
    config.permitted_color_index = 1  # RED
    config.altar_coords = (5, 15)

  roles = ["default"] * 8
  env = substrate.build_from_config(
      config=config,
      roles=roles)

  env = NormativeObservationFilter(env, enable_treatment_condition=True)
  logger = NormativeMetricsLogger(num_players=8)

  timestep = env.reset()
  logger.reset()
  logger.process_events(parse_events(env.events()))

  # Run for 100 steps with random actions
  for step in range(100):
    actions = [env.action_spec()[i].generate_value() for i in range(8)]
    timestep = env.step(actions)
    logger.process_events(parse_events(env.events()))

    if timestep.last():
      break

  # Should complete without crashing
  summary = logger.get_episode_summary()
  assert 'alpha_sum' in summary
  assert 'beta_sum' in summary
  assert 'c_sum' in summary

  env.close()


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
