#!/usr/bin/env python3
"""Test new ResidentPolicy implementation.

Added by RST: Test for new ResidentPolicy-based implementation.

This test verifies:
1. Substrate loads with new observations
2. ResidentPolicy works correctly
3. Episode runs without errors
"""

import sys
from pathlib import Path

# Add meltingpot to path
sys.path.insert(0, str(Path(__file__).parent / "meltingpot"))

import numpy as np
import dm_env

from meltingpot.configs.substrates import allelopathic_harvest_normative
from agents.envs.resident_wrapper import ResidentWrapper
from agents.residents.resident_policy import ResidentPolicy


def test_resident_policy_basic():
  """Test that ResidentPolicy can be instantiated and step."""
  print("\n=== Test 1: ResidentPolicy Basic Functionality ===")

  policy = ResidentPolicy(seed=42)
  state = policy.initial_state()

  # Create dummy observation
  dummy_obs = {
      'ALTAR': np.array(1),  # Red
      'READY_TO_SHOOT': np.array(1.0),
      'AGENT_COLORS': np.zeros(16, dtype=np.int32),
      'IMMUNITY_STATUS': np.zeros(16, dtype=np.int32),
      'AVATAR_IDS_IN_RANGE_TO_ZAP': np.zeros(16, dtype=np.int32),
      'PLAYER_INDEX': np.array(1.0),
  }

  # Create dummy timestep
  timestep = dm_env.restart(dummy_obs)

  # Step policy
  action, new_state = policy.step(timestep, state)

  print(f"✓ Policy instantiated and stepped successfully")
  print(f"  Action: {action}")
  print(f"  State: {new_state}")

  policy.close()
  return True


def test_substrate_with_residents():
  """Test that substrate loads with new observations and residents work."""
  print("\n=== Test 2: Substrate with ResidentWrapper ===")

  # Create substrate config
  config = allelopathic_harvest_normative.get_config()

  # Build substrate
  print("Building substrate...")
  substrate = allelopathic_harvest_normative.build(config, roles=["default"] * 16)

  print("✓ Substrate built successfully")

  # Wrap with ResidentWrapper (all 16 are residents)
  print("Creating ResidentWrapper...")
  env = ResidentWrapper(
      env=substrate,
      resident_indices=list(range(16)),
      ego_index=None,
      seed=42)

  print("✓ ResidentWrapper created successfully")

  # Reset environment
  print("Resetting environment...")
  timestep = env.reset()

  print(f"✓ Environment reset successfully")
  print(f"  Timestep type: {timestep.step_type}")
  print(f"  Num observations: {len(timestep.observation)}")

  # Check observations for agent 0
  obs = timestep.observation[0]
  print(f"\n  Agent 0 observations:")
  for key in sorted(obs.keys()):
    value = obs[key]
    if isinstance(value, np.ndarray):
      print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    else:
      print(f"    {key}: {value}")

  # Step environment a few times
  print("\n  Stepping environment (10 steps)...")
  for i in range(10):
    timestep = env.step()
    if i == 0:
      print(f"    Step {i}: step_type={timestep.step_type}")

  print("✓ Environment stepped successfully")

  env.close()
  return True


def test_short_episode():
  """Test running a short episode with all residents."""
  print("\n=== Test 3: Short Episode (100 steps) ===")
  print("Running 100-step episode with all residents...")

  # Create substrate config
  config = allelopathic_harvest_normative.get_config()

  # Build substrate
  substrate = allelopathic_harvest_normative.build(config, roles=["default"] * 16)

  # Wrap with ResidentWrapper (all 16 are residents)
  env = ResidentWrapper(
      env=substrate,
      resident_indices=list(range(16)),
      ego_index=None,
      seed=42)

  # Reset environment
  timestep = env.reset()

  # Track altar color from observations
  altar_obs = timestep.observation[0].get('ALTAR')
  if isinstance(altar_obs, np.ndarray):
    altar_color = int(altar_obs.item() if altar_obs.size == 1 else altar_obs[0])
  else:
    altar_color = int(altar_obs)

  print(f"  Altar color: {altar_color} ({['', 'RED', 'GREEN', 'BLUE'][altar_color]})")

  # Run short episode (100 steps)
  episode_length = 100
  step_count = 0

  while not timestep.last() and step_count < episode_length:
    timestep = env.step()
    step_count += 1

    if step_count % 25 == 0:
      print(f"  Step {step_count}/{episode_length}")

  print(f"\n✓ Episode completed ({step_count} steps)")
  print("  Residents successfully executed policy for 100 steps")

  env.close()
  return True


def main():
  """Run all tests."""
  print("=" * 60)
  print("Testing New Resident Implementation")
  print("=" * 60)

  tests = [
      ("ResidentPolicy Basic", test_resident_policy_basic),
      ("Substrate with Residents", test_substrate_with_residents),
      ("Short Episode", test_short_episode),
  ]

  results = []
  for name, test_fn in tests:
    try:
      success = test_fn()
      results.append((name, success, None))
    except Exception as e:
      print(f"\n✗ Test failed with error: {e}")
      import traceback
      traceback.print_exc()
      results.append((name, False, str(e)))

  # Print summary
  print("\n" + "=" * 60)
  print("Test Summary")
  print("=" * 60)

  for name, success, error in results:
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status}: {name}")
    if error:
      print(f"      Error: {error}")

  # Overall result
  all_passed = all(success for _, success, _ in results)
  print("\n" + "=" * 60)
  if all_passed:
    print("✓ ALL TESTS PASSED")
  else:
    print("✗ SOME TESTS FAILED")
  print("=" * 60)

  return 0 if all_passed else 1


if __name__ == "__main__":
  sys.exit(main())
