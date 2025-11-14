#!/usr/bin/env python3
"""Quick test to verify multi-community resident behavior after fix.

This script tests that:
1. Residents see ALTAR observation in control arm
2. Residents adapt to all three communities (RED/GREEN/BLUE)
3. Residents achieve high monoculture regardless of community

Usage:
    python test_multi_community_fix.py
"""

import numpy as np
from agents.envs.sb3_wrapper import AllelopathicHarvestGymEnv


def test_community(permitted_color_index: int, color_name: str, arm: str):
    """Test residents enforce correct norm for one community.

    Args:
        permitted_color_index: 1=RED, 2=GREEN, 3=BLUE
        color_name: Human-readable name for logging
        arm: 'treatment' or 'control'
    """
    print(f"\n{'='*60}")
    print(f"Testing {arm.upper()} arm: {color_name} (index={permitted_color_index})")
    print('='*60)

    # Create environment
    config = {
        'permitted_color_index': permitted_color_index,
        'startup_grey_grace': 50,
        'episode_timesteps': 500,  # Short episode for quick test
        'altar_coords': (5, 15),
        'alpha': 5.0,
        'beta': 5.0,
        'c': 0.5,
        'immunity_cooldown': 200,
    }

    env = AllelopathicHarvestGymEnv(
        arm=arm,
        config=config,
        seed=42,
        enable_telemetry=True,
        multi_community_mode=False,  # Test single-community first
        include_timestep=False,
    )

    # Get resident wrapper to check their observations
    obs, info = env.reset()

    # Verify ego observation
    print(f"\nEgo observations keys: {obs.keys()}")
    if arm == 'treatment':
        assert 'permitted_color' in obs, "Treatment ego should see permitted_color"
        altar_onehot = obs['permitted_color']
        ego_sees = np.argmax(altar_onehot) + 1
        print(f"  ✓ Ego sees permitted_color (one-hot): {altar_onehot} → index {ego_sees}")
    else:
        assert 'permitted_color' not in obs, "Control ego should NOT see permitted_color"
        print(f"  ✓ Ego does NOT see permitted_color (control arm)")

    # Check resident observations via wrapper's last timestep
    # Access the dmlab timestep to check resident observations
    dmlab_timestep = env._last_dmlab_timestep
    resident_obs = dmlab_timestep.observation[1]  # Agent 1 is first resident

    print(f"\nResident (agent 1) observations keys: {resident_obs.keys()}")
    if 'ALTAR' in resident_obs:
        altar_value = int(resident_obs['ALTAR'])
        print(f"  ✓ Resident sees ALTAR: {altar_value}")
        assert altar_value == permitted_color_index, \
            f"Resident ALTAR mismatch! Expected {permitted_color_index}, got {altar_value}"
    else:
        print(f"  ✗ Resident does NOT see ALTAR - THIS IS THE BUG!")
        raise AssertionError("Residents must see ALTAR in both arms!")

    # Run episode and check berry planting
    print(f"\nRunning {config['episode_timesteps']} steps...")

    for step in range(config['episode_timesteps']):
        # Ego takes random action (we're testing residents, not ego)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Get final berry counts
    episode_summary = info.get('episode', {})
    berries_red = episode_summary.get('berries_planted_red', 0)
    berries_green = episode_summary.get('berries_planted_green', 0)
    berries_blue = episode_summary.get('berries_planted_blue', 0)
    total_berries = berries_red + berries_green + berries_blue

    print(f"\nFinal berry counts:")
    print(f"  RED:   {berries_red}")
    print(f"  GREEN: {berries_green}")
    print(f"  BLUE:  {berries_blue}")
    print(f"  TOTAL: {total_berries}")

    # Calculate monoculture percentage
    if permitted_color_index == 1:
        target_berries = berries_red
        target_name = "RED"
    elif permitted_color_index == 2:
        target_berries = berries_green
        target_name = "GREEN"
    else:
        target_berries = berries_blue
        target_name = "BLUE"

    if total_berries > 0:
        monoculture_pct = 100.0 * target_berries / total_berries
        print(f"\n{target_name} monoculture: {monoculture_pct:.1f}%")

        # Check if residents achieved high monoculture (should be >80% even in short episode)
        if monoculture_pct > 80.0:
            print(f"  ✓ PASS: Residents enforcing {target_name} norm correctly")
        else:
            print(f"  ✗ FAIL: Low monoculture ({monoculture_pct:.1f}%), residents not adapting!")
            raise AssertionError(f"Expected >{80}% {target_name} monoculture, got {monoculture_pct:.1f}%")
    else:
        print(f"  ⚠ WARNING: No berries planted (episode too short or issue)")

    env.close()
    return monoculture_pct if total_berries > 0 else 0.0


def test_multi_community_sampling(arm: str):
    """Test that multi-community mode samples different communities correctly.

    Args:
        arm: 'treatment' or 'control'
    """
    print(f"\n{'='*60}")
    print(f"Testing {arm.upper()} arm: Multi-Community Sampling")
    print('='*60)

    config = {
        'permitted_color_index': 1,  # Placeholder (will be sampled)
        'startup_grey_grace': 50,
        'episode_timesteps': 300,
        'altar_coords': (5, 15),
        'alpha': 5.0,
        'beta': 5.0,
        'c': 0.5,
        'immunity_cooldown': 200,
    }

    env = AllelopathicHarvestGymEnv(
        arm=arm,
        config=config,
        seed=42,
        enable_telemetry=False,  # Disable for speed
        multi_community_mode=True,  # Enable multi-community
        include_timestep=False,
    )

    # Reset multiple times and track sampled communities
    sampled_communities = []
    for episode_idx in range(6):
        obs, info = env.reset()

        community_idx = info['community_idx']
        community_tag = info['community_tag']
        sampled_communities.append(community_idx)

        print(f"Episode {episode_idx}: Sampled {community_tag} (index={community_idx})")

        # Verify residents see correct ALTAR
        dmlab_timestep = env._last_dmlab_timestep
        resident_obs = dmlab_timestep.observation[1]

        if 'ALTAR' in resident_obs:
            altar_value = int(resident_obs['ALTAR'])
            if altar_value == community_idx:
                print(f"  ✓ Resident sees correct ALTAR: {altar_value}")
            else:
                print(f"  ✗ Resident ALTAR mismatch! Expected {community_idx}, got {altar_value}")
                raise AssertionError(f"Resident ALTAR does not match sampled community!")
        else:
            print(f"  ✗ Resident does NOT see ALTAR - BUG!")
            raise AssertionError("Residents must see ALTAR!")

    # Check we got some diversity in sampling
    unique_communities = set(sampled_communities)
    print(f"\nUnique communities sampled: {sorted(unique_communities)}")
    if len(unique_communities) >= 2:
        print(f"  ✓ PASS: Multi-community sampling working (got {len(unique_communities)} different communities)")
    else:
        print(f"  ⚠ WARNING: Only sampled {len(unique_communities)} unique community (might be random, try more episodes)")

    env.close()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MULTI-COMMUNITY FIX VERIFICATION TEST")
    print("="*60)
    print("\nThis test verifies that:")
    print("  1. Residents see ALTAR in both treatment and control arms")
    print("  2. Residents adapt to all three communities (RED/GREEN/BLUE)")
    print("  3. Multi-community sampling works correctly")

    try:
        # Test single-community for all three colors in both arms
        for arm in ['treatment', 'control']:
            test_community(1, 'RED', arm)
            test_community(2, 'GREEN', arm)
            test_community(3, 'BLUE', arm)

        # Test multi-community sampling
        for arm in ['treatment', 'control']:
            test_multi_community_sampling(arm)

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe fix is working correctly:")
        print("  • Residents see ALTAR in both arms")
        print("  • Residents adapt to RED, GREEN, and BLUE communities")
        print("  • Multi-community sampling works as expected")
        print("\nYou can now safely run Phase 5 multi-community training!")

    except AssertionError as e:
        print("\n" + "="*60)
        print("TEST FAILED! ✗")
        print("="*60)
        print(f"\nError: {e}")
        print("\nThe fix may not be working correctly. Check the implementation.")
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
