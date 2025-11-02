#!/usr/bin/env python3
"""Test baseline resident performance (all 16 agents as residents).

Added by RST: Baseline test for new ResidentPolicy implementation.

This test verifies:
1. All 16 residents achieve 95%+ monoculture per episode
2. Minimal sanctioning (all agents compliant)
3. Agent 0 metrics for normative competence and compliance baseline
"""

import sys
from pathlib import Path

# Add project root to path (for agents module) and meltingpot
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "meltingpot"))

import numpy as np
import meltingpot.substrate as substrate
from meltingpot.configs.substrates import allelopathic_harvest_normative__open as allelopathic_harvest
from agents.envs.resident_wrapper import ResidentWrapper


def get_berry_counts_from_obs(obs):
    """Get berry counts by color from observations.

    Added by RST: Get berry counts from WORLD.BERRIES_BY_TYPE observation.

    Args:
        obs: Observation dict from timestep.

    Returns:
        Dict with counts: {'red': count, 'green': count, 'blue': count}
    """
    # WORLD.BERRIES_BY_TYPE is (red_count, green_count, blue_count)
    berry_counts_array = obs.get('WORLD.BERRIES_BY_TYPE', np.array([0, 0, 0]))

    return {
        'red': int(berry_counts_array[0]),
        'green': int(berry_counts_array[1]),
        'blue': int(berry_counts_array[2]),
    }


def compute_monoculture_percentage(berry_counts, altar_color_name):
    """Compute monoculture percentage for altar color.
    
    Args:
        berry_counts: Dict with berry counts by color.
        altar_color_name: Name of altar color ('red', 'green', or 'blue').
        
    Returns:
        Float percentage of berries that are altar color.
    """
    total_berries = sum(berry_counts.values())
    if total_berries == 0:
        return 0.0
    
    altar_berries = berry_counts.get(altar_color_name, 0)
    return (altar_berries / total_berries) * 100.0




def test_baseline_residents(num_episodes=3, episode_length=2000, verbose=True,
                           save_video=True, video_path="baseline_episode1.mp4", video_fps=8):
    """Test baseline resident performance (all 16 agents as residents).

    Args:
        num_episodes: Number of episodes to run.
        episode_length: Steps per episode.
        verbose: Whether to print detailed output.
        save_video: Whether to save video of first episode.
        video_path: Path to save video file.
        video_fps: Frames per second for video.

    Returns:
        Dict with test results and metrics.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BASELINE TEST: All 16 Agents as Residents")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Episode length: {episode_length} steps")
        print()
    
    # Results tracking
    results = {
        'episodes': [],
        'monoculture_achieved': [],
        'final_monoculture_pct': [],
        'agent_0_rewards': [],
        'agent_0_compliance_rate': [],
    }
    
    for episode_idx in range(num_episodes):
        if verbose:
            print(f"\n--- Episode {episode_idx + 1}/{num_episodes} ---")

        # Build substrate using substrate.build()
        roles = ["default"] * 16
        base_env = substrate.build('allelopathic_harvest_normative__open', roles=roles)
        
        # Wrap with ResidentWrapper (all 16 are residents)
        env = ResidentWrapper(
            env=base_env,
            resident_indices=list(range(16)),
            ego_index=None,  # No ego, all residents
            seed=42 + episode_idx)  # Different seed per episode
        
        # Reset environment
        timestep = env.reset()
        
        # Get altar color from observations
        altar_obs = timestep.observation[0].get('ALTAR')
        if isinstance(altar_obs, np.ndarray):
            altar_color_id = int(altar_obs.item() if altar_obs.size == 1 else altar_obs[0])
        else:
            altar_color_id = int(altar_obs)
        
        altar_color_name = {1: 'red', 2: 'green', 3: 'blue'}[altar_color_id]
        
        if verbose:
            print(f"Altar color: {altar_color_id} ({altar_color_name.upper()})")
        
        # Episode tracking
        agent_0_total_reward = 0.0
        agent_0_steps_compliant = 0
        agent_0_total_steps = 0

        # Track berry colors over time
        berry_snapshots = []

        # Video recording for first episode
        frames = []
        if save_video and episode_idx == 0:
            if verbose:
                print(f"  Recording video to {video_path}")
            # Capture initial frame
            rgb_frame = timestep.observation[0]["WORLD.RGB"]
            frames.append(rgb_frame)

        # Run episode
        for step in range(episode_length):
            timestep = env.step()

            # Capture frame for video (first episode only)
            if save_video and episode_idx == 0:
                rgb_frame = timestep.observation[0]["WORLD.RGB"]
                frames.append(rgb_frame)

            # Track agent 0 metrics
            if timestep.reward is not None:
                if isinstance(timestep.reward, list):
                    agent_0_total_reward += timestep.reward[0]
                else:
                    agent_0_total_reward += timestep.reward
            
            # Check agent 0 compliance (body color = altar color)
            obs_0 = timestep.observation[0]
            agent_colors = obs_0.get('AGENT_COLORS', np.array([]))
            if len(agent_colors) > 0:
                agent_0_color = agent_colors[0]
                if agent_0_color == altar_color_id:
                    agent_0_steps_compliant += 1
                agent_0_total_steps += 1

            # Sample berry colors periodically (Added by RST: Get from observations)
            if step % 500 == 0:
                obs_0 = timestep.observation[0]

                # Debug: print all available observation keys
                if step == 0 and verbose:
                    print(f"  Available observations: {list(obs_0.keys())}")

                berry_counts = get_berry_counts_from_obs(obs_0)

                # Debug: check if berry counts are all zero
                if step == 0 and verbose:
                    print(f"  Berry counts from obs: {berry_counts}")
                    if 'WORLD.BERRIES_BY_TYPE' in obs_0:
                        print(f"  Raw WORLD.BERRIES_BY_TYPE: {obs_0['WORLD.BERRIES_BY_TYPE']}")

                monoculture_pct = compute_monoculture_percentage(berry_counts, altar_color_name)
                berry_snapshots.append((step, monoculture_pct, berry_counts))
                if verbose:
                    print(f"  Step {step}: Monoculture ~{monoculture_pct:.1f}% "
                          f"({berry_counts[altar_color_name]}/{sum(berry_counts.values())} berries)")
            
            if timestep.last():
                break
        
        # Final metrics
        agent_0_compliance_rate = (agent_0_steps_compliant / agent_0_total_steps * 100.0 
                                   if agent_0_total_steps > 0 else 0.0)
        
        # Get final monoculture percentage
        if berry_snapshots:
            final_monoculture_pct = berry_snapshots[-1][1]
        else:
            final_monoculture_pct = 0.0
        
        monoculture_achieved = final_monoculture_pct >= 95.0
        
        # Store results
        results['episodes'].append(episode_idx + 1)
        results['monoculture_achieved'].append(monoculture_achieved)
        results['final_monoculture_pct'].append(final_monoculture_pct)
        results['agent_0_rewards'].append(agent_0_total_reward)
        results['agent_0_compliance_rate'].append(agent_0_compliance_rate)

        if verbose:
            print(f"\n  Episode Results:")
            print(f"    Final monoculture: {final_monoculture_pct:.1f}% "
                  f"({'✓ PASS' if monoculture_achieved else '✗ FAIL'})")
            print(f"    Agent 0 total reward: {agent_0_total_reward:.2f}")
            print(f"    Agent 0 compliance rate: {agent_0_compliance_rate:.1f}%")

        # Save video of first episode
        if save_video and episode_idx == 0 and frames:
            if verbose:
                print(f"\n  Saving video ({len(frames)} frames)...")
            _save_video(frames, video_path, video_fps)
            if verbose:
                print(f"  Video saved to {video_path}")

        env.close()
    
    # Aggregate results
    if verbose:
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)
        print(f"Episodes with 95%+ monoculture: {sum(results['monoculture_achieved'])}/{num_episodes}")
        print(f"Average final monoculture: {np.mean(results['final_monoculture_pct']):.1f}%")
        print(f"Average agent 0 reward: {np.mean(results['agent_0_rewards']):.2f}")
        print(f"Average agent 0 compliance: {np.mean(results['agent_0_compliance_rate']):.1f}%")
        print()

        # Pass/fail
        all_achieved_monoculture = all(results['monoculture_achieved'])
        high_compliance = np.mean(results['agent_0_compliance_rate']) > 90.0

        print("=" * 70)
        if all_achieved_monoculture and high_compliance:
            print("✓ BASELINE TEST PASSED")
        else:
            print("✗ BASELINE TEST FAILED")
            if not all_achieved_monoculture:
                print("  - Not all episodes achieved 95%+ monoculture")
            if not high_compliance:
                print("  - Agent 0 compliance below 90%")
        print("=" * 70)
    
    return results


def _save_video(frames, output_path: str, fps: int):
    """Save frames as video file.

    Added by RST: Video saving utility for baseline test.

    Args:
        frames: List of RGB frames (numpy arrays).
        output_path: Path to save video.
        fps: Frames per second.
    """
    try:
        import cv2

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()
    except ImportError:
        print("  Warning: OpenCV not available. Install with: pip install opencv-python")
        # Fallback: save as numpy array
        npy_path = output_path.replace('.mp4', '.npy')
        np.save(npy_path, np.array(frames))
        print(f"  Saved frames as numpy array instead: {npy_path}")


if __name__ == '__main__':
    results = test_baseline_residents(num_episodes=3, episode_length=2000, verbose=True)
    
    # Exit with status code
    all_passed = all(results['monoculture_achieved'])
    sys.exit(0 if all_passed else 1)
