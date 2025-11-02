#!/usr/bin/env python3
"""Render baseline resident episode to video.

Added by RST: Video rendering for baseline residents test.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "meltingpot"))

import numpy as np
import meltingpot.substrate as substrate
from agents.envs.resident_wrapper import ResidentWrapper


def render_baseline_episode(output_path: str, episode_length: int = 2000, fps: int = 8, seed: int = 42):
    """Render baseline residents episode (all 16 as residents).

    Args:
        output_path: Path to save video file (e.g., "baseline_episode.mp4").
        episode_length: Number of steps to render.
        fps: Frames per second for video.
        seed: Random seed for reproducibility.
    """
    print(f"Rendering baseline episode to {output_path}")
    print(f"  Episode length: {episode_length} steps")
    print(f"  FPS: {fps}")
    print()

    # Build substrate (same as test)
    roles = ["default"] * 16
    base_env = substrate.build('allelopathic_harvest_normative__open', roles=roles)

    # Wrap with ResidentWrapper (all 16 are residents)
    env = ResidentWrapper(
        env=base_env,
        resident_indices=list(range(16)),
        ego_index=None,
        seed=seed)

    # Reset environment
    timestep = env.reset()

    # Get altar color
    altar_obs = timestep.observation[0].get('ALTAR')
    if isinstance(altar_obs, np.ndarray):
        altar_color = int(altar_obs.item() if altar_obs.size == 1 else altar_obs[0])
    else:
        altar_color = int(altar_obs)

    altar_color_name = {1: 'RED', 2: 'GREEN', 3: 'BLUE'}[altar_color]
    print(f"Altar color: {altar_color} ({altar_color_name})")
    print()

    # Collect frames
    frames = []

    # Get initial frame
    rgb_frame = timestep.observation[0]["WORLD.RGB"]
    frames.append(rgb_frame)

    print("Running episode...")
    for step in range(episode_length):
        timestep = env.step()

        # Get RGB frame
        rgb_frame = timestep.observation[0]["WORLD.RGB"]
        frames.append(rgb_frame)

        # Progress update
        if (step + 1) % 500 == 0:
            # Get berry counts for progress
            berry_counts_array = timestep.observation[0].get('WORLD.BERRIES_BY_TYPE', np.array([0, 0, 0]))
            berry_counts = {
                'red': int(berry_counts_array[0]),
                'green': int(berry_counts_array[1]),
                'blue': int(berry_counts_array[2]),
            }
            total_berries = sum(berry_counts.values())
            altar_berries = berry_counts[altar_color_name.lower()]
            monoculture_pct = (altar_berries / total_berries * 100.0) if total_berries > 0 else 0.0

            print(f"  Step {step + 1}/{episode_length}: Monoculture ~{monoculture_pct:.1f}%")

        if timestep.last():
            break

    print(f"\nEpisode complete: {len(frames)} frames")

    # Save video
    _save_video(frames, output_path, fps)
    print(f"Video saved to {output_path}")

    env.close()


def _save_video(frames, output_path: str, fps: int):
    """Save frames as video file.

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
        print("Video encoding complete.")
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
        # Fallback: save as numpy array
        npy_path = output_path.replace('.mp4', '.npy')
        np.save(npy_path, np.array(frames))
        print(f"Saved frames as numpy array instead: {npy_path}")
        print("To view: load the .npy file and display frames manually")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Render baseline residents episode")
    parser.add_argument("--output", type=str, default="baseline_episode.mp4",
                        help="Output video path")
    parser.add_argument("--episode_length", type=int, default=2000,
                        help="Number of steps to render")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    render_baseline_episode(
        output_path=args.output,
        episode_length=args.episode_length,
        fps=args.fps,
        seed=args.seed)
