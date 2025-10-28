# Added by RST: Enhanced video rendering with overlays for Phase 3
"""Video capture and rendering with optional telemetry overlays.

This module provides episode rendering with overlays showing:
- Berry counts by color (running totals)
- Permitted color indicator
- Compliance status (green=compliant, red=violating)
- Sanction counter
- Altar visibility (treatment only)

Usage:
  frames = render_episode_with_overlays(
      env=env,
      ego_policy=policy,
      permitted_color_index=1,
      show_altar=True,
      overlay_config={'show_berry_counts': True, 'show_compliance': True})

  save_video(frames, 'episode.mp4', fps=8)
"""

from typing import List, Optional, Dict, Callable
import numpy as np


def render_episode_with_overlays(
    env,
    ego_policy: Callable,
    recorder,  # MetricsRecorder instance
    permitted_color_index: int,
    show_altar: bool,
    max_steps: int = 2000,
    overlay_config: Optional[Dict] = None,
) -> List[np.ndarray]:
  """Render episode with optional telemetry overlays.

  Args:
    env: ResidentWrapper-wrapped environment.
    ego_policy: Function mapping observation → action for ego.
    recorder: MetricsRecorder instance (already reset).
    permitted_color_index: 1=RED, 2=GREEN, 3=BLUE.
    show_altar: True if treatment (altar visible), False if control.
    max_steps: Maximum episode length.
    overlay_config: Dict with overlay flags:
      - 'show_berry_counts': Show running berry counts (default True)
      - 'show_permitted_color': Show permitted color chip (default True)
      - 'show_compliance': Show compliance border (default True)
      - 'show_sanction_count': Show sanction counter (default True)

  Returns:
    List of RGB frames (numpy arrays).
  """
  if overlay_config is None:
    overlay_config = {}

  show_berry_counts = overlay_config.get('show_berry_counts', True)
  show_permitted_color_chip = overlay_config.get('show_permitted_color', True)
  show_compliance_border = overlay_config.get('show_compliance', True)
  show_sanction_count = overlay_config.get('show_sanction_count', True)

  # Reset environment
  timestep = env.reset()
  events = env.events()

  frames = []

  # Get initial RGB frame from WORLD.RGB observation
  rgb_frame = env.observation()[0]["WORLD.RGB"]
  frames.append(rgb_frame.copy())

  # Track sanctions received
  num_sanctions_received = 0

  for step in range(max_steps):
    # Get ego observation and action
    ego_obs = timestep.observation[0]  # Ego is index 0
    ego_action = ego_policy(ego_obs)

    # Step environment
    timestep = env.step(ego_action)
    events = env.events()

    # Record telemetry
    recorder.record_step(step + 1, timestep, events, ego_action)

    # Get RGB frame
    rgb_frame = env.observation()[0]["WORLD.RGB"]

    # Apply overlays if OpenCV available
    try:
      import cv2
      rgb_frame_with_overlays = _apply_overlays(
          rgb_frame.copy(),
          recorder=recorder,
          permitted_color_index=permitted_color_index,
          show_berry_counts=show_berry_counts,
          show_permitted_color_chip=show_permitted_color_chip,
          show_compliance_border=show_compliance_border,
          show_sanction_count=show_sanction_count,
      )
      frames.append(rgb_frame_with_overlays)
    except ImportError:
      # OpenCV not available, save raw frames
      frames.append(rgb_frame.copy())

    if timestep.last():
      break

  return frames


def _apply_overlays(
    frame: np.ndarray,
    recorder,
    permitted_color_index: int,
    show_berry_counts: bool,
    show_permitted_color_chip: bool,
    show_compliance_border: bool,
    show_sanction_count: bool,
) -> np.ndarray:
  """Apply overlays to RGB frame.

  Args:
    frame: RGB frame (H, W, 3).
    recorder: MetricsRecorder with current state.
    permitted_color_index: 1=RED, 2=GREEN, 3=BLUE.
    show_berry_counts: Show berry count text.
    show_permitted_color_chip: Show permitted color chip.
    show_compliance_border: Show compliance border.
    show_sanction_count: Show sanction counter.

  Returns:
    RGB frame with overlays applied.
  """
  import cv2

  # Color mappings (BGR for OpenCV)
  COLOR_MAP = {
      1: (0, 0, 255),    # RED
      2: (0, 255, 0),    # GREEN
      3: (255, 0, 0),    # BLUE
      0: (128, 128, 128),  # GREY
  }

  COLOR_NAMES = {1: 'R', 2: 'G', 3: 'B'}

  # === Permitted color chip (top-right) ===
  if show_permitted_color_chip:
    chip_size = 30
    margin = 10
    h, w = frame.shape[:2]
    top_left = (w - chip_size - margin, margin)
    bottom_right = (w - margin, margin + chip_size)
    color_bgr = COLOR_MAP[permitted_color_index]
    cv2.rectangle(frame, top_left, bottom_right, color_bgr, -1)  # Filled
    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)  # Border

  # === Berry counts (top-left) ===
  if show_berry_counts:
    step_metrics = recorder.get_step_metrics()
    if step_metrics:
      berry_counts = step_metrics[-1].berry_counts  # (red, green, blue)
      text = f"R:{berry_counts[0]} G:{berry_counts[1]} B:{berry_counts[2]}"
      cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                  0.6, (255, 255, 255), 2, cv2.LINE_AA)

  # === Compliance border ===
  if show_compliance_border:
    ego_body_color = recorder.get_ego_body_color()
    step_metrics = recorder.get_step_metrics()
    if step_metrics:
      current_step = step_metrics[-1]
      t = current_step.t
      grace_period = 25  # From config

      # Check compliance
      compliant = (ego_body_color == permitted_color_index) or \
                  (ego_body_color == 0 and t < grace_period)

      # Draw border
      h, w = frame.shape[:2]
      border_color = (0, 255, 0) if compliant else (0, 0, 255)  # Green or Red
      thickness = 5
      cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, thickness)

  # === Sanction count (bottom-left) ===
  if show_sanction_count:
    step_metrics = recorder.get_step_metrics()
    num_sanctions = 0
    if step_metrics:
      for step in step_metrics:
        for sanction in step.sanctions:
          if sanction.zappee_id == 0 and sanction.applied_minus10:  # Ego is index 0
            num_sanctions += 1

    h, w = frame.shape[:2]
    text = f"Sanctions: {num_sanctions}"
    cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

  return frame


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 8,
) -> None:
  """Save frames as mp4 video.

  Args:
    frames: List of RGB frames (numpy arrays).
    output_path: Path to save video (e.g., 'episode.mp4').
    fps: Frames per second.
  """
  try:
    import cv2

    if not frames:
      print("No frames to save")
      return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
      # Convert RGB to BGR for OpenCV
      bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      out.write(bgr_frame)

    out.release()
    print(f"Video saved to {output_path}")

  except ImportError:
    print("OpenCV not available. Install with: pip install opencv-python")
    # Fallback: save as numpy array
    fallback_path = output_path.replace('.mp4', '.npy')
    np.save(fallback_path, np.array(frames))
    print(f"Saved frames as numpy array instead: {fallback_path}")


def save_frames_as_gif(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 8,
) -> None:
  """Save frames as GIF (alternative to mp4).

  Args:
    frames: List of RGB frames (numpy arrays).
    output_path: Path to save GIF (e.g., 'episode.gif').
    fps: Frames per second.
  """
  try:
    from PIL import Image

    if not frames:
      print("No frames to save")
      return

    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0)

    print(f"GIF saved to {output_path}")

  except ImportError:
    print("PIL not available. Install with: pip install pillow")
    # Fallback: save as numpy array
    fallback_path = output_path.replace('.gif', '.npy')
    np.save(fallback_path, np.array(frames))
    print(f"Saved frames as numpy array instead: {fallback_path}")
