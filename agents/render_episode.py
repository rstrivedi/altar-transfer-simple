# Added by RST: Video rendering script for normative allelopathic harvest
"""Render episodes of normative allelopathic harvest to video for visualization.

This script allows visual inspection of:
- Agent behavior (planting, eating, zapping)
- Color changes (compliance vs violation)
- Sanction patterns
- Altar visibility (treatment vs control)
"""

import argparse
from typing import Optional

import numpy as np
from meltingpot.utils.substrates import substrate
from meltingpot.configs.substrates import allelopathic_harvest
from agents.envs import NormativeObservationFilter, NormativeMetricsLogger


def render_episode(
    num_players: int = 16,
    permitted_color: int = 1,
    enable_treatment: bool = True,
    output_path: Optional[str] = None,
    episode_length: int = 1000,
    fps: int = 8,
    interactive: bool = False,
    ego_player: int = 0,
):
  """Render an episode to video.

  Args:
    num_players: Number of players in the environment.
    permitted_color: Permitted color index (1=RED, 2=GREEN, 3=BLUE).
    enable_treatment: If True, treatment condition (show altar and PERMITTED_COLOR).
      If False, control condition.
    output_path: Path to save video file. If None, displays interactively.
    episode_length: Number of frames to render.
    fps: Frames per second for video.
    interactive: If True, use keyboard controls for ego player.
    ego_player: Which player to control interactively (0-indexed).
  """
  # Get config with normative gate enabled
  config = allelopathic_harvest.get_config()
  config.normative_gate = True
  config.enable_treatment_condition = enable_treatment
  config.permitted_color_index = permitted_color

  # Set altar position if treatment (center-top of map for visibility)
  if enable_treatment:
    config.altar_coords = (5, 15)  # Adjust based on map layout

  # Build substrate
  roles = ["default"] * num_players
  env = substrate.build(
      substrate_name="allelopathic_harvest",
      roles=roles,
      config=config)

  # Wrap with observation filter
  env = NormativeObservationFilter(env, enable_treatment_condition=enable_treatment)

  # Initialize metrics logger
  logger = NormativeMetricsLogger(num_players)

  # Reset environment
  timestep = env.reset()
  logger.reset()
  logger.process_events(env.events())

  frames = []
  rgb_frame = env.observation()[0]["WORLD.RGB"]
  frames.append(rgb_frame)

  # Run episode
  if interactive:
    print("\nInteractive mode - Keyboard controls:")
    print("  Arrow keys: move")
    print("  WASD: turn left/right, move forward/backward")
    print("  Space: fire zap")
    print("  1,2,3: plant red/green/blue berry")
    print("  Q: quit")
    print()

  for step in range(episode_length):
    if interactive:
      ego_action = _get_interactive_action(env, ego_player)
      if ego_action is None:  # User quit
        break
      actions = [ego_action if i == ego_player else env.action_spec()[i].generate_value()
                 for i in range(num_players)]
    else:
      actions = [env.action_spec()[i].generate_value() for i in range(num_players)]

    timestep = env.step(actions)
    events = env.events()
    logger.process_events(events)

    rgb_frame = env.observation()[0]["WORLD.RGB"]
    frames.append(rgb_frame)

    if interactive:
      _print_step_info(step, ego_player, ego_action, timestep, events, logger)
      _display_frame(rgb_frame, step)

    if timestep.last():
      break

  # Get episode summary
  summary = logger.get_episode_summary()
  print(f"Episode complete: {len(frames)} frames")
  print(f"Alpha total: {summary['alpha_total']:.2f}")
  print(f"Beta total: {summary['beta_total']:.2f}")
  print(f"C total: {summary['c_total']:.2f}")

  # Save or display video
  if output_path:
    _save_video(frames, output_path, fps)
    print(f"Video saved to {output_path}")
  else:
    print("Interactive display not implemented. Specify --output to save video.")

  env.close()


def _print_step_info(step: int, ego_player: int, ego_action: int,
                     timestep, events, logger):
  """Print information about what happened this step.

  Args:
    step: Current step number.
    ego_player: Ego player index (0-indexed).
    ego_action: Action taken by ego.
    timestep: Timestep from environment.
    events: Events from env.events().
    logger: NormativeMetricsLogger instance.
  """
  action_names = ['NOOP', 'FORWARD', 'STEP_RIGHT', 'BACKWARD', 'STEP_LEFT',
                  'TURN_LEFT', 'TURN_RIGHT', 'FIRE_ZAP',
                  'PLANT_RED', 'PLANT_GREEN', 'PLANT_BLUE']

  print(f"\n--- Step {step} ---")
  print(f"Action: {action_names[ego_action] if ego_action < len(action_names) else ego_action}")
  print(f"Reward: {timestep.reward[ego_player]:.2f}")

  # Show reward components for ego
  alpha = logger.get_alpha_sum(ego_player)
  beta = logger.get_beta_sum(ego_player)
  c = logger.get_c_sum(ego_player)
  print(f"Cumulative - α: {alpha:.2f}, β: {beta:.2f}, c: {c:.2f}")

  # Show relevant events
  for event in events:
    event_name = event.get('name', '')

    # Sanction events involving ego
    if event_name == 'sanction_event':
      zapper_idx = event.get('zapper_index', 0) - 1  # Convert to 0-indexed
      target_idx = event.get('target_index', 0) - 1

      if zapper_idx == ego_player:
        is_violation = event.get('is_violation', False)
        immune = event.get('target_immune', False)
        tied = event.get('tie_break_blocked', False)

        if immune:
          print(f"  → Zapped player {target_idx} (IMMUNE - no effect)")
        elif tied:
          print(f"  → Zapped player {target_idx} (TIE-BREAK - blocked)")
        elif is_violation:
          print(f"  → Zapped player {target_idx} (CORRECT - violator)")
        else:
          print(f"  → Zapped player {target_idx} (MIS-ZAP - compliant)")

      if target_idx == ego_player:
        applied = event.get('sanction_applied', False)
        if applied:
          print(f"  ← Got zapped by player {zapper_idx} (-10 reward)")


def _get_interactive_action(env, player_idx: int):
  """Get action from keyboard input.

  Args:
    env: Environment.
    player_idx: Which player to get action for.

  Returns:
    Action dictionary or None if user quits.
  """
  try:
    import readchar
  except ImportError:
    print("readchar not available. Install with: pip install readchar")
    print("Using dummy action instead.")
    return env.action_spec()[player_idx].generate_value()

  # Action space: NOOP, FORWARD, STEP_RIGHT, BACKWARD, STEP_LEFT,
  #               TURN_LEFT, TURN_RIGHT, FIRE_ZAP, FIRE_ONE, FIRE_TWO, FIRE_THREE
  key = readchar.readkey()

  if key == 'q' or key == 'Q':
    return None  # Quit
  elif key == 'w' or key == readchar.key.UP:
    return 1  # FORWARD
  elif key == 's' or key == readchar.key.DOWN:
    return 3  # BACKWARD
  elif key == 'd' or key == readchar.key.RIGHT:
    return 2  # STEP_RIGHT
  elif key == 'a' or key == readchar.key.LEFT:
    return 4  # STEP_LEFT
  elif key == 'e':
    return 6  # TURN_RIGHT
  elif key == 'q':
    return 5  # TURN_LEFT
  elif key == ' ':
    return 7  # FIRE_ZAP
  elif key == '1':
    return 8  # FIRE_ONE (plant red)
  elif key == '2':
    return 9  # FIRE_TWO (plant green)
  elif key == '3':
    return 10  # FIRE_THREE (plant blue)
  else:
    return 0  # NOOP


def _display_frame(frame, step: int):
  """Display frame in terminal or window.

  Args:
    frame: RGB frame array.
    step: Current step number.
  """
  try:
    import cv2
    cv2.imshow(f"Step {step}", frame)
    cv2.waitKey(50)  # 50ms delay for display
  except ImportError:
    # No display, just continue
    pass


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
  except ImportError:
    print("OpenCV not available. Install with: pip install opencv-python")
    # Fallback: save as numpy array
    np.save(output_path.replace('.mp4', '.npy'), np.array(frames))
    print(f"Saved frames as numpy array instead: {output_path.replace('.mp4', '.npy')}")


def main():
  parser = argparse.ArgumentParser(
      description="Render normative allelopathic harvest episode")
  parser.add_argument("--num_players", type=int, default=16,
                      help="Number of players")
  parser.add_argument("--permitted_color", type=int, default=1, choices=[1, 2, 3],
                      help="Permitted color: 1=RED, 2=GREEN, 3=BLUE")
  parser.add_argument("--treatment", action="store_true",
                      help="Enable treatment condition (show altar and PERMITTED_COLOR)")
  parser.add_argument("--control", action="store_true",
                      help="Enable control condition (no altar, no PERMITTED_COLOR)")
  parser.add_argument("--output", type=str, default=None,
                      help="Output video path (e.g., episode.mp4)")
  parser.add_argument("--episode_length", type=int, default=1000,
                      help="Number of frames to render")
  parser.add_argument("--fps", type=int, default=8,
                      help="Frames per second for video")
  parser.add_argument("--interactive", action="store_true",
                      help="Enable interactive keyboard control for ego player")
  parser.add_argument("--ego_player", type=int, default=0,
                      help="Which player to control interactively (0-indexed)")

  args = parser.parse_args()

  # Determine treatment condition
  if args.treatment and args.control:
    parser.error("Cannot specify both --treatment and --control")
  elif args.control:
    enable_treatment = False
  else:
    enable_treatment = True  # Default to treatment

  render_episode(
      num_players=args.num_players,
      permitted_color=args.permitted_color,
      enable_treatment=enable_treatment,
      output_path=args.output,
      episode_length=args.episode_length,
      fps=args.fps,
      interactive=args.interactive,
      ego_player=args.ego_player)


if __name__ == "__main__":
  main()
