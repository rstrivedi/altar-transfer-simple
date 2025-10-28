# Added by RST: Phase 2 acceptance tests for scripted resident agents
"""Tests that residents enforce norms, achieve monoculture, and behave identically across conditions.

R1: Selectivity - residents never zap compliant agents
R2: Coverage - ≥80% of violators sanctioned within 10 frames
R3: No dogpiling - no duplicate -10 on immune targets, residents don't attempt
R4: Plant/harvest purity - 100% plant permitted, ≥95% harvest permitted
R5: Arm invariance - identical decisions in control vs treatment
R6: Same-step tie-break - only one -10 lands when multiple zaps
R7: No hidden dependencies - grep for freeze/removal references
R8: Monoculture achievement - ≥85% permitted berries at t=2000
"""

import sys
sys.path.insert(0, 'meltingpot')

from meltingpot.python.utils import substrate
from meltingpot.configs.substrates import allelopathic_harvest

from agents.residents.config import COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_GREY
from agents.residents.info_extractor import ResidentInfoExtractor
from agents.residents.scripted_residents import ResidentController
from agents.envs.resident_wrapper import ResidentWrapper


def test_r1_selectivity():
  """R1: Residents never zap compliant agents.

  Setup: All 16 agents are residents. Run for 500 steps.
  Track all zap events via reward_component events (alpha/beta).
  Verify: All zaps are on violators (beta == 0, only alpha > 0).
  """
  # Configure environment
  config = allelopathic_harvest.get_config()
  config.normative_gate = True
  config.permitted_color_index = COLOR_RED  # RED is permitted
  config.ego_index = None  # All-residents baseline mode
  config.episode_timesteps = 500  # Shorter episode for test

  roles = ["default"] * 16
  base_env = substrate.build("allelopathic_harvest__normative", roles, config)

  # Setup resident controller and wrapper
  extractor = ResidentInfoExtractor(
      num_players=16,
      permitted_color_index=config.permitted_color_index,
      startup_grey_grace=config.startup_grey_grace)

  controller = ResidentController()
  controller.reset(seed=42)

  env = ResidentWrapper(
      env=base_env,
      resident_indices=list(range(16)),  # All residents
      ego_index=None,
      resident_controller=controller,
      info_extractor=extractor)

  # Run episode
  timestep = env.reset()
  alpha_events = 0  # Correct zaps
  beta_events = 0   # Mis-zaps

  for step in range(500):
    events = env.events()

    # Track reward_component events
    for event in events:
      if event.get('name') == 'reward_component':
        reward_type = event.get('type')
        value = event.get('value', 0.0)

        if reward_type == 'alpha' and value > 0:
          alpha_events += 1  # Correct zap
        elif reward_type == 'beta' and value < 0:
          beta_events += 1  # Mis-zap

    # Step with all residents (ego_action=None in baseline mode)
    timestep = env.step(ego_action=None)

    if timestep.last():
      break

  # Verify selectivity
  assert beta_events == 0, \
      f"R1 FAILED: Residents mis-zapped {beta_events} times"
  print(f"✓ R1 PASSED: Selectivity verified. {alpha_events} correct zaps, 0 mis-zaps")


def test_r2_coverage():
  """R2: ≥80% of violators sanctioned within 10 frames.

  Setup: All residents baseline mode. Track sanctions on all agents.
  Natural violations (grey agents, eating wrong color) should be sanctioned quickly.
  Verify: At least 80% of violations get sanctioned within 10 frames.
  """
  # Configure environment
  config = allelopathic_harvest.get_config()
  config.normative_gate = True
  config.permitted_color_index = COLOR_RED
  config.ego_index = None  # All residents baseline
  config.episode_timesteps = 1000  # Longer to capture natural violations

  roles = ["default"] * 16
  base_env = substrate.build("allelopathic_harvest__normative", roles, config)

  # Setup
  extractor = ResidentInfoExtractor(
      num_players=16,
      permitted_color_index=config.permitted_color_index,
      startup_grey_grace=config.startup_grey_grace)

  controller = ResidentController()
  controller.reset(seed=42)

  env = ResidentWrapper(
      env=base_env,
      resident_indices=list(range(16)),
      ego_index=None,
      resident_controller=controller,
      info_extractor=extractor)

  # Track violations and sanctions
  timestep = env.reset()

  # Track per-agent violation state
  violation_start = {}  # agent_id -> frame when violation started
  violations_detected = []  # (agent_id, start_frame, sanction_frame or None)

  for step in range(1000):
    events = env.events()
    info = extractor.extract_info(timestep.observation, events)

    # Check each agent's state
    for agent_id in range(16):
      agent_info = info['residents'][agent_id]
      nearby_agents = agent_info.get('nearby_agents', [])

      # Check if this agent is currently violating by looking at others' views
      current_violating = False
      for other_id in range(16):
        if other_id == agent_id:
          continue
        other_nearby = info['residents'][other_id].get('nearby_agents', [])
        for neighbor in other_nearby:
          if neighbor['agent_id'] == agent_id:
            # Found this agent in someone's view
            body_color = neighbor['body_color']
            permitted = info['permitted_color_index']
            grace = info['startup_grey_grace']
            is_viol = (body_color != permitted) or \
                     (body_color == COLOR_GREY and step >= grace)
            if is_viol:
              current_violating = True
              break
        if current_violating:
          break

      # Track violation periods
      if current_violating and agent_id not in violation_start:
        violation_start[agent_id] = step
      elif not current_violating and agent_id in violation_start:
        # Violation ended without sanction
        violations_detected.append((agent_id, violation_start[agent_id], None))
        del violation_start[agent_id]

    # Check for sanctions
    for event in events:
      if event.get('name') == 'sanction':
        zappee_id = event.get('zappee_id', 0) - 1  # Convert to 0-indexed
        if zappee_id in violation_start:
          # Sanction landed during violation
          violations_detected.append((zappee_id, violation_start[zappee_id], step))
          del violation_start[zappee_id]

    timestep = env.step(ego_action=None)
    if timestep.last():
      break

  # Close open violations
  for agent_id, start in violation_start.items():
    violations_detected.append((agent_id, start, None))

  # Analyze coverage
  sanctioned_within_10 = 0
  total_violations = len(violations_detected)

  for agent_id, start, sanction_frame in violations_detected:
    if sanction_frame is not None and (sanction_frame - start) <= 10:
      sanctioned_within_10 += 1

  coverage = sanctioned_within_10 / total_violations if total_violations > 0 else 0

  print(f"R2: {sanctioned_within_10}/{total_violations} violations sanctioned within 10 frames ({coverage:.1%})")
  assert coverage >= 0.80, \
      f"R2 FAILED: Coverage {coverage:.1%} < 80%"
  print(f"✓ R2 PASSED: Coverage {coverage:.1%} ≥ 80%")


def test_r3_no_dogpiling():
  """R3: No dogpiling - residents don't attempt to zap immune targets.

  Setup: All residents. Track immunity and zap attempts.
  Verify: No zap actions fired at targets with immune_ticks_remaining > 0.
  """
  # Configure environment
  config = allelopathic_harvest.get_config()
  config.normative_gate = True
  config.permitted_color_index = COLOR_RED
  config.ego_index = None
  config.episode_timesteps = 1000

  roles = ["default"] * 16
  base_env = substrate.build("allelopathic_harvest__normative", roles, config)

  # Setup
  extractor = ResidentInfoExtractor(
      num_players=16,
      permitted_color_index=config.permitted_color_index,
      startup_grey_grace=config.startup_grey_grace)

  controller = ResidentController()
  controller.reset(seed=42)

  env = ResidentWrapper(
      env=base_env,
      resident_indices=list(range(16)),
      ego_index=None,
      resident_controller=controller,
      info_extractor=extractor)

  timestep = env.reset()

  # Track dogpiling attempts
  total_sanctions = 0
  duplicate_sanctions = 0  # Sanctions on already-immune targets

  for step in range(1000):
    events = env.events()

    # Check for sanctions and track if target was already immune
    for event in events:
      if event.get('name') == 'sanction':
        total_sanctions += 1
        was_immune = event.get('immune', False)
        if was_immune:
          duplicate_sanctions += 1

    timestep = env.step(ego_action=None)
    if timestep.last():
      break

  print(f"R3: {total_sanctions} total sanctions, {duplicate_sanctions} on immune targets")
  assert duplicate_sanctions == 0, \
      f"R3 FAILED: {duplicate_sanctions} sanctions on immune targets (dogpiling)"
  print(f"✓ R3 PASSED: No dogpiling detected")


def test_r4_plant_harvest_purity():
  """R4: Plant/harvest purity - 100% plant permitted, ≥95% harvest permitted.

  Setup: All residents. Track plant and harvest actions.
  Verify: All plant actions are permitted color. ≥95% of harvests are permitted color.
  """
  # Configure environment
  config = allelopathic_harvest.get_config()
  config.normative_gate = True
  config.permitted_color_index = COLOR_RED
  config.ego_index = None
  config.episode_timesteps = 1000

  roles = ["default"] * 16
  base_env = substrate.build("allelopathic_harvest__normative", roles, config)

  # Setup
  extractor = ResidentInfoExtractor(
      num_players=16,
      permitted_color_index=config.permitted_color_index,
      startup_grey_grace=config.startup_grey_grace)

  controller = ResidentController()
  controller.reset(seed=42)

  env = ResidentWrapper(
      env=base_env,
      resident_indices=list(range(16)),
      ego_index=None,
      resident_controller=controller,
      info_extractor=extractor)

  timestep = env.reset()

  # Track plant and harvest actions via events
  plant_permitted = 0
  plant_other = 0
  harvest_permitted = 0
  harvest_other = 0

  for step in range(1000):
    events = env.events()

    for event in events:
      event_name = event.get('name', '')

      # Track berry planting via 'replanting' event
      if event_name == 'replanting':
        target_berry = event.get('target_berry', 0)
        if target_berry == config.permitted_color_index:
          plant_permitted += 1
        else:
          plant_other += 1

      # Track berry consumption via 'eating' event
      elif event_name == 'eating':
        berry_color = event.get('berry_color', 0)
        if berry_color == config.permitted_color_index:
          harvest_permitted += 1
        else:
          harvest_other += 1

    timestep = env.step(ego_action=None)
    if timestep.last():
      break

  # Verify purity
  total_plants = plant_permitted + plant_other
  total_harvests = harvest_permitted + harvest_other

  plant_purity = plant_permitted / total_plants if total_plants > 0 else 1.0
  harvest_purity = harvest_permitted / total_harvests if total_harvests > 0 else 1.0

  print(f"R4: Plant purity: {plant_permitted}/{total_plants} = {plant_purity:.1%}")
  print(f"R4: Harvest purity: {harvest_permitted}/{total_harvests} = {harvest_purity:.1%}")

  assert plant_purity == 1.0, \
      f"R4 FAILED: Plant purity {plant_purity:.1%} < 100%"
  assert harvest_purity >= 0.95, \
      f"R4 FAILED: Harvest purity {harvest_purity:.1%} < 95%"
  print(f"✓ R4 PASSED: Plant purity 100%, harvest purity {harvest_purity:.1%}")


def test_r5_arm_invariance():
  """R5: Arm invariance - identical resident decisions in control vs treatment.

  Setup: Two envs - control and treatment. Same seed. Track resident actions.
  Verify: Resident actions are identical across conditions.
  """
  # Configure control environment
  config_control = allelopathic_harvest.get_config()
  config_control.normative_gate = True
  config_control.permitted_color_index = COLOR_RED
  config_control.ego_index = None  # All residents
  config_control.enable_treatment_condition = False  # Control
  config_control.episode_timesteps = 500

  roles = ["default"] * 16
  env_control = substrate.build("allelopathic_harvest__normative", roles, config_control)

  # Configure treatment environment
  config_treatment = allelopathic_harvest.get_config()
  config_treatment.normative_gate = True
  config_treatment.permitted_color_index = COLOR_RED
  config_treatment.ego_index = None  # All residents
  config_treatment.enable_treatment_condition = True  # Treatment
  config_treatment.altar_coords = (5, 15)
  config_treatment.episode_timesteps = 500

  env_treatment = substrate.build("allelopathic_harvest__normative", roles, config_treatment)

  # Setup identical controllers (same seed!)
  extractor_control = ResidentInfoExtractor(16, COLOR_RED, config_control.startup_grey_grace)
  extractor_treatment = ResidentInfoExtractor(16, COLOR_RED, config_treatment.startup_grey_grace)

  controller_control = ResidentController()
  controller_control.reset(seed=42)

  controller_treatment = ResidentController()
  controller_treatment.reset(seed=42)

  wrapper_control = ResidentWrapper(env_control, list(range(16)), None, controller_control, extractor_control)
  wrapper_treatment = ResidentWrapper(env_treatment, list(range(16)), None, controller_treatment, extractor_treatment)

  # Run both environments in parallel
  ts_control = wrapper_control.reset()
  ts_treatment = wrapper_treatment.reset()

  action_differences = 0
  total_steps = 0

  for step in range(500):
    # Get actions from both controllers for all agents
    info_control = extractor_control.extract_info(ts_control.observation, wrapper_control.events())
    info_treatment = extractor_treatment.extract_info(ts_treatment.observation, wrapper_treatment.events())

    # Compare actions for each resident
    for agent_id in range(16):
      action_control = controller_control.act(agent_id, info_control)
      action_treatment = controller_treatment.act(agent_id, info_treatment)

      if action_control != action_treatment:
        action_differences += 1

    total_steps += 16  # 16 agents

    ts_control = wrapper_control.step(ego_action=None)
    ts_treatment = wrapper_treatment.step(ego_action=None)

    if ts_control.last() or ts_treatment.last():
      break

  print(f"R5: {action_differences} action differences out of {total_steps} total actions")
  assert action_differences == 0, \
      f"R5 FAILED: Residents made different decisions across control/treatment"
  print(f"✓ R5 PASSED: Arm invariance verified")


def test_r6_same_step_tie_break():
  """R6: Same-step tie-break - only one -10 lands when multiple zaps hit same target.

  Setup: All residents. Track sanctions and rewards.
  Verify: At most one -10 per target per frame (via reward tracking).
  """
  # This is enforced by the SameStepSanctionTracker in Lua
  # We verify by checking that no agent receives < -10 in a single frame
  # (which would indicate multiple -10s landed)

  config = allelopathic_harvest.get_config()
  config.normative_gate = True
  config.permitted_color_index = COLOR_RED
  config.ego_index = None
  config.episode_timesteps = 1000

  roles = ["default"] * 16
  base_env = substrate.build("allelopathic_harvest__normative", roles, config)

  extractor = ResidentInfoExtractor(16, COLOR_RED, config.startup_grey_grace)
  controller = ResidentController()
  controller.reset(seed=42)

  env = ResidentWrapper(base_env, list(range(16)), None, controller, extractor)

  timestep = env.reset()

  # Track rewards per step
  max_single_step_penalty = 0.0

  for step in range(1000):
    step_rewards = timestep.reward if hasattr(timestep, 'reward') else [0] * 16

    for reward in step_rewards:
      if reward < max_single_step_penalty:
        max_single_step_penalty = reward

    timestep = env.step(ego_action=None)
    if timestep.last():
      break

  print(f"R6: Max single-step penalty observed: {max_single_step_penalty}")

  # If tie-break works, worst case is -10 (one sanction)
  # If tie-break fails, we'd see -20, -30, etc.
  assert max_single_step_penalty >= -10.5, \
      f"R6 FAILED: Observed penalty {max_single_step_penalty} < -10 (dogpiling detected)"
  print(f"✓ R6 PASSED: Tie-break enforced (max penalty ≥ -10)")


def test_r7_no_hidden_dependencies():
  """R7: No hidden dependencies - grep for freeze/removal references.

  Verify: Code doesn't contain references to removed features (freeze, removal).
  """
  import subprocess

  # Grep for freeze/removal in components.lua
  result = subprocess.run(
      ['grep', '-i', 'freeze\\|removal',
       'meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua'],
      capture_output=True, text=True)

  if result.returncode == 0:
    # Found matches
    print(f"R7 FAILED: Found freeze/removal references:\n{result.stdout}")
    assert False, "R7 FAILED: Code contains freeze/removal references"
  else:
    print(f"✓ R7 PASSED: No freeze/removal references found")


def test_r8_monoculture_achievement(render_video: bool = False, video_path: str = "test_r8_monoculture.mp4"):
  """R8: Monoculture achievement - ≥85% permitted berries at t=2000.

  Setup: All residents. Full 2000-step episode.
  Verify: At end of episode, ≥85% of berries are permitted color.

  Args:
    render_video: If True, render episode to video file.
    video_path: Path to save video file.
  """
  config = allelopathic_harvest.get_config()
  config.normative_gate = True
  config.permitted_color_index = COLOR_RED
  config.ego_index = None
  config.episode_timesteps = 2000  # Full episode

  roles = ["default"] * 16
  base_env = substrate.build("allelopathic_harvest__normative", roles, config)

  extractor = ResidentInfoExtractor(16, COLOR_RED, config.startup_grey_grace)
  controller = ResidentController()
  controller.reset(seed=42)

  env = ResidentWrapper(base_env, list(range(16)), None, controller, extractor)

  timestep = env.reset()

  # Collect frames if rendering
  frames = []
  if render_video:
    rgb_frame = base_env.observation()[0]["WORLD.RGB"]
    frames.append(rgb_frame)

  # Track monoculture progress over time
  monoculture_history = []
  sample_interval = 100  # Sample every 100 steps

  # Run full episode
  for step in range(2000):
    timestep = env.step(ego_action=None)

    if render_video:
      rgb_frame = base_env.observation()[0]["WORLD.RGB"]
      frames.append(rgb_frame)

    # Sample monoculture fraction periodically
    if step % sample_interval == 0:
      obs = timestep.observation[0]
      berries_by_type = obs.get('BERRIES_BY_TYPE', None)
      if berries_by_type is not None:
        permitted_idx = config.permitted_color_index - 1
        permitted_berries = berries_by_type[permitted_idx]
        total_berries = sum(berries_by_type)
        mono_frac = permitted_berries / total_berries if total_berries > 0 else 0.0
        monoculture_history.append((step, mono_frac))

    if timestep.last():
      break

  # Get final berry counts from GlobalBerryTracker observations
  final_obs = timestep.observation[0]
  berries_by_type = final_obs.get('BERRIES_BY_TYPE', None)

  if berries_by_type is None:
    raise RuntimeError("BERRIES_BY_TYPE observation not found")

  # Convert permitted_color_index (1=RED, 2=GREEN, 3=BLUE) to 0-indexed
  permitted_idx = config.permitted_color_index - 1

  permitted_berries = berries_by_type[permitted_idx]
  total_berries = sum(berries_by_type)

  monoculture_fraction = permitted_berries / total_berries if total_berries > 0 else 0.0

  # Print monoculture progression
  print(f"R8: Monoculture progression:")
  for step, mono_frac in monoculture_history:
    print(f"  Step {step:4d}: {mono_frac:.1%}")

  print(f"\nR8: Final berry counts by color: RED={berries_by_type[0]}, GREEN={berries_by_type[1]}, BLUE={berries_by_type[2]}")
  print(f"R8: Final monoculture: {permitted_berries}/{total_berries} = {monoculture_fraction:.1%}")

  # Save video if requested
  if render_video:
    _save_video_r8(frames, video_path, fps=8)
    print(f"R8: Video saved to {video_path} ({len(frames)} frames)")

  assert monoculture_fraction >= 0.85, \
      f"R8 FAILED: Monoculture {monoculture_fraction:.1%} < 85%"
  print(f"✓ R8 PASSED: Monoculture achieved ({monoculture_fraction:.1%} ≥ 85%)")


def _save_video_r8(frames, output_path: str, fps: int):
  """Save frames as video file for R8 visualization.

  Args:
    frames: List of RGB frames (numpy arrays).
    output_path: Path to save video.
    fps: Frames per second.
  """
  try:
    import cv2
    import numpy as np

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
      # Convert RGB to BGR for OpenCV
      bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      out.write(bgr_frame)

    out.release()
    print(f"Video saved successfully: {output_path}")
  except ImportError:
    print("WARNING: OpenCV not available. Install with: pip install opencv-python")
    # Fallback: save as numpy array
    import numpy as np
    np.save(output_path.replace('.mp4', '.npy'), np.array(frames))
    print(f"Saved frames as numpy array instead: {output_path.replace('.mp4', '.npy')}")


if __name__ == "__main__":
  print("Running Phase 2 Resident Acceptance Tests...")
  print()

  print("=" * 60)
  print("R1: Selectivity Test")
  print("=" * 60)
  test_r1_selectivity()
  print()

  print("=" * 60)
  print("R2: Coverage Test")
  print("=" * 60)
  test_r2_coverage()
  print()

  print("=" * 60)
  print("R3: No Dogpiling Test")
  print("=" * 60)
  test_r3_no_dogpiling()
  print()

  print("=" * 60)
  print("R4: Plant/Harvest Purity Test")
  print("=" * 60)
  test_r4_plant_harvest_purity()
  print()

  print("=" * 60)
  print("R5: Arm Invariance Test")
  print("=" * 60)
  test_r5_arm_invariance()
  print()

  print("=" * 60)
  print("R6: Same-Step Tie-Break Test")
  print("=" * 60)
  test_r6_same_step_tie_break()
  print()

  print("=" * 60)
  print("R7: No Hidden Dependencies Test")
  print("=" * 60)
  test_r7_no_hidden_dependencies()
  print()

  print("=" * 60)
  print("R8: Monoculture Achievement Test")
  print("=" * 60)
  # Enable video rendering for R8 (can be disabled by passing render_video=False)
  test_r8_monoculture_achievement(render_video=True, video_path="test_r8_monoculture.mp4")
  print()

  print("=" * 60)
  print("✓✓✓ All Phase 2 acceptance tests PASSED! ✓✓✓")
  print("=" * 60)
  print()
  print("Summary:")
  print("  R1: Selectivity - Residents never mis-zap compliant agents")
  print("  R2: Coverage - ≥80% of violators sanctioned within 10 frames")
  print("  R3: No Dogpiling - No sanctions on immune targets")
  print("  R4: Purity - 100% plant permitted, ≥95% harvest permitted")
  print("  R5: Arm Invariance - Identical actions in control vs treatment")
  print("  R6: Tie-Break - Max one -10 per target per frame")
  print("  R7: No Dependencies - No freeze/removal references")
  print("  R8: Monoculture - ≥85% permitted berries at episode end")
  print()
  print("Video output: test_r8_monoculture.mp4")
