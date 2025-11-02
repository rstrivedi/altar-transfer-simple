#!/usr/bin/env python3
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple human player for testing `allelopathic_harvest_normative`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to fire the zapper.
Use `1, 2, 3` to plant RED, GREEN, BLUE berries.
Use `TAB` to switch between players.

Press --verbose=True to see altar color and rewards.
Press --print_events=True to see sanction events.
"""

import argparse
import json

from meltingpot.configs.substrates import allelopathic_harvest_normative__open
from meltingpot.human_players import level_playing_utils
from meltingpot.utils.substrates import builder
from ml_collections import config_dict

environment_configs = {
    'allelopathic_harvest_normative__open': allelopathic_harvest_normative__open,
}

_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'fireZap': level_playing_utils.get_space_key_pressed,
    'fire_1': level_playing_utils.get_key_number_one_pressed,
    'fire_2': level_playing_utils.get_key_number_two_pressed,
    'fire_3': level_playing_utils.get_key_number_three_pressed,
}

# Added by RST: Color names for display
COLOR_NAMES = {
    0: "GREY",
    1: "RED",
    2: "GREEN",
    3: "BLUE",
}

# Added by RST: Track last state to detect changes
_last_player = -1
_last_reward = {}
_current_controlled_player = 0
_env_ref = None  # Store env reference for event access


def format_event(event, controlled_player):
  """Added by RST: Format events in human-readable way, filter to controlled player."""
  event_name = event[0].decode() if isinstance(event[0], bytes) else event[0]
  event_data = {}

  # Parse event data (it's a list like [b'dict', b'key1', value1, b'key2', value2, ...])
  data_list = event[1]
  i = 1  # Skip first element which is b'dict'
  while i < len(data_list):
    key = data_list[i].decode() if isinstance(data_list[i], bytes) else data_list[i]
    value = data_list[i + 1]
    # Convert numpy arrays to scalars
    if hasattr(value, 'item'):
      value = value.item()
    event_data[key] = value
    i += 2

  # Filter: Only show events involving controlled player
  if event_name == 'sanction':
    zapper_id = int(event_data.get('zapper_id', -1))
    zappee_id = int(event_data.get('zappee_id', -1))
    if zapper_id != controlled_player and zappee_id != controlled_player:
      return None  # Not involving controlled player, skip

    # Format sanction event nicely
    t = int(event_data.get('t', 0))
    zappee_color = int(event_data.get('zappee_color', 0))
    was_violation = int(event_data.get('was_violation', 0))
    applied = int(event_data.get('applied_minus10', 0))
    immune = int(event_data.get('immune', 0))
    tie_break = int(event_data.get('tie_break', 0))

    color_name = COLOR_NAMES.get(zappee_color, 'UNKNOWN')

    if zapper_id == controlled_player:
      # You zapped someone
      if immune:
        result = "TARGET WAS IMMUNE → fizzled (no effect)"
      elif tie_break:
        result = "TIE-BREAK → fizzled (someone else zapped them first this frame)"
      elif applied:
        result = "APPLIED -10 PENALTY to target"
      else:
        result = "GRACE PERIOD → fizzled (first 25 frames)"

      violation_str = "target in VIOLATION" if was_violation else "target COMPLIANT"
      return f"⚡ [t={t}] YOU ZAPPED P{zappee_id} ({color_name}) | {violation_str} | {result}"
    else:
      # You were zapped
      if immune:
        result = "You were IMMUNE → no damage"
      elif applied:
        result = "YOU TOOK -10 PENALTY"
      else:
        result = "Grace period → no damage"

      violation_str = "You were VIOLATING" if was_violation else "You were COMPLIANT"
      return f"⚡ [t={t}] P{zapper_id} ZAPPED YOU ({color_name}) | {violation_str} | {result}"

  # Suppress other events (eating, replanting, zap) - too noisy
  # Only showing sanction events for clarity
  return None


def verbose_fn(timestep, player_index, current_player_index):
  """Added by RST: Print altar color on player switch. Sanction events shown separately."""
  global _last_player, _last_reward, _current_controlled_player

  # Track current controlled player for event filtering
  _current_controlled_player = current_player_index

  # Only print for currently controlled player
  if player_index != current_player_index:
    return

  lua_index = player_index + 1

  # Print altar color when switching players
  if _last_player != current_player_index:
    _last_player = current_player_index
    altar_key = f'{lua_index}.ALTAR'
    if altar_key in timestep.observation:
      altar_id = int(timestep.observation[altar_key])
      altar_name = COLOR_NAMES.get(altar_id, f"UNKNOWN({altar_id})")
      print(f"\n{'='*60}")
      print(f">>> PLAYER {player_index} | Altar (Permitted): {altar_name} <<<")
      print(f"{'='*60}")

    # Initialize reward tracking
    reward_key = f'{lua_index}.REWARD'
    if reward_key in timestep.observation:
      _last_reward[player_index] = timestep.observation[reward_key]

  # Track reward changes but only print sanction-related ones
  reward_key = f'{lua_index}.REWARD'
  if reward_key in timestep.observation:
    current_reward = timestep.observation[reward_key]
    if player_index not in _last_reward:
      _last_reward[player_index] = current_reward

    reward_delta = current_reward - _last_reward[player_index]
    if reward_delta != 0:
      # Only print sanction-related rewards (suppress berry eating noise)
      is_sanction = False
      breakdown = ""

      if abs(reward_delta - (-0.2)) < 0.01:
        breakdown = "Zap cost -0.2 (fizzled or grace period)"
        is_sanction = True
      elif abs(reward_delta - 0.3) < 0.01:
        breakdown = "✓ Correct sanction: -0.2 (cost) +0.5 (alpha) = +0.3"
        is_sanction = True
      elif abs(reward_delta - (-0.7)) < 0.01:
        breakdown = "✗ Incorrect sanction: -0.2 (cost) -0.5 (beta) = -0.7"
        is_sanction = True
      elif abs(reward_delta - (-10)) < 0.01:
        breakdown = "⚡ YOU WERE SANCTIONED: -10.0 penalty"
        is_sanction = True
      elif abs(reward_delta - (-10.2)) < 0.01:
        breakdown = "⚡ SANCTIONED while zapping: -10.0 (penalty) -0.2 (zap cost)"
        is_sanction = True
      elif abs(reward_delta - (-9.7)) < 0.01:
        breakdown = "⚡ SANCTIONED but your zap landed: -10.0 (penalty) +0.3 (your sanction)"
        is_sanction = True

      # Only print if it's sanction-related
      if is_sanction:
        print(f"\n💰 {breakdown}")
        print(f"   Total reward: {current_reward:.1f}")

      _last_reward[player_index] = current_reward


def print_formatted_events(env):
  """Added by RST: Print formatted events for controlled player only."""
  if not hasattr(env, 'events'):
    print("[DEBUG] env has no 'events' method")
    return

  events = env.events()
  if not events:
    return  # No events this timestep

  # Filter and format events for controlled player only
  sanction_count = 0
  for event in events:
    event_name = event[0].decode() if isinstance(event[0], bytes) else event[0]
    if event_name == 'sanction':
      sanction_count += 1
      print(f"[DEBUG] Found sanction event, controlled player: {_current_controlled_player}")
      formatted = format_event(event, _current_controlled_player)
      print(f"[DEBUG] Formatted result: {formatted}")
      if formatted:
        print(formatted)
        print()  # Blank line for readability


class VerboseWithEvents:
  """Added by RST: Wrapper that combines verbose output with formatted event printing.

  This class monkey-patches level_playing_utils to add custom event printing.
  """

  def __init__(self, show_events=True):
    self.show_events = show_events
    self.env = None

  def __call__(self, timestep, player_index, current_player_index):
    """Called by level_playing_utils as verbose_fn."""
    # Call regular verbose function
    verbose_fn(timestep, player_index, current_player_index)

    # Print formatted events after verbose output (only for controlled player)
    # ONLY call once per timestep by checking if it's the last player in the loop
    if self.show_events and self.env and player_index == current_player_index:
      print(f"[DEBUG] Calling print_formatted_events, env={self.env is not None}")
      print_formatted_events(self.env)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='allelopathic_harvest_normative__open',
      choices=environment_configs.keys(),
      help='Level name to load')
  parser.add_argument(
      '--observation', type=str, default='RGB', help='Observation to render')
  parser.add_argument(
      '--settings', type=json.loads, default={}, help='Settings as JSON string')
  # Added by RST: Enable verbose by default for normative testing
  parser.add_argument(
      '--verbose', type=bool, default=True, help='Print altar color and rewards')
  # Added by RST: Raw events for debugging (disabled by default, very verbose)
  parser.add_argument(
      '--raw_events', type=bool, default=False, help='Print raw events for all players (debug only)')

  args = parser.parse_args()
  env_module = environment_configs[args.level_name]
  env_config = env_module.get_config()

  # Added by RST: Print startup info
  print("\n" + "="*70)
  print("NORMATIVE SANCTIONING TEST - Allelopathic Harvest")
  print("="*70)
  print("\nCONTROLS:")
  print("  WASD or Arrow Keys - Move")
  print("  Q/E - Turn left/right")
  print("  SPACE - Zap (sanction)")
  print("  1/2/3 - Plant RED/GREEN/BLUE berry (changes body color)")
  print("  TAB - Switch players")
  print("\nMECHANICS:")
  print("  - Altar color = permitted body color")
  print("  - Violation = body_color ≠ altar_color")
  print("  - Grace period: First 25 frames (~3 sec, sanctions fizzle)")
  print("  - Immunity: 200 frames (~20 sec) after sanction OR until color change")
  print("\nREWARDS:")
  print("  - Correct sanction (violation):  -0.2 (cost) + 0.5 (alpha) = +0.3")
  print("  - Incorrect sanction (no viol.): -0.2 (cost) - 0.5 (beta)  = -0.7")
  print("  - Fizzled zap (grace/immune):    -0.2 (cost only)")
  print("  - Target penalty (sanctioned):   -10.0")
  print("  - Tasty berry:                   +2.0")
  print("\nOUTPUT (Clean - only sanction events shown):")
  print("  - Press TAB to see altar color for each player")
  print("  - Sanction events show: who zapped whom, violation status, result")
  print("  - Sanction rewards shown with breakdown")
  print("  - Berry eating NOT shown (reduces noise)")
  print("  - Use --raw_events=True for all raw events (debug only)")
  print("="*70 + "\n")

  with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(roles, env_config)

  # Added by RST: Create verbose function with event support
  verbose_with_events = VerboseWithEvents(show_events=args.verbose)
  print(f"[DEBUG] Created VerboseWithEvents wrapper, show_events={args.verbose}")

  # Added by RST: Monkey-patch to inject env reference after creation
  # We'll wrap the env_builder to capture the env
  original_builder_fn = builder.builder

  def builder_with_capture(*args_list, **kwargs):
    env = original_builder_fn(*args_list, **kwargs)
    verbose_with_events.env = env  # Capture env reference
    print(f"[DEBUG] Captured env reference: {env}")
    return env

  # Temporarily replace builder
  builder.builder = builder_with_capture

  try:
    level_playing_utils.run_episode(
        args.observation, args.settings, _ACTION_MAP,
        env_config, level_playing_utils.RenderType.PYGAME,
        verbose_fn=verbose_with_events if args.verbose else None,
        print_events=args.raw_events)  # Disabled by default, use --raw_events=True for debugging
  finally:
    # Restore original builder
    builder.builder = original_builder_fn


if __name__ == '__main__':
    main()
