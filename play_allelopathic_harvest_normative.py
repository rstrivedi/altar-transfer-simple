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


def verbose_fn(timestep, player_index, current_player_index):
  """Added by RST: Print altar color on player switch and rewards on change."""
  global _last_player, _last_reward

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
      print(f"\n>>> PLAYER {player_index} | Altar (Permitted): {altar_name} <<<")

    # Initialize reward tracking
    reward_key = f'{lua_index}.REWARD'
    if reward_key in timestep.observation:
      _last_reward[player_index] = timestep.observation[reward_key]

  # Print reward changes
  reward_key = f'{lua_index}.REWARD'
  if reward_key in timestep.observation:
    current_reward = timestep.observation[reward_key]
    if player_index not in _last_reward:
      _last_reward[player_index] = current_reward

    reward_delta = current_reward - _last_reward[player_index]
    if reward_delta != 0:
      print(f"[Player {player_index}] Reward: {reward_delta:+.1f} (Total: {current_reward:.1f})")
      _last_reward[player_index] = current_reward


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
  # Added by RST: Enable events by default to see sanction details
  parser.add_argument(
      '--print_events', type=bool, default=True, help='Print sanction events')

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
  print("  - Grace period: First 25 frames (sanctions fizzle)")
  print("  - Immunity: 200 frames after sanction OR until color change")
  print("  - Rewards: c=-0.2, target=-10, alpha=+0.5, beta=-0.5")
  print("\nOUTPUT:")
  print("  - Verbose: Altar color (on TAB), Rewards (on change)")
  print("  - Events: Sanction attempts with flags (was_violation, applied_minus10, immune, tie_break)")
  print("="*70 + "\n")

  with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(roles, env_config)
  level_playing_utils.run_episode(
      args.observation, args.settings, _ACTION_MAP,
      env_config, level_playing_utils.RenderType.PYGAME,
      verbose_fn=verbose_fn if args.verbose else None,
      print_events=args.print_events)


if __name__ == '__main__':
    main()
