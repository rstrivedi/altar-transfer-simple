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
Use `1, 2, 3` to plant RED, GREEN, BLUE berries respectively.
Use `TAB` to switch between players.

This version prints detailed information about normative mechanics:
- Altar color (permitted color)
- Ready to shoot status
Use --print_events=True to see sanction events with full details.
"""

import argparse
import json

# Added by RST: Import normative substrate instead of base
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


def verbose_fn(timestep, player_index, current_player_index):
    """Added by RST: Print ALTAR observation for current player."""
    # Only print for the currently controlled player
    if player_index != current_player_index:
        return

    lua_index = player_index + 1

    # Get ALTAR observation (scalar: 1=RED, 2=GREEN, 3=BLUE)
    altar_obs_key = f'{lua_index}.ALTAR'
    if altar_obs_key in timestep.observation:
        altar_color_id = int(timestep.observation[altar_obs_key])
        altar_color_name = COLOR_NAMES.get(altar_color_id, f"UNKNOWN({altar_color_id})")
        print(f"Player {player_index} | Altar Color: {altar_color_name} ({altar_color_id})", end="")

    # Get READY_TO_SHOOT observation
    ready_obs_key = f'{lua_index}.READY_TO_SHOOT'
    if ready_obs_key in timestep.observation:
        ready_to_shoot = timestep.observation[ready_obs_key]
        print(f" | Ready to Zap: {bool(ready_to_shoot)}")
    else:
        print()  # newline


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
    # Added by RST: Default verbose to True for normative testing
    parser.add_argument(
        '--verbose', type=bool, default=True, help='Print debug information')
    # Added by RST: Default print_events to True to see sanction events
    parser.add_argument(
        '--print_events', type=bool, default=True, help='Print all events (raw)')

    args = parser.parse_args()
    env_module = environment_configs[args.level_name]
    env_config = env_module.get_config()

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
