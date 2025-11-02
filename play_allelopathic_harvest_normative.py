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

# Added by RST: Track state for verbose printing
_last_reward = {}
_last_ready_to_shoot = {}


def verbose_fn(timestep, player_index, current_player_index):
    """Added by RST: Print useful info when actions happen."""
    global _last_reward, _last_ready_to_shoot

    # Only print for the currently controlled player
    if player_index != current_player_index:
        return

    lua_index = player_index + 1

    # Initialize tracking if needed
    if not hasattr(verbose_fn, 'last_player'):
        verbose_fn.last_player = -1

    # Print header when switching players
    if verbose_fn.last_player != current_player_index:
        verbose_fn.last_player = current_player_index
        altar_obs_key = f'{lua_index}.ALTAR'
        if altar_obs_key in timestep.observation:
            altar_color_id = int(timestep.observation[altar_obs_key])
            altar_color_name = COLOR_NAMES.get(altar_color_id, f"UNKNOWN({altar_color_id})")
            print(f"\n{'='*60}")
            print(f">>> CONTROLLING PLAYER {player_index} <<<")
            print(f">>> Altar Color (Permitted): {altar_color_name} ({altar_color_id}) <<<")
            print(f"{'='*60}\n")
        _last_reward[player_index] = 0
        _last_ready_to_shoot[player_index] = None

    # Check for reward changes (indicates action happened)
    reward_key = f'{lua_index}.REWARD'
    if reward_key in timestep.observation:
        current_reward = timestep.observation[reward_key]
        if player_index not in _last_reward:
            _last_reward[player_index] = current_reward

        reward_delta = current_reward - _last_reward[player_index]
        if reward_delta != 0:
            print(f"[Player {player_index}] Reward: {reward_delta:+.1f} (Total: {current_reward:.1f})")
            _last_reward[player_index] = current_reward

    # Check if zap was fired (ready_to_shoot changes)
    ready_key = f'{lua_index}.READY_TO_SHOOT'
    if ready_key in timestep.observation:
        ready = bool(timestep.observation[ready_key])
        if player_index in _last_ready_to_shoot:
            if _last_ready_to_shoot[player_index] and not ready:
                print(f"[Player {player_index}] 🔫 ZAP FIRED!")
        _last_ready_to_shoot[player_index] = ready


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
    # Added by RST: Print events shows sanction events
    parser.add_argument(
        '--print_events', type=bool, default=True, help='Print all events (raw)')
    # Added by RST: Screen size for better visibility
    parser.add_argument(
        '--screen_width', type=int, default=1200, help='Screen width in pixels')
    parser.add_argument(
        '--screen_height', type=int, default=900, help='Screen height in pixels')

    args = parser.parse_args()
    env_module = environment_configs[args.level_name]
    env_config = env_module.get_config()

    with config_dict.ConfigDict(env_config).unlocked() as env_config:
        roles = env_config.default_player_roles
        env_config.lab2d_settings = env_module.build(roles, env_config)

    # Added by RST: Print controls at startup
    print("\n" + "="*80)
    print("ALLELOPATHIC HARVEST NORMATIVE - INTERACTIVE PLAY")
    print("="*80)
    print("\n⚠️  IMPORTANT: Click on the pygame window to give it focus for keyboard input!")
    print("\nKEY CONTROLS:")
    print("  Movement:  W (forward), A (left), S (backward), D (right)")
    print("             Arrow keys also work for movement")
    print("  Turn:      Q (turn left), E (turn right)")
    print("  Actions:   SPACE (zap/sanction another player)")
    print("             1 (plant RED berry - changes your body color)")
    print("             2 (plant GREEN berry - changes your body color)")
    print("             3 (plant BLUE berry - changes your body color)")
    print("  Switch:    TAB (switch between the 16 players)")
    print("\nGAME MECHANICS:")
    print("  - Altar Color: The permitted body color (shown when you switch players)")
    print("  - Violation: Having body_color != altar_color (after grace period)")
    print("  - Planting berries changes your body color to that berry color")
    print("  - Sanctions apply -10 to target, -0.2 cost to you")
    print("  - Correct sanctions (target violating): +0.5, Incorrect: -0.5")
    print("  - First 25 frames: Grace period (sanctions fizzle, no penalties)")
    print("  - Immunity: 200 frames after being sanctioned (or until color change)")
    print("  - Zap range: 3 cells, cooldown: 4 frames")
    print("\n📊 FEEDBACK:")
    print("  - Console shows: Rewards when they change, Zap events, Sanction events")
    print("  - Look for 'ZAP FIRED!' message when you press SPACE")
    print("  - Sanction events show zapper/zappee and whether it was applied")
    print("="*80 + "\n")

    level_playing_utils.run_episode(
        args.observation, args.settings, _ACTION_MAP,
        env_config, level_playing_utils.RenderType.PYGAME,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        verbose_fn=verbose_fn if args.verbose else None,
        print_events=args.print_events)


if __name__ == '__main__':
    main()
