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
"""Configuration for Allelopathic Harvest Normative (open).

Added by RST: Normative variant with institutional sanctioning mechanics.

This substrate extends allelopathic harvest with normative sanctioning:
- Violation = body_color != permitted_color (after grace period)
- Sanctioning via zap beams with rewards (c, alpha, beta)
- Immunity system and tie-break mechanism
- PERMITTED_COLOR observation for residents

Based on allelopathic harvest. See allelopathic_harvest_normative.py for details.
"""

from meltingpot.configs.substrates import allelopathic_harvest_normative as base_config

OPEN_ASCII_MAP = """
333PPPP12PPP322P32PPP1P13P3P3
1PPPP2PP122PPP3P232121P2PP2P1
P1P3P11PPP13PPP31PPPP23PPPPPP
PPPPP2P2P1P2P3P33P23PP2P2PPPP
P1PPPPPPP2PPP12311PP3321PPPPP
133P2PP2PPP3PPP1PPP2213P112P1
3PPPPPPPPPPPPP31PPPPPP1P3112P
PP2P21P21P33PPPPPPP3PP2PPPP1P
PPPPP1P1P32P3PPP22PP1P2PPPP2P
PPP3PP3122211PPP2113P3PPP1332
PP12132PP1PP1P321PP1PPPPPP1P3
PPP222P12PPPP1PPPP1PPP321P11P
PPP2PPPP3P2P1PPP1P23322PP1P13
23PPP2PPPP2P3PPPP3PP3PPP3PPP2
2PPPP3P3P3PP3PP3P1P3PP11P21P1
21PPP2PP331PP3PPP2PPPPP2PP3PP
P32P2PP2P1PPPPPPP12P2PPP1PPPP
P3PP3P2P21P3PP2PP11PP1323P312
2P1PPPPP1PPP1P2PPP3P32P2P331P
PPPPP1312P3P2PPPP3P32PPPP2P11
P3PPPP221PPP2PPPPPPPP1PPP311P
32P3PPPPPPPPPP31PPPP3PPP13PPP
PPP3PPPPP3PPPPPP232P13PPPPP1P
P1PP1PPP2PP3PPPPP33321PP2P3PP
P13PPPP1P333PPPP2PP213PP2P3PP
1PPPPP3PP2P1PP21P3PPPP231P2PP
1331P2P12P2PPPP2PPP3P23P21PPP
P3P131P3PPP13P1PPP222PPPP11PP
2P3PPPPPPPP2P323PPP2PPP1PPP2P
21PPPPPPP12P23P1PPPPPP13P3P11
"""

build = base_config.build


def get_config():
  """Adjust default configuration."""
  config = base_config.get_config()
  config.ascii_map = OPEN_ASCII_MAP

  config.default_player_roles = (
      ("player_who_likes_red",) * 8 + ("player_who_likes_green",) * 8)

  return config
