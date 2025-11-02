# NEW Resident Implementation - Based on Code Study

## Summary of Findings

After studying the altar-transfer codebase and current altar-transfer-simple implementation, here's how residents work:

### Architecture (EXISTING)
```
dmlab2d Environment (Lua)
  ↓ timestep.observation (dict)
  ↓ env.events() (list of events)
  ↓
ResidentWrapper
  ↓ calls ResidentInfoExtractor.extract_info(observations, events)
  ↓ returns info dict
  ↓
ResidentController.act(agent_id, info)
  ↓
returns action (int)
```

### What Residents Receive

**Observations** (from timestep.observation list):
- `RGB` - Egocentric visual view
- `READY_TO_SHOOT` - Zap cooldown status (1.0 = ready, 0.0 = cooling)
- `ALTAR` - Permitted color (1=RED, 2=GREEN, 3=BLUE)

**Events** (from env.events()):
- `sanction` - Zap attempts with results
- `eating` - Berry consumption
- `replanting` - Berry recoloring

### Key Insight: NO Direct Lua Component Access

From studying the code:
- Policies/Puppeteers receive `dm_env.TimeStep` which contains observations
- They do NOT have direct access to Lua simulation object
- They can only access what's in observations and events
- The dmlab2d wrapper abstracts away the Lua layer

---

## What's MISSING in Current Implementation

The RESIDENT_ANALYSIS.md identified ResidentObserver as missing. After studying the code, here's what's actually missing:

### Missing Files
1. **`agents/utils/event_parser.py`** - Parses raw dmlab2d events
2. **Observation components** - To expose agent colors and immunity status

### The REAL Problem

**Current observations DON'T include**:
- Other agents' body colors (ColorZapper.colorId)
- Other agents' immunity status (ImmunityTracker.isImmune())
- Positions of other agents
- Positions of berries

**Without this info, residents can't**:
- Detect who is violating (need to know body_color)
- Avoid sanctioning immune agents (need immunity status)
- Navigate to targets

---

## NEW Implementation Approach

### Option A: Add Observation Components (RECOMMENDED)

**Create observation components in Lua** that expose the needed information:

#### 1. ColorStateObservation Component

```lua
-- Added by RST: Exposes all agents' body colors as observation
local ColorStateObservation = class.Class(component.Component)

function ColorStateObservation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ColorStateObservation')},
  })
  ColorStateObservation.Base.__init__(self, kwargs)
end

function ColorStateObservation:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local sim = self.gameObject.simulation
  local numPlayers = sim:getNumPlayers()

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.AGENT_COLORS',
      type = 'tensor.Int32Tensor',
      shape = {numPlayers},
      func = function(grid)
        local colors = tensor.Int32Tensor(numPlayers):fill(0)
        for i = 1, numPlayers do
          local avatar = sim:getAvatarFromIndex(i)
          local colorId = avatar:getComponent('ColorZapper').colorId
          colors(i):val(colorId)
        end
        return colors
      end
  }
end
```

#### 2. ImmunityStateObservation Component

```lua
-- Added by RST: Exposes all agents' immunity status as observation
local ImmunityStateObservation = class.Class(component.Component)

function ImmunityStateObservation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ImmunityStateObservation')},
  })
  ImmunityStateObservation.Base.__init__(self, kwargs)
end

function ImmunityStateObservation:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local sim = self.gameObject.simulation
  local numPlayers = sim:getNumPlayers()

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.IMMUNITY_STATUS',
      type = 'tensor.Int32Tensor',
      shape = {numPlayers},
      func = function(grid)
        local immunity = tensor.Int32Tensor(numPlayers):fill(0)
        for i = 1, numPlayers do
          local avatar = sim:getAvatarFromIndex(i)
          local immunityObjects = avatar:getComponent('Avatar'):getAllConnectedObjectsWithNamedComponent('ImmunityTracker')
          if #immunityObjects > 0 then
            local isImmune = immunityObjects[1]:getComponent('ImmunityTracker'):isImmune()
            immunity(i):val(isImmune and 1 or 0)
          end
        end
        return immunity
      end
  }
end
```

**Add to avatar_library.lua**:
```lua
local allComponents = {
    -- ... existing components ...
    ColorStateObservation = ColorStateObservation,  -- Added by RST
    ImmunityStateObservation = ImmunityStateObservation,  -- Added by RST
}
```

**Add to substrate config** (for residents only, not ego):
```python
if player_idx in resident_indices:  # Only for residents
    avatar_object["components"].append({
        "component": "ColorStateObservation",
    })
    avatar_object["components"].append({
        "component": "ImmunityStateObservation",
    })
```

---

### Python Implementation

#### 1. Create agents/utils/event_parser.py

```python
# Added by RST: Parse dmlab2d events into Python-friendly format
"""Event parsing utilities for dmlab2d events."""

from typing import List, Dict, Any

def parse_events(raw_events) -> List[Dict[str, Any]]:
    """Parse raw dmlab2d events into Python dicts.

    dmlab2d events format:
    [
        [b'event_name', [b'dict', b'key1', value1, b'key2', value2, ...]],
        ...
    ]

    Args:
        raw_events: Raw events from env.events()

    Returns:
        List of dicts like:
        [
            {
                'name': 'sanction',
                'data': {'t': 100, 'zapper_id': 1, 'zappee_id': 2, ...}
            },
            ...
        ]
    """
    parsed = []

    for event in raw_events:
        event_name = event[0].decode() if isinstance(event[0], bytes) else event[0]
        event_data = {}

        # Parse event data (list like [b'dict', b'key1', value1, ...])
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

        parsed.append({
            'name': event_name,
            'data': event_data
        })

    return parsed
```

#### 2. Update agents/residents/info_extractor.py

```python
# Added by RST: Extract resident-specific info from observations and events
"""Extract information needed by ResidentController from observations/events."""

from typing import Dict, List, Any
import numpy as np

class ResidentInfoExtractor:
    """Extracts info dict for ResidentController from observations and events."""

    def __init__(self, num_players: int):
        """Initialize extractor.

        Args:
            num_players: Total number of players in environment.
        """
        self._num_players = num_players
        self._timestep = 0

    def reset(self):
        """Reset extractor for new episode."""
        self._timestep = 0

    def extract_info(self, observations: List[Dict], events: List[Dict]) -> Dict[str, Any]:
        """Extract info dict for all residents.

        Args:
            observations: List of observation dicts (one per player)
            events: Parsed events from this timestep

        Returns:
            Info dict with keys:
            - 'world_step': Current timestep
            - 'altar_color': Permitted color (1=RED, 2=GREEN, 3=BLUE)
            - 'residents': List of dicts (one per resident) with:
                - 'ready_to_shoot': 1.0 if can zap, 0.0 if cooling
                - 'agent_colors': Array of all agents' body colors
                - 'immunity_status': Array of all agents' immunity (1=immune, 0=not)
        """
        self._timestep += 1

        # Get altar color from first player's observation (global, same for all)
        altar_color = int(observations[0]['ALTAR'])

        # Extract per-resident info
        residents_info = []
        for player_idx in range(self._num_players):
            obs = observations[player_idx]

            resident_info = {
                'ready_to_shoot': float(obs['READY_TO_SHOOT']),
                'agent_colors': np.asarray(obs['AGENT_COLORS']),  # Shape: [num_players]
                'immunity_status': np.asarray(obs['IMMUNITY_STATUS']),  # Shape: [num_players]
            }
            residents_info.append(resident_info)

        return {
            'world_step': self._timestep,
            'altar_color': altar_color,
            'residents': residents_info,
        }
```

#### 3. Update agents/residents/scripted_residents.py

```python
# Added by RST: Scripted resident controller for normative sanctioning
"""Deterministic scripted resident controller implementing equilibrium play."""

import numpy as np
from typing import Dict, Optional
from agents.residents import config as cfg


class ResidentController:
    """Deterministic scripted controller for resident agents."""

    def __init__(self):
        """Initialize controller."""
        self._rng = None
        self._patrol_state = {}
        self._last_plant_step = {}
        self._step_count = 0

    def reset(self, seed: int = cfg.DEFAULT_SEED):
        """Reset controller for new episode."""
        self._rng = np.random.RandomState(seed)
        self._patrol_state = {}
        self._last_plant_step = {}
        self._step_count = 0

    def act(self, resident_id: int, info: Dict) -> int:
        """Select action for a resident agent.

        Priority:
        1. Maintain compliance (plant permitted color)
        2. Sanction violators (if ready and target eligible)
        3. Patrol

        Args:
            resident_id: Agent ID (0-indexed)
            info: Info dict from ResidentInfoExtractor

        Returns:
            Action index (0-10)
        """
        self._step_count += 1

        resident_info = info['residents'][resident_id]
        altar_color = info['altar_color']
        world_step = info['world_step']

        # P1: Ensure own compliance by planting permitted color
        own_color = resident_info['agent_colors'][resident_id]
        if own_color != altar_color:
            # Plant correct color to become compliant
            return cfg.PLANT_ACTION_MAP[altar_color]

        # P2: Sanction violators if ready
        if resident_info['ready_to_shoot'] > 0.5:
            zap_action = self._try_zap(resident_id, resident_info, altar_color, world_step)
            if zap_action is not None:
                return zap_action

        # P3: Plant permitted color (even if already compliant, to maintain color)
        # Frequency control: only plant every N steps
        last_plant = self._last_plant_step.get(resident_id, -999)
        if self._step_count - last_plant >= cfg.PLANT_FREQUENCY:
            self._last_plant_step[resident_id] = self._step_count
            return cfg.PLANT_ACTION_MAP[altar_color]

        # P4: Patrol
        return self._patrol(resident_id)

    def _try_zap(self, resident_id: int, resident_info: Dict, altar_color: int, world_step: int) -> Optional[int]:
        """Try to zap eligible violators.

        Args:
            resident_id: This resident's ID
            resident_info: This resident's info dict
            altar_color: Permitted color
            world_step: Current timestep

        Returns:
            ACTION_FIRE_ZAP if should zap, None otherwise
        """
        agent_colors = resident_info['agent_colors']
        immunity_status = resident_info['immunity_status']

        # Check each agent for violations
        for agent_id in range(len(agent_colors)):
            if agent_id == resident_id:
                continue  # Don't zap self

            body_color = agent_colors[agent_id]
            is_immune = (immunity_status[agent_id] == 1)

            # Check violation: grey (0) OR wrong color
            is_violating = (body_color == 0) or (body_color != altar_color)

            # Check grace period (first 50 frames)
            in_grace = (world_step < cfg.GRACE_PERIOD)

            # Eligible if violating AND not immune AND not in grace
            if is_violating and not is_immune and not in_grace:
                # TODO: Should check if in range, but for now just zap
                # (zap will fizzle if not in range, no penalty)
                return cfg.ACTION_FIRE_ZAP

        return None

    def _patrol(self, resident_id: int) -> int:
        """Random patrol movement.

        Args:
            resident_id: Agent ID

        Returns:
            Random movement action
        """
        if resident_id not in self._patrol_state:
            self._patrol_state[resident_id] = {
                'action': self._random_patrol_action(),
                'frames_remaining': cfg.PATROL_PERSISTENCE
            }

        state = self._patrol_state[resident_id]
        state['frames_remaining'] -= 1

        if state['frames_remaining'] <= 0:
            state['action'] = self._random_patrol_action()
            state['frames_remaining'] = cfg.PATROL_PERSISTENCE

        return state['action']

    def _random_patrol_action(self) -> int:
        """Pick random patrol action."""
        direction = self._rng.choice(cfg.PATROL_DIRECTIONS)
        if direction == 'FORWARD':
            return cfg.ACTION_FORWARD
        elif direction == 'TURN_LEFT':
            return cfg.ACTION_TURN_LEFT
        elif direction == 'TURN_RIGHT':
            return cfg.ACTION_TURN_RIGHT
        else:
            raise ValueError(f"Unknown patrol direction: {direction}")
```

#### 4. Update agents/residents/config.py

```python
# Added by RST: Configuration for scripted residents
"""Configuration constants for resident behavior."""

# Action indices (from allelopathic_harvest_normative.py)
ACTION_NOOP = 0
ACTION_FORWARD = 1
ACTION_BACKWARD = 2
ACTION_STEP_LEFT = 3
ACTION_STEP_RIGHT = 4
ACTION_TURN_LEFT = 5
ACTION_TURN_RIGHT = 6
ACTION_FIRE_ZAP = 7
ACTION_FIRE_ONE = 8  # Plant RED
ACTION_FIRE_TWO = 9  # Plant GREEN
ACTION_FIRE_THREE = 10  # Plant BLUE

# Color indices (Lua 1-indexed)
COLOR_GREY = 0
COLOR_RED = 1
COLOR_GREEN = 2
COLOR_BLUE = 3

# Plant action mapping
PLANT_ACTION_MAP = {
    COLOR_RED: ACTION_FIRE_ONE,
    COLOR_GREEN: ACTION_FIRE_TWO,
    COLOR_BLUE: ACTION_FIRE_THREE,
}

# Behavioral parameters
GRACE_PERIOD = 50  # frames (matches substrate config)
PLANT_FREQUENCY = 2  # Plant every N steps
PATROL_PERSISTENCE = 10  # Frames to continue in same patrol direction
PATROL_DIRECTIONS = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
DEFAULT_SEED = 12345
```

---

## Implementation Steps

### Phase 1: Add Observation Components (Lua)
1. Add ColorStateObservation to avatar_library.lua
2. Add ImmunityStateObservation to avatar_library.lua
3. Register components
4. Add to substrate config (residents only)

### Phase 2: Implement Python Support
1. Create agents/utils/__init__.py
2. Create agents/utils/event_parser.py
3. Update agents/residents/info_extractor.py
4. Update agents/residents/scripted_residents.py
5. Update agents/residents/config.py

### Phase 3: Update Substrate Config
1. Add new observations to individual_observation_names
2. Update specs.py to include new observation specs
3. Only attach observation components to residents, not ego

### Phase 4: Testing
1. Test event_parser with raw dmlab2d events
2. Test info_extractor with sample observations
3. Test ResidentController.act() with sample info dicts
4. Integration test with full environment

---

## Key Differences from Old Approach

**OLD (Broken)**:
- ResidentObserver Lua component emitting events
- Event parsing for all information
- Complex event matching logic
- Broke when events were missed

**NEW (Clean)**:
- Observation components exposing state directly
- Simple observation array access
- Events only for debugging/metrics, not core logic
- Reliable, uses existing MeltingPot patterns

**SAME (Keep)**:
- Policy priority: compliance > sanction > patrol
- Violation logic: grey OR body_color ≠ altar_color
- Immunity checking before sanctioning
- Python-side controller with simple act() interface

---

## What's REALLY NEEDED

**CRITICAL**:
- ColorStateObservation component (Lua)
- ImmunityStateObservation component (Lua)
- event_parser.py (Python)

**IMPORTANT**:
- Update info_extractor.py to use new observations
- Simplify scripted_residents.py (remove broken event-based logic)

**NICE TO HAVE**:
- Add position observations for smarter navigation
- Add berry position observations for better harvesting

---

## Next Steps

Should I:
1. Start implementing Phase 1 (add Lua observation components)?
2. Start implementing Phase 2 (Python utilities)?
3. You review the plan and suggest changes?

The plan is ready to execute. All code snippets are based on existing patterns in the codebase.
