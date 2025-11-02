# Resident Implementation - CORRECT Plan

## What I Learned from altar-transfer

Studied `/home/ubuntu/altar/altar-transfer/agents/sa/utils/frozen_agent_loader.py`:

**HandcodedMonoculturePolicy** shows the RIGHT way:

```python
class HandcodedMonoculturePolicy(policy.Policy):
    def initial_state(self) -> Any:
        return None

    def step(self, timestep: dm_env.TimeStep, prev_state: Any) -> tuple[int, Any]:
        obs = timestep.observation

        # Decision based ONLY on observations
        if 'READY_TO_SHOOT' in obs and obs['READY_TO_SHOOT'] == 1.0:
            # Can zap
            if self._detect_violator(obs):
                return FIRE_ZAP, None

        # Plant, harvest, patrol...
        return action, None
```

**Key insights**:
- `policy.Policy` interface: `step(timestep, state) -> (action, state)`
- Receives `dm_env.TimeStep` with `.observation` dict
- Returns `(action_int, new_state)`
- Makes decisions from observations ONLY
- NO event parsing for decisions
- State can be None (stateless policy)

---

## What We Need

### 1. Add Observation Components (Lua)

Residents need to observe:
- Other agents' body colors → **ColorStateObservation**
- Other agents' immunity → **ImmunityStateObservation**
- Altar color → **Already have AltarObservation**
- Zap cooldown → **Already have ReadyToShootObservation**

### 2. Implement ResidentPolicy (Python)

**NEW file**: `agents/residents/resident_policy.py`

```python
# Added by RST: Resident policy for normative sanctioning
"""Policy that implements equilibrium play for residents."""

import numpy as np
from typing import Any, Tuple
import dm_env
from meltingpot.utils.policies import policy


class ResidentPolicy(policy.Policy):
    """Deterministic resident policy for normative sanctioning.

    Residents play at equilibrium:
    - Always compliant (body_color = altar_color)
    - Sanction violators (grey OR wrong color)
    - Never sanction immune agents
    """

    def __init__(self, seed: int = 12345):
        """Initialize resident policy.

        Args:
            seed: Random seed for patrol behavior
        """
        self._rng = np.random.RandomState(seed)
        self._patrol_action = None
        self._patrol_frames_remaining = 0
        self._frames_since_plant = 0

        # Action constants
        self.NOOP = 0
        self.FORWARD = 1
        self.TURN_LEFT = 5
        self.TURN_RIGHT = 6
        self.FIRE_ZAP = 7
        self.FIRE_RED = 8
        self.FIRE_GREEN = 9
        self.FIRE_BLUE = 10

        # Plant actions by color (Lua 1-indexed)
        self.PLANT_MAP = {
            1: self.FIRE_RED,
            2: self.FIRE_GREEN,
            3: self.FIRE_BLUE,
        }

        # Behavioral parameters
        self.PLANT_FREQUENCY = 2  # Plant every N frames
        self.PATROL_PERSISTENCE = 10  # Frames per patrol direction
        self.GRACE_PERIOD = 50  # frames

    def initial_state(self) -> Any:
        """Return initial state (None for stateless policy)."""
        return None

    def step(self, timestep: dm_env.TimeStep, prev_state: Any) -> Tuple[int, Any]:
        """Select action based on observations.

        Args:
            timestep: Environment timestep with observations
            prev_state: Previous state (unused, we're stateless)

        Returns:
            (action, new_state) tuple
        """
        obs = timestep.observation

        # Extract observations
        altar_color = int(obs['ALTAR'])
        ready_to_shoot = float(obs['READY_TO_SHOOT'])
        agent_colors = np.asarray(obs['AGENT_COLORS'])  # [num_players]
        immunity_status = np.asarray(obs['IMMUNITY_STATUS'])  # [num_players]

        # Get own color (residents are indexed from 0 in Python)
        # But need to figure out which index we are...
        # This is tricky - we need player_index passed somehow
        # For now, assume we can infer from timestep or pass as init param

        # P1: Ensure compliance - plant permitted color
        own_color = agent_colors[0]  # FIXME: need actual index
        if own_color != altar_color:
            return self.PLANT_MAP[altar_color], None

        # P2: Sanction violators if ready
        if ready_to_shoot > 0.5:
            action = self._try_sanction(agent_colors, immunity_status, altar_color, timestep)
            if action is not None:
                return action, None

        # P3: Plant frequently to maintain compliance
        self._frames_since_plant += 1
        if self._frames_since_plant >= self.PLANT_FREQUENCY:
            self._frames_since_plant = 0
            return self.PLANT_MAP[altar_color], None

        # P4: Patrol
        return self._patrol(), None

    def _try_sanction(self, agent_colors, immunity_status, altar_color, timestep) -> int:
        """Try to sanction violators.

        Args:
            agent_colors: Array of all agents' colors
            immunity_status: Array of immunity (1=immune, 0=not)
            altar_color: Permitted color
            timestep: Full timestep for checking grace period

        Returns:
            FIRE_ZAP if should sanction, None otherwise
        """
        # Check grace period (TODO: need world_step from somewhere)
        # For now skip grace check

        # Check each agent
        for i, color in enumerate(agent_colors):
            if i == 0:  # FIXME: skip self
                continue

            # Violation: grey (0) OR wrong color
            is_violating = (color == 0) or (color != altar_color)
            is_immune = (immunity_status[i] == 1)

            if is_violating and not is_immune:
                # TODO: Check if in range
                # For now, just zap (will fizzle if not in range)
                return self.FIRE_ZAP

        return None

    def _patrol(self) -> int:
        """Random patrol movement."""
        if self._patrol_frames_remaining <= 0:
            # Pick new action
            self._patrol_action = self._rng.choice([
                self.FORWARD, self.TURN_LEFT, self.TURN_RIGHT
            ])
            self._patrol_frames_remaining = self.PATROL_PERSISTENCE

        self._patrol_frames_remaining -= 1
        return self._patrol_action

    def close(self) -> None:
        """Cleanup."""
        pass
```

---

## Issues to Fix

### Problem 1: Player Index

The policy needs to know its own player index to:
- Skip itself when checking for violators
- Know its own color

**Solution Options**:
1. Pass player_index in `__init__`
2. Add to observation (PLAYER_INDEX scalar)
3. Use a wrapper that binds player_index

**Recommendation**: Add `PLAYER_INDEX` observation

### Problem 2: World Step (for grace period)

Need current timestep to check grace period.

**Solution**: Add `WORLD_STEP` observation

### Problem 3: Integration

Current code uses ResidentController which expects `info` dict.
Need to use `policy.Policy` interface instead.

**Solution**: Update ResidentWrapper to use policies

---

## Implementation Steps

### Phase 1: Add Lua Observations

**File**: `meltingpot/meltingpot/lua/modules/avatar_library.lua`

Add 4 new observation components:

1. **ColorStateObservation** - All agents' colors
2. **ImmunityStateObservation** - All agents' immunity
3. **PlayerIndexObservation** - This player's index
4. **WorldStepObservation** - Current timestep

### Phase 2: Create ResidentPolicy

**File**: `agents/residents/resident_policy.py`

Implement `policy.Policy` with:
- `step(timestep, state)` using observations
- Compliance, sanctioning, patrol logic
- NO event parsing

### Phase 3: Update Integration

**File**: `agents/envs/resident_wrapper.py`

Change from:
```python
# OLD
resident_action = controller.act(agent_id, info)
```

To:
```python
# NEW
resident_action, new_state = resident_policy.step(timestep, prev_state)
```

### Phase 4: Delete Old Files

Remove:
- `agents/residents/info_extractor.py` (not needed)
- `agents/residents/scripted_residents.py` (replaced by resident_policy.py)
- `agents/utils/event_parser.py` (if it was only for residents)

---

## Key Differences

**OLD (Wrong)**:
- ResidentController.act(agent_id, info)
- info_extractor parsing events
- Custom info dict format
- Event-based decision making

**NEW (Correct)**:
- ResidentPolicy.step(timestep, state)
- Standard policy.Policy interface
- Observations only, no events for decisions
- Follows MeltingPot patterns

---

## Next Steps

1. Implement Phase 1 (Lua observations)
2. Implement Phase 2 (Python policy)
3. Update integration
4. Test

NO EVENT PARSING for decision making!
