# Resident Implementation - FINAL Plan (Multi-Env Compatible)

## What I Learned from Multi-Env Code

Studied `/home/ubuntu/altar/altar-transfer/agents/sa/utils/multi_env_frozen_agent_loader.py`:

### Key Pattern: NormativeMonoculturePolicy

```python
class NormativeMonoculturePolicy(policy.Policy):
    """Policy that observes altar and enforces matching berry planting."""

    def step(self, timestep: dm_env.TimeStep, prev_state: Any) -> tuple[int, Any]:
        obs = timestep.observation

        # Observe the current norm from WORLD.ALTAR_COLOR
        if 'WORLD.ALTAR_COLOR' in obs:
            altar_color = int(obs['WORLD.ALTAR_COLOR'])
            self._current_altar_color = altar_color

        # Adapt behavior to current altar color
        if can_plant:
            return self._fire_actions[self._current_altar_color], None

        # Detect violators based on current altar
        if self._detect_non_conforming_agent(rgb_obs):
            return self._sanction_action, None
```

**Key insights**:
1. Policy reads `WORLD.ALTAR_COLOR` from observations
2. Adapts behavior dynamically to current norm
3. Same policy works for all environments
4. **MultiEnvWrapper** randomly selects environment on each reset
5. Each environment can have different altar color

### How Multi-Env Training Works

```python
class MultiEnvAllelopathicHarvestWrapper:
    def reset(self):
        # Select environment (e.g., red norm, green norm, or blue norm)
        self._current_env_name = self._select_environment()
        self._current_wrapper = self._env_wrappers[self._current_env_name]

        # Reset selected environment
        obs, info = self._current_wrapper.reset()
        return obs, info
```

- On each reset, randomly samples an environment
- Each environment has different altar color (norm)
- Agent must adapt to sampled norm each episode
- **Same policy works for single-env (always RED) or multi-env (RED/GREEN/BLUE)**

---

## Our Implementation

We'll create `ResidentPolicy` that:
1. Observes `ALTAR` to know current norm
2. Observes `AGENT_COLORS` to detect violators
3. Observes `IMMUNITY_STATUS` to avoid re-sanctioning
4. Adapts behavior to current norm
5. **Works for both single-norm and multi-norm training**

---

## What We Need

### 1. Add Observation Components (Lua)

**File**: `meltingpot/meltingpot/lua/modules/avatar_library.lua`

Add 3 new observation components:

#### A. ColorStateObservation

```lua
--[[ Added by RST: Exposes all agents' body colors as observation.

This component provides residents with information about other agents' colors
to detect norm violations.
]]
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

#### B. ImmunityStateObservation

```lua
--[[ Added by RST: Exposes all agents' immunity status as observation.

This component provides residents with immunity information to avoid
re-sanctioning immune agents.
]]
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

#### C. PlayerIndexObservation

```lua
--[[ Added by RST: Exposes this player's index as observation.

This allows residents to identify themselves in arrays.
]]
local PlayerIndexObservation = class.Class(component.Component)

function PlayerIndexObservation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PlayerIndexObservation')},
  })
  PlayerIndexObservation.Base.__init__(self, kwargs)
end

function PlayerIndexObservation:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.PLAYER_INDEX',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        return playerIndex
      end
  }
end
```

**Register components**:
```lua
local allComponents = {
    -- ... existing ...
    ColorStateObservation = ColorStateObservation,  -- Added by RST
    ImmunityStateObservation = ImmunityStateObservation,  -- Added by RST
    PlayerIndexObservation = PlayerIndexObservation,  -- Added by RST
}
```

### 2. Create ResidentPolicy (Python)

**File**: `agents/residents/resident_policy.py`

```python
# Added by RST: Resident policy for normative sanctioning
"""Policy that implements equilibrium play for residents."""

import numpy as np
from typing import Any, Tuple, Optional
import dm_env
from meltingpot.utils.policies import policy


class ResidentPolicy(policy.Policy):
    """Resident policy that adapts to current norm (altar color).

    This policy works for both single-environment and multi-environment training:
    - Single-env: Always same altar color (e.g., RED)
    - Multi-env: Altar color changes each episode (RED/GREEN/BLUE)

    Behavior:
    - Always compliant: body_color = altar_color
    - Sanction violators: grey OR body_color != altar_color
    - Never sanction immune agents
    - Works with any altar color (adapts dynamically)
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize resident policy.

        Args:
            seed: Random seed for patrol behavior
        """
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._patrol_action = None
        self._patrol_frames_remaining = 0
        self._frames_since_plant = 0
        self._step_count = 0

        # Action constants
        self.NOOP = 0
        self.FORWARD = 1
        self.TURN_LEFT = 5
        self.TURN_RIGHT = 6
        self.FIRE_ZAP = 7
        self.FIRE_RED = 8
        self.FIRE_GREEN = 9
        self.FIRE_BLUE = 10

        # Plant actions by altar color (Lua 1-indexed)
        self.PLANT_MAP = {
            1: self.FIRE_RED,    # RED altar
            2: self.FIRE_GREEN,  # GREEN altar
            3: self.FIRE_BLUE,   # BLUE altar
        }

        # Behavioral parameters
        self.PLANT_FREQUENCY = 2  # Plant every N frames
        self.PATROL_PERSISTENCE = 10  # Frames per patrol direction
        self.GRACE_PERIOD = 50  # Frames (matches substrate config)

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
        self._step_count += 1

        # Extract observations
        altar_color = int(obs['ALTAR'])  # Current norm
        ready_to_shoot = float(obs['READY_TO_SHOOT'])
        agent_colors = np.asarray(obs['AGENT_COLORS'])  # [num_players], Lua 1-indexed
        immunity_status = np.asarray(obs['IMMUNITY_STATUS'])  # [num_players]
        player_index = int(obs['PLAYER_INDEX'])  # This resident's Lua index (1-indexed)

        # Get own color (convert Lua 1-indexed to array index)
        own_color = agent_colors[player_index - 1]

        # P1: Ensure compliance - plant permitted color
        if own_color != altar_color:
            return self.PLANT_MAP[altar_color], None

        # P2: Sanction violators if ready
        if ready_to_shoot > 0.5:
            action = self._try_sanction(
                agent_colors, immunity_status, altar_color,
                player_index, self._step_count
            )
            if action is not None:
                return action, None

        # P3: Plant frequently to maintain compliance
        self._frames_since_plant += 1
        if self._frames_since_plant >= self.PLANT_FREQUENCY:
            self._frames_since_plant = 0
            return self.PLANT_MAP[altar_color], None

        # P4: Patrol
        return self._patrol(), None

    def _try_sanction(
        self,
        agent_colors: np.ndarray,
        immunity_status: np.ndarray,
        altar_color: int,
        own_player_index: int,
        world_step: int
    ) -> Optional[int]:
        """Try to sanction violators.

        Args:
            agent_colors: Array of all agents' colors (Lua 1-indexed positions)
            immunity_status: Array of immunity (1=immune, 0=not)
            altar_color: Current permitted color
            own_player_index: This resident's Lua player index (1-indexed)
            world_step: Current timestep

        Returns:
            FIRE_ZAP if should sanction, None otherwise
        """
        # Check grace period
        if world_step < self.GRACE_PERIOD:
            return None

        # Check each agent (Lua uses 1-indexing)
        for lua_idx in range(1, len(agent_colors) + 1):
            array_idx = lua_idx - 1  # Convert to 0-indexed for array access

            # Skip self
            if lua_idx == own_player_index:
                continue

            color = agent_colors[array_idx]
            is_immune = (immunity_status[array_idx] == 1)

            # Violation: grey (0) OR wrong color
            is_violating = (color == 0) or (color != altar_color)

            if is_violating and not is_immune:
                # TODO: Check if in range using AVATAR_IDS_IN_RANGE_TO_ZAP
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

### 3. Update Substrate Config

**File**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative.py`

Add new observations to individual_observation_names:

```python
_INDIVIDUAL_OBSERVATION_NAMES = [
    'RGB',
    'READY_TO_SHOOT',
    'ALTAR',
    'AGENT_COLORS',  # Added by RST
    'IMMUNITY_STATUS',  # Added by RST
    'PLAYER_INDEX',  # Added by RST
]
```

Add components to residents only (in `create_avatar_object` function):

```python
if player_idx >= num_focal_players:  # Residents only
    avatar_object["components"].append({
        "component": "ColorStateObservation",
    })
    avatar_object["components"].append({
        "component": "ImmunityStateObservation",
    })
    avatar_object["components"].append({
        "component": "PlayerIndexObservation",
    })
```

### 4. Update specs.py

**File**: `meltingpot/meltingpot/utils/substrates/specs.py`

Add observation specs:

```python
OBSERVATION = {
    # ... existing ...
    'AGENT_COLORS': dm_env.specs.Array(
        shape=(num_players,), dtype=np.int32, name='AGENT_COLORS'),
    'IMMUNITY_STATUS': dm_env.specs.Array(
        shape=(num_players,), dtype=np.int32, name='IMMUNITY_STATUS'),
    'PLAYER_INDEX': dm_env.specs.Array(
        shape=(), dtype=np.float64, name='PLAYER_INDEX'),
}
```

---

## Integration for Multi-Env

For multi-environment training, follow altar-transfer pattern:

**File**: `agents/residents/multi_env_resident_loader.py`

```python
# Added by RST: Multi-env resident loader
"""Loader for residents in multi-environment training."""

from typing import Dict
from meltingpot.utils.policies import policy
from agents.residents.resident_policy import ResidentPolicy


class MultiEnvResidentLoader:
    """Loads residents for multi-environment training."""

    def __init__(self, config, base_seed: int = None):
        """Initialize loader.

        Args:
            config: Training config
            base_seed: Base seed for deterministic residents
        """
        self._config = config
        self._base_seed = base_seed

    def get_residents_for_environment(self, environment_name: str) -> Dict[int, policy.Policy]:
        """Get resident policies for a specific environment.

        Args:
            environment_name: Name of environment (e.g., 'norm_red', 'norm_green')

        Returns:
            Dict mapping agent indices to policies
        """
        residents = {}

        # Create residents for all non-ego agents
        for idx in range(self._config.num_agents):
            if idx != self._config.ego_agent_idx:
                # Use deterministic seed per agent for consistency
                seed = (self._base_seed + idx) if self._base_seed is not None else None
                residents[idx] = ResidentPolicy(seed=seed)

        return residents

    def get_all_residents(self) -> Dict[str, Dict[int, policy.Policy]]:
        """Get residents for all environments.

        Returns:
            Dict mapping environment names to resident dicts
        """
        all_residents = {}
        for env_name in self._config.environments:
            all_residents[env_name] = self.get_residents_for_environment(env_name)
        return all_residents
```

**Key insight**: Same `ResidentPolicy` works for all environments because it reads `ALTAR` and adapts!

---

## Why This Works for Both Single and Multi-Env

**Single-Environment Training**:
- Altar always RED (altar_color = 1)
- ResidentPolicy reads `ALTAR` = 1 every step
- Plants RED, sanctions non-RED violators
- Works perfectly

**Multi-Environment Training**:
- Episode 1: Altar RED (altar_color = 1)
- Episode 2: Altar GREEN (altar_color = 2)
- Episode 3: Altar BLUE (altar_color = 3)
- ResidentPolicy reads `ALTAR` each episode
- Adapts behavior to current norm
- Works perfectly

**Same policy, works everywhere!**

---

## Implementation Steps

### Phase 1: Add Lua Observation Components
1. Add ColorStateObservation to avatar_library.lua
2. Add ImmunityStateObservation to avatar_library.lua
3. Add PlayerIndexObservation to avatar_library.lua
4. Register all components

### Phase 2: Update Substrate Config
1. Add observations to individual_observation_names
2. Add components to residents in create_avatar_object
3. Update specs.py

### Phase 3: Create ResidentPolicy
1. Create agents/residents/resident_policy.py
2. Implement policy.Policy interface
3. Adapt to ALTAR observation dynamically

### Phase 4: Delete Old Code
1. Remove agents/residents/info_extractor.py
2. Remove agents/residents/scripted_residents.py
3. Remove agents/residents/config.py (or simplify)

### Phase 5: Update Integration
1. Update ResidentWrapper to use policy.Policy
2. Or create new wrapper following altar-transfer pattern

---

## Next Steps

Should I start implementing Phase 1 (Lua observation components)?
