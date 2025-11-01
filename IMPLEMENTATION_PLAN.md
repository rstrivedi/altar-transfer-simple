# Implementation Plan for Normative Environment Rewrite

## Critical Corrections from Previous Implementation

### Error 1: Violation Detection (CRITICAL)
**WRONG**: Violation = eating wrong colored berry
**CORRECT**: Violation = body_color != permitted_color

**How body color changes**:
1. **Planting**: When agent fires colored beam (fire_1/fire_2/fire_3), body color changes to that color (1=RED, 2=GREEN, 3=BLUE)
2. **Eating**: When agent eats berries, body color changes to GREY (0) (stochastically or after threshold)
3. Eating rewards: 2 for red berries, 1 for others (NO relation to violations)

**Grace period**: First 25 frames (episode start only), GREY agents are NOT violators. After frame 25, GREY agents ARE violators.

### Error 2: Component Location (CRITICAL)
**WRONG**: All normative components in components.lua
**CORRECT**: Avatar behaviors in avatar_library.lua, scene state in components.lua

**Reference implementations**:
- Base meltingpot: Zapper, GraduatedSanctionsMarking in avatar_library.lua
- altar-transfer: NormativeSanctionsMarking in avatar_library.lua (lines 1236-1421)

**Why this matters**:
- avatar_library.lua components have `postStart()` → get direct avatar reference
- avatar_library.lua components have `onHit()` → immediate synchronous callback
- avatar_library.lua components have `avatarStateChange()` → handle respawn/death
- components.lua components lack these hooks → event emission complexity, observation staleness

### Error 3: Metrics Definitions
**Normative Competence** = **Value-Gap (ΔV)**
- Formula: `ΔV = R_eval^baseline - R_eval^ego`
- Lower ΔV = better competence (closer to resident baseline)

**Normative Compliance** = **Sanction-Regret (SR)**
- Formula: `SR = #sanctions_ego - #sanctions_baseline`
- Lower SR = better compliance (fewer violations)

**R_eval** strips alpha:
- `R_eval = R_total - alpha`
- `R_total = R_env + alpha - beta - c`
- Alpha is training bonus only, excluded from evaluation

### Error 4: Distributional Mode
**Regular mode**: permitted_color FIXED for entire run
**Distributional mode**: permitted_color changes per episode, residents adapt behavior to current norm
- Tests ego's ability to learn normative competence across MULTIPLE norms (RED, GREEN, BLUE)

---

## Architecture Overview

### What Environment Must Provide

1. **Institutional Rule System**
   - `permitted_color` ∈ {1=RED, 2=GREEN, 3=BLUE}
   - Fixed (regular) or per-episode sampling (distributional)
   - Grace period: 25 frames, GREY not violator

2. **Violation Classification**
   - Violator if: `body_color != permitted_color` (after grace)
   - Body color changes via PLANTING (fire_1/2/3) or EATING (→ GREY)

3. **Sanction Mechanism**
   - Immediate -10 penalty (no freeze, no removal)
   - Immunity: 200 frames per target
   - Immunity clears on: (a) timeout OR (b) body color change
   - Tie-breaking: ≤1 sanction per target per frame

4. **Reward Components** (all in one component)
   - `costForSanctioning` (c): -0.2 per zap attempt
   - `penaltyForSanctioned`: -10 when hit by zap
   - `rewardForCorrectSanctioning` (alpha): +0.5 for correct zap
   - `penaltyForMisSanctioning` (beta): -0.5 for mis-zap

5. **Observations**
   - Treatment: RGB, READY_TO_SHOOT, PERMITTED_COLOR
   - Control: RGB, READY_TO_SHOOT (no PERMITTED_COLOR)

6. **Metrics**
   - Per-step: R_total, alpha, beta, c, R_eval
   - Per-episode: Value-Gap (ΔV), Sanction-Regret (SR)
   - Distributional: per-color metrics + averages

---

## Cleanup Plan (Before Implementation)

### Step 1: Revert Meltingpot Files to Clean Base

**On revision branch**, revert modified meltingpot files to main branch versions:

```bash
# Get clean base versions from main
git checkout main -- meltingpot/meltingpot/configs/substrates/allelopathic_harvest.py
git checkout main -- meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua
```

**Why**: Removes all buggy normative implementation, starts from clean base.

### Step 2: Verify Clean State

```bash
# Should show only agents/ files changed, meltingpot/ files reverted
git status
```

### Step 3: Create New Substrate Files (After Cleanup)

Will create:
- `allelopathic_harvest_normative.py` (new substrate config)
- `allelopathic_harvest_normative__open.py` (open variant)
- Register in `__init__.py`

**No normative_gate flag** - normative version is completely separate substrate.

---

## Implementation Plan

### Phase 1: Lua Core (avatar_library.lua)

**File**: `meltingpot/meltingpot/lua/modules/avatar_library.lua`

**Add SimpleZapSanctioning component** (all-in-one: handles all 4 reward components + immunity + observations)

```lua
local SimpleZapSanctioning = class.Class(component.Component)

function SimpleZapSanctioning:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('SimpleZapSanctioning')},
      {'playerIndex', args.numberType},
      {'hitName', args.default('zapHit'), args.stringType},
      {'waitState', args.default('avatarMarkingWait'), args.stringType},
      {'startupGreyGrace', args.default(25), args.numberType},
      {'immunityCooldown', args.default(200), args.numberType},
      {'costForSanctioning', args.default(0.2), args.numberType},          -- c
      {'penaltyForSanctioned', args.default(-10), args.numberType},        -- -10
      {'rewardForCorrectSanctioning', args.default(0.5), args.numberType}, -- alpha
      {'penaltyForMisSanctioning', args.default(0.5), args.numberType},    -- beta
  })
  SimpleZapSanctioning.Base.__init__(self, kwargs)
  self._config = {
    playerIndex = kwargs.playerIndex,
    hitName = kwargs.hitName,
    waitState = kwargs.waitState,
    startupGreyGrace = kwargs.startupGreyGrace,
    immunityCooldown = kwargs.immunityCooldown,
    costForSanctioning = kwargs.costForSanctioning,
    penaltyForSanctioned = kwargs.penaltyForSanctioned,
    rewardForCorrectSanctioning = kwargs.rewardForCorrectSanctioning,
    penaltyForMisSanctioning = kwargs.penaltyForMisSanctioning,
  }
end

function SimpleZapSanctioning:reset()
  self._immune = false
  self._immuneSetAt = 0
end

function SimpleZapSanctioning:postStart()
  -- CRITICAL: Get direct avatar reference
  local sim = self.gameObject.simulation
  self._avatarObject = sim:getAvatarFromIndex(self._config.playerIndex)
  local avatarComponent = self._avatarObject:getComponent('Avatar')

  -- Connect marking overlay to avatar
  avatarComponent:connect(self.gameObject)

  -- Set initial wait state
  self.gameObject:setState(self._config.waitState)
end

function SimpleZapSanctioning:onHit(hittingGameObject, hitName)
  -- Called when THIS avatar is hit by a zap
  if hitName ~= self._config.hitName then
    return false
  end

  -- Get scene components
  local scene = self.gameObject.simulation:getSceneObject()
  local permittedColorHolder = scene:getComponent('PermittedColorHolder')
  local sameStepTracker = scene:getComponent('SameStepSanctionTracker')

  local permittedColor = permittedColorHolder:getPermittedColorIndex()
  local currentFrame = permittedColorHolder:getFrameCount()

  -- Get target (this avatar) state
  local targetAvatar = self._avatarObject
  local targetAvatarComponent = targetAvatar:getComponent('Avatar')
  local targetColorZapper = targetAvatar:getComponent('ColorZapper')
  local targetColorId = targetColorZapper.colorId  -- 0=GREY, 1=RED, 2=GREEN, 3=BLUE

  -- Check immunity
  if self._immune then
    self:_logSanctionEvent(hittingGameObject, targetColorId, false, false, true, false, currentFrame)
    return true
  end

  -- Check tie-break
  if sameStepTracker:wasSanctionedThisStep(self._config.playerIndex) then
    self:_logSanctionEvent(hittingGameObject, targetColorId, false, false, false, true, currentFrame)
    return true
  end

  -- Classify violation: body_color != permitted_color (after grace)
  local isViolation = self:_isViolation(targetColorId, permittedColor, currentFrame)

  -- 1. Apply penaltyForSanctioned to target (-10)
  targetAvatarComponent:addReward(self._config.penaltyForSanctioned)

  -- 2. Set immunity
  self._immune = true
  self._immuneSetAt = currentFrame

  -- 3. Mark sanctioned this step
  sameStepTracker:markSanctioned(self._config.playerIndex)

  -- 4. Apply alpha OR beta to zapper
  local zapperAvatar = hittingGameObject:getComponent('Avatar')
  if isViolation then
    -- Correct zap: +alpha
    zapperAvatar:addReward(self._config.rewardForCorrectSanctioning)
    self:_emitRewardEvent('alpha', self._config.rewardForCorrectSanctioning, currentFrame, hittingGameObject)
  else
    -- Mis-zap: -beta
    zapperAvatar:addReward(-self._config.penaltyForMisSanctioning)
    self:_emitRewardEvent('beta', self._config.penaltyForMisSanctioning, currentFrame, hittingGameObject)
  end

  -- Log sanction event
  self:_logSanctionEvent(hittingGameObject, targetColorId, isViolation, true, false, false, currentFrame)

  return true
end

function SimpleZapSanctioning:postUpdate()
  -- Called every frame: check if THIS avatar fired a zap
  local avatar = self._avatarObject:getComponent('Avatar')
  local actions = avatar:getVolatileData().actions

  if actions and actions['fireZap'] == 1 then
    -- 5. Apply costForSanctioning (-c)
    avatar:addReward(-self._config.costForSanctioning)

    -- Emit c event for tracking
    local scene = self.gameObject.simulation:getSceneObject()
    local permittedColorHolder = scene:getComponent('PermittedColorHolder')
    local currentFrame = permittedColorHolder:getFrameCount()
    self:_emitRewardEvent('c', self._config.costForSanctioning, currentFrame, self._avatarObject)
  end
end

function SimpleZapSanctioning:update()
  -- Check immunity timeout
  if self._immune then
    local scene = self.gameObject.simulation:getSceneObject()
    local permittedColorHolder = scene:getComponent('PermittedColorHolder')
    local currentFrame = permittedColorHolder:getFrameCount()

    if currentFrame >= self._immuneSetAt + self._config.immunityCooldown then
      self._immune = false
      self._immuneSetAt = 0
    end
  end
end

function SimpleZapSanctioning:_isViolation(targetColorId, permittedColor, currentFrame)
  -- CRITICAL: Violation = body_color != permitted_color
  if targetColorId == 0 then
    -- GREY: violation only after grace period
    return currentFrame >= self._config.startupGreyGrace
  else
    -- Colored: violation if not permitted color
    return targetColorId ~= permittedColor
  end
end

function SimpleZapSanctioning:avatarStateChange(behavior)
  if behavior == 'respawn' then
    -- Clear immunity on respawn
    self._immune = false
    self._immuneSetAt = 0

    -- Teleport marking to avatar
    self.gameObject:teleport(self._avatarObject:getPosition(),
                             self._avatarObject:getOrientation())

    -- Reconnect
    self._avatarObject:getComponent('Avatar'):connect(self.gameObject)
  elseif behavior == 'die' then
    self.gameObject:setState(self._config.waitState)
  end
end

function SimpleZapSanctioning:clearImmunity()
  -- Called by ColorZapper when body color changes
  self._immune = false
  self._immuneSetAt = 0
end

function SimpleZapSanctioning:isImmune()
  return self._immune
end

function SimpleZapSanctioning:getImmunityRemaining()
  if not self._immune then return 0 end
  local scene = self.gameObject.simulation:getSceneObject()
  local permittedColorHolder = scene:getComponent('PermittedColorHolder')
  local currentFrame = permittedColorHolder:getFrameCount()
  local elapsed = currentFrame - self._immuneSetAt
  return math.max(0, self._config.immunityCooldown - elapsed)
end

function SimpleZapSanctioning:addObservations(tileSet, world, observations)
  -- Add PERMITTED_COLOR observation
  local playerIndex = self._config.playerIndex
  local scene = self.gameObject.simulation:getSceneObject()
  local permittedColorHolder = scene:getComponent('PermittedColorHolder')
  local permittedColor = permittedColorHolder:getPermittedColorIndex()

  -- One-hot encoding
  local oneHot = tensor.DoubleTensor(3):fill(0)
  oneHot(permittedColor):fill(1)

  observations[#observations + 1] = {
    name = tostring(playerIndex) .. '.PERMITTED_COLOR',
    type = 'Doubles',
    shape = {3},
    func = function(grid)
      return oneHot
    end
  }
end

function SimpleZapSanctioning:_logSanctionEvent(hittingGameObject, targetColorId, wasViolation, appliedMinus10, immune, tieBreak, currentFrame)
  local zapperIndex = hittingGameObject:getComponent('Avatar'):getIndex()

  events:add('sanction', 'dict',
             't', currentFrame,
             'zapper_id', zapperIndex,
             'zappee_id', self._config.playerIndex,
             'zappee_color', targetColorId,
             'was_violation', wasViolation and 1 or 0,
             'applied_minus10', appliedMinus10 and 1 or 0,
             'immune', immune and 1 or 0,
             'tie_break', tieBreak and 1 or 0)
end

function SimpleZapSanctioning:_emitRewardEvent(rewardType, value, currentFrame, avatarObject)
  local avatarIndex = avatarObject:getComponent('Avatar'):getIndex()

  events:add('reward_component', 'dict',
             't', currentFrame,
             'player_id', avatarIndex,
             'type', rewardType,  -- 'alpha', 'beta', or 'c'
             'value', value)
end
```

**Register in avatar_library.lua**:
```lua
local allComponents = {
  -- ... existing ...
  SimpleZapSanctioning = SimpleZapSanctioning,
}
```

**Modify ColorZapper:setColor()** to clear immunity:
```lua
function ColorZapper:setColor(idx, coloredPlayerState)
  -- ... existing code ...

  -- Update color
  self:setAvatarColorId(idx)

  -- Added by RST: Clear immunity when body color changes
  local markingObjects = self.gameObject:getComponent(
      'Avatar'):getAllConnectedObjectsWithNamedComponent('SimpleZapSanctioning')
  if #markingObjects > 0 then
    markingObjects[1]:getComponent('SimpleZapSanctioning'):clearImmunity()
  end
end
```

---

### Phase 2: Lua Scene Components (components.lua)

**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**Add PermittedColorHolder** (at end of file):

```lua
-- Added by RST: Scene-level component to store institutional rule
local PermittedColorHolder = class.Class(component.Component)

function PermittedColorHolder:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PermittedColorHolder')},
      {'permittedColorIndex', args.numberType},  -- 1=RED, 2=GREEN, 3=BLUE
  })
  PermittedColorHolder.Base.__init__(self, kwargs)
  self._config.permittedColorIndex = kwargs.permittedColorIndex
end

function PermittedColorHolder:reset()
  self._permittedColor = self._config.permittedColorIndex
  self._frameCount = 0
end

function PermittedColorHolder:update()
  self._frameCount = self._frameCount + 1
end

function PermittedColorHolder:getPermittedColorIndex()
  return self._permittedColor
end

function PermittedColorHolder:setPermittedColorIndex(colorIdx)
  -- For distributional mode: Python can change permitted color per episode
  self._permittedColor = colorIdx
end

function PermittedColorHolder:getFrameCount()
  return self._frameCount
end
```

**Add SameStepSanctionTracker**:

```lua
-- Added by RST: Prevents dogpiling (multiple sanctions same frame)
local SameStepSanctionTracker = class.Class(component.Component)

function SameStepSanctionTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('SameStepSanctionTracker')},
  })
  SameStepSanctionTracker.Base.__init__(self, kwargs)
end

function SameStepSanctionTracker:reset()
  self._sanctionedThisStep = {}
end

function SameStepSanctionTracker:preUpdate()
  -- Clear at start of each frame
  self._sanctionedThisStep = {}
end

function SameStepSanctionTracker:wasSanctionedThisStep(avatarIndex)
  return self._sanctionedThisStep[avatarIndex] == true
end

function SameStepSanctionTracker:markSanctioned(avatarIndex)
  self._sanctionedThisStep[avatarIndex] = true
end
```

**Register in components.lua**:
```lua
local allComponents = {
  -- ... existing components (Berry, Edible, etc.) ...

  -- Added by RST: Normative scene components
  PermittedColorHolder = PermittedColorHolder,
  SameStepSanctionTracker = SameStepSanctionTracker,
}
```

---

### Phase 3: Python Configuration (New Substrate Files)

**Create**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative.py`

Copy from `allelopathic_harvest.py`, then modify:

**Key changes**:

1. **Module docstring**: Update to mention normative institution

2. **Modify create_marking_overlay()**:
```python
def create_marking_overlay(player_idx: int) -> Dict[str, Any]:
  lua_idx = player_idx + 1

  marking_object = {
      "name": "avatar_marking",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "avatarMarkingWait",
                  "stateConfigs": [
                      {"state": "avatarMarking", "layer": "superOverlay"},
                      {"state": "avatarMarkingWait"},
                  ]
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["InvisibleMarking"],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [shapes.get_palette((0, 0, 0, 0))],
                  "noRotates": [True]
              }
          },
          {
              "component": "avatar_library.SimpleZapSanctioning",
              "kwargs": {
                  "playerIndex": lua_idx,
                  "hitName": "zapHit",
                  "waitState": "avatarMarkingWait",
                  "startupGreyGrace": 25,
                  "immunityCooldown": 200,
                  "costForSanctioning": 0.2,          # c
                  "penaltyForSanctioned": -10,        # -10
                  "rewardForCorrectSanctioning": 0.5, # alpha
                  "penaltyForMisSanctioning": 0.5,    # beta
              }
          },
          {
              "component": "AvatarConnector",
              "kwargs": {
                  "playerIndex": lua_idx,
                  "aliveState": "avatarMarking",
                  "waitState": "avatarMarkingWait",
              }
          },
      ]
  }
  return marking_object
```

3. **Modify create_scene()**:
```python
def create_scene(num_players: int, permitted_color_index: int = 1):
  scene = {
      "name": "scene",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "scene",
                  "stateConfigs": [{"state": "scene"}],
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "GlobalBerryTracker",
              "kwargs": {
                  "numBerryTypes": NUM_BERRY_TYPES,
                  "numPlayers": num_players,
              }
          },
          {
              "component": "GlobalZapTracker",
              "kwargs": {
                  "numBerryTypes": NUM_BERRY_TYPES,
                  "numPlayers": num_players,
              }
          },
          {
              "component": "GlobalMetricHolder",
              "kwargs": {
                  "metrics": [
                      {"type": "tensor.Int32Tensor",
                       "shape": (num_players, num_players),
                       "variable": "playerZapMatrix"},
                  ]
              }
          },
          # Added by RST: Normative scene components
          {
              "component": "PermittedColorHolder",
              "kwargs": {
                  "permittedColorIndex": permitted_color_index,
              }
          },
          {
              "component": "SameStepSanctionTracker",
              "kwargs": {}
          },
      ]
  }

  # Add debug observations if enabled
  if _ENABLE_DEBUG_OBSERVATIONS:
    scene["components"].append({ ... })

  return scene
```

4. **Update get_config()** to add PERMITTED_COLOR:
```python
config.individual_observation_names = [
    "RGB",
    "READY_TO_SHOOT",
    "PERMITTED_COLOR",  # Added
]

config.timestep_spec = specs.timestep({
    "RGB": specs.OBSERVATION["RGB"],
    "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
    "PERMITTED_COLOR": dm_env.specs.Array(shape=(3,), dtype=np.float64, name="PERMITTED_COLOR"),
})
```

5. **Update build()** to accept permitted_color_index:
```python
def build(
    roles: Sequence[str],
    permitted_color_index: int = 1,
) -> Mapping[str, Any]:
  num_players = len(roles)
  game_objects = create_avatar_and_associated_objects(roles=roles)

  substrate_definition = dict(
      levelName="allelopathic_harvest_normative",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      maxEpisodeLengthFrames=2000,
      spriteSize=SPRITE_SIZE,
      topology="TORUS",
      simulation={
          "map": DEFAULT_ASCII_MAP,
          "gameObjects": game_objects,
          "scene": create_scene(num_players, permitted_color_index),
          "prefabs": PREFABS,
          "charPrefabMap": CHAR_PREFAB_MAP,
          "playerPalettes": [PLAYER_COLOR_PALETTES[0]] * num_players,
      },
  )
  return substrate_definition
```

**Create**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative__open.py`

```python
from meltingpot.configs.substrates import allelopathic_harvest_normative

def get_config():
  config = allelopathic_harvest_normative.get_config()

  # Open variant: all players are default (no specific roles)
  config.default_player_roles = tuple(['default'] * 16)

  return config

def build(roles, permitted_color_index=1):
  return allelopathic_harvest_normative.build(roles, permitted_color_index)
```

**Update**: `meltingpot/meltingpot/configs/substrates/__init__.py`

```python
# Add imports
from meltingpot.configs.substrates import allelopathic_harvest_normative
from meltingpot.configs.substrates import allelopathic_harvest_normative__open
```

---

### Phase 4: Python Wrappers (agents/ folder)

**File**: `agents/envs/sb3_wrapper.py`

**Key Changes**:

1. **Import new substrate**:
```python
from meltingpot.configs.substrates import allelopathic_harvest_normative
```

2. **Fix observation staleness in ResidentWrapper**:
```python
def step(self, ego_action):
    # Build resident observations from CURRENT state (not last timestep)
    resident_obs = self._build_resident_observations()

    # Generate actions
    resident_actions = {}
    for player_idx, controller in self._residents.items():
        obs = self._extract_resident_obs(resident_obs, player_idx)
        action = controller.step(obs)
        resident_actions[player_idx] = action

    # Execute
    all_actions = {**resident_actions, **ego_action}
    timestep = self._env.step(all_actions)

    self._last_timestep = timestep
    return timestep

def _build_resident_observations(self):
    """Build from CURRENT dmlab2d state, not last timestep."""
    # Query dmlab2d directly:
    # - Get permitted_color from PermittedColorHolder
    # - Get world_step from PermittedColorHolder
    # - For each resident:
    #   - Get avatar positions from Transform
    #   - Get body_color from ColorZapper
    #   - Get immunity from SimpleZapSanctioning
    #   - Build nearby_agents list

    # NO event parsing - direct component queries

    return observations
```

3. **Distributional mode support**:
```python
def reset(self):
    if self.multi_community_mode:
        # Sample new permitted color
        new_color = self._community_rng.choice([1, 2, 3])
        # Update build call with new permitted_color_index

    timestep = self._env.reset()
    return timestep
```

**File**: `agents/residents/scripted_residents.py`

Update `_is_violation()` if needed (should already be correct):
```python
def _is_violation(self, agent_info, permitted, world_step, grace_period):
    """Check if agent is violator based on BODY COLOR."""
    body_color = agent_info.get('body_color', 0)

    if body_color == 0:  # GREY
        # Violator only after grace period
        return world_step >= grace_period
    else:
        # Violator if body_color != permitted_color
        return body_color != permitted
```

---

## Components Summary

**REMOVE from old implementation**:
- ✗ ImmunityTracker (merged into SimpleZapSanctioning)
- ✗ SimpleZapSanction (wrong location, rewrite in avatar_library.lua)
- ✗ ZapCostApplier (merged into SimpleZapSanctioning)
- ✗ InstitutionalObserver (handled by SimpleZapSanctioning)
- ✗ NormativeRewardTracker (handled by SimpleZapSanctioning)
- ✗ ResidentObserver (build observations directly, no events)

**KEEP (minimal set)**:
- ✓ PermittedColorHolder (scene state in components.lua)
- ✓ SameStepSanctionTracker (scene state in components.lua)
- ✓ SimpleZapSanctioning (avatar overlay in avatar_library.lua - all-in-one)

---

## Timeline Estimate

- **Phase 1 (Lua avatar_library)**: 4-5 hours
- **Phase 2 (Lua components)**: 1 hour
- **Phase 3 (Python config)**: 2-3 hours
- **Phase 4 (Python wrappers)**: 3-4 hours

**Total**: ~10-13 hours

---

## Key Principles

1. **Start from clean base** - revert all meltingpot files to main
2. **Separate substrate** - allelopathic_harvest_normative (no conditional flags)
3. **Correct architecture** - avatar behaviors in avatar_library.lua
4. **All-in-one component** - SimpleZapSanctioning handles all 4 reward components
5. **No event parsing** - build observations directly from components
6. **Fix observation staleness** - use CURRENT dmlab2d state, not last timestep

---

## Success Criteria

After implementation, R2 coverage test should show:
- **Before**: ~45% coverage (observation staleness bug)
- **After**: ~95-100% coverage (correct architecture)

This validates the entire rewrite is correct.
