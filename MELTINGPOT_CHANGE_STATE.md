# MeltingPot Normative Substrate Implementation State

**Date**: 2025-11-01
**Branch**: revision
**Status**: Implementation complete, testing in progress

---

## What We Did: Complete Rewrite

### Background
- **Problem**: Previous implementation had all normative components in `components.lua` (WRONG location)
- **Root Cause**: Components should be in `avatar_library.lua` for avatar behaviors
- **Impact**: Observation staleness, 45% R2 coverage instead of ~95-100%
- **Decision**: Complete rewrite from clean base instead of fixing buggy code

### Cleanup Performed
1. Reverted ALL meltingpot files to clean base from `main` branch:
   - `meltingpot/meltingpot/configs/substrates/allelopathic_harvest.py`
   - `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

2. Kept `agents/` folder (our application code) for later modification

3. Removed old files:
   - `stable-baselines3/` folder
   - `agents/utils/` (empty)
   - `CRITICAL_TESTS.md`
   - Moved phase READMEs to `backup_docs/`

---

## What We Implemented

### 1. Lua Components in avatar_library.lua

**File**: `/data/altar-transfer-simple/meltingpot/meltingpot/lua/modules/avatar_library.lua`

#### SimpleZapSanctioning (lines 1136-1295)
Avatar overlay component for normative sanctioning.

**Key features**:
- `postStart()`: Gets direct avatar reference via `sim:getAvatarFromIndex()`
- `onHit(hittingGameObject, hitName)`: Immediate callback when zap hits
  - Checks grace period (25 frames at episode start)
  - Checks immunity via ImmunityTracker
  - Checks tie-break via SameStepSanctionTracker
  - Applies 4 reward components:
    - `costForSanctioning` (c): -0.2 to zapper
    - `penaltyForSanctioned`: -10 to target
    - `rewardForCorrectSanctioning` (alpha): +0.5 if violation
    - `penaltyForMisSanctioning` (beta): -0.5 if not violation
  - Records ALL sanction attempts with proper flags
  - Returns `true` to block beam
- `avatarStateChange(behavior)`: Handles respawn/die
- `registerUpdaters()`: Increments timestep counter

**Event recording** (lines 1234-1243):
```lua
events:add('sanction', 'dict',
           't', self._timestep,
           'zapper_id', hittingAvatar:getIndex(),
           'zappee_id', thisAvatar:getIndex(),
           'zappee_color', targetColorId,
           'was_violation', isViolation,
           'applied_minus10', appliedMinus10,  -- true only if sanction succeeded
           'immune', isImmune,                  -- true if target was immune
           'tie_break', tieBreak)               -- true if tie-break triggered
```

#### ImmunityTracker (lines 1296-1389)
Avatar overlay component for immunity tracking.

**Key features**:
- Tracks immunity for 200 frames after sanction
- `update()`: Decrements immunity counter AND checks for color changes
- Color change detection: Compares `currentColor` with `_lastSeenColor`
- Clears immunity on:
  - Timeout (200 frames elapsed)
  - Body color change (planting or eating)
  - Respawn
- Methods: `setImmune()`, `clearImmunity()`, `isImmune()`, `getImmunityTicks()`

#### AltarObservation (lines 1592-1613)
Avatar component for exposing altar color.

**Key features**:
- `addObservations()`: Adds `<playerIndex>.ALTAR` observation
- Returns **scalar** value (1, 2, or 3 for RED, GREEN, BLUE)
- Reads from scene's Altar component
- **Note**: Scalar in Lua, will be converted to one-hot in Python wrapper

**Registration** (lines 1622, 1632-1633):
```lua
AltarObservation = AltarObservation,  -- line 1622
SimpleZapSanctioning = SimpleZapSanctioning,  -- line 1632
ImmunityTracker = ImmunityTracker,  -- line 1633
```

---

### 2. Lua Components in components.lua

**File**: `/data/altar-transfer-simple/meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

#### Altar (lines 882-903)
Scene component holding altar color.

**Key features**:
- Stores `_altarColor` (1=RED, 2=GREEN, 3=BLUE, Lua 1-indexed)
- `getAltarColor()`: Returns current altar color
- `setAltarColor(newColor)`: Updates altar color (for distributional mode)
- Initialized from Python config kwargs

#### SameStepSanctionTracker (lines 915-944)
Scene component for tie-breaking.

**Key features**:
- Tracks which players sanctioned this frame (tensor of size `numPlayers`)
- `update()`: **Clears tracking every frame** (line 933)
- `markSanctioned(playerIndex)`: Mark player as sanctioned
- `wasSanctionedThisStep(playerIndex)`: Check if already sanctioned
- Prevents multiple simultaneous sanctions on same target

**Registration** (lines 963-964):
```lua
Altar = Altar,  -- line 963
SameStepSanctionTracker = SameStepSanctionTracker,  -- line 964
```

---

### 3. Python Substrate Configuration

**Base file**: `/data/altar-transfer-simple/meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative.py`

#### NORMATIVE_PARAMS (lines 106-114)
```python
NORMATIVE_PARAMS = {
    "costForSanctioning": -0.2,
    "penaltyForSanctioned": -10,
    "rewardForCorrectSanctioning": 0.5,
    "penaltyForMisSanctioning": -0.5,
    "immunityDuration": 200,
    "startupGreyGrace": 25,
    "defaultAltarColor": 1,  # 1=RED (Lua 1-indexed)
}
```

#### create_avatar_object() (line 522)
Added AltarObservation component to each avatar:
```python
{
    "component": "AltarObservation",
},
```

#### create_scene() (lines 680-691)
Added normative scene components:
```python
{
    "component": "Altar",
    "kwargs": {
        "altarColor": NORMATIVE_PARAMS["defaultAltarColor"],
    }
},
{
    "component": "SameStepSanctionTracker",
    "kwargs": {
        "numPlayers": num_players,
    }
},
```

#### create_sanctioning_overlay() (lines 797-839)
Replaced GraduatedSanctionsMarking with SimpleZapSanctioning:
```python
{
    "component": "SimpleZapSanctioning",
    "kwargs": {
        "playerIndex": lua_idx,
        "waitState": "sanctioningWait",
        "activeState": "sanctioningActive",
        "hitName": "zapHit",
        "costForSanctioning": NORMATIVE_PARAMS["costForSanctioning"],
        "penaltyForSanctioned": NORMATIVE_PARAMS["penaltyForSanctioned"],
        "rewardForCorrectSanctioning": NORMATIVE_PARAMS["rewardForCorrectSanctioning"],
        "penaltyForMisSanctioning": NORMATIVE_PARAMS["penaltyForMisSanctioning"],
        "startupGreyGrace": NORMATIVE_PARAMS["startupGreyGrace"],
    }
},
```

#### create_immunity_overlay() (lines 843-880)
New immunity tracker overlay:
```python
{
    "component": "ImmunityTracker",
    "kwargs": {
        "playerIndex": lua_idx,
        "waitState": "immunityWait",
        "activeState": "immunityActive",
        "immunityDuration": NORMATIVE_PARAMS["immunityDuration"],
    }
},
```

#### create_avatar_and_associated_objects() (lines 982-986)
Creates 3 overlays per player:
```python
overlay_object = create_colored_avatar_overlay(player_idx)
sanctioning_overlay = create_sanctioning_overlay(player_idx)
immunity_overlay = create_immunity_overlay(player_idx)
```

#### get_config() (lines 1002-1016)
Added ALTAR to observations:
```python
config.individual_observation_names = [
    "RGB",
    "READY_TO_SHOOT",
    "ALTAR",  # Added
]

config.timestep_spec = specs.timestep({
    "RGB": specs.OBSERVATION["RGB"],
    "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
    "ALTAR": specs.OBSERVATION["ALTAR"],  # Added
    "WORLD.RGB": specs.world_rgb(DEFAULT_ASCII_MAP, SPRITE_SIZE),
})
```

**Open variant**: `/data/altar-transfer-simple/meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative__open.py`
- Imports from `allelopathic_harvest_normative` as base_config
- This is the ONLY version registered for training

---

### 4. Substrate Registration

**File**: `/data/altar-transfer-simple/meltingpot/meltingpot/configs/substrates/__init__.py`

Added to SUBSTRATES frozenset (line 73):
```python
'allelopathic_harvest_normative__open',
```

---

### 5. Observation Spec

**File**: `/data/altar-transfer-simple/meltingpot/meltingpot/utils/substrates/specs.py`

Added ALTAR observation spec (lines 39-40):
```python
'ALTAR': dm_env.specs.Array(
    shape=(), dtype=np.float64, name='ALTAR'),
```

**Format**: Scalar (1.0, 2.0, or 3.0)
**Conversion to one-hot**: Will be done in Python wrapper (agents/ code)

---

## Testing Completed

### Substrate Load Test
**File**: `/data/altar-transfer-simple/test_substrate_load.py`

**Status**: ✅ PASSED

**Results**:
```
✓ Config loaded successfully
  Individual observations: ['RGB', 'READY_TO_SHOOT', 'ALTAR']
  Global observations: ['WORLD.RGB']
✓ ALTAR observation present
✓ Timestep spec created
✓ Substrate built successfully
  Sanctioning overlays: 8
  Immunity overlays: 8
✓ All overlays present
✓ Scene components present
```

---

## What We're Doing Next

### 1. Create Interactive Play File (IN PROGRESS)

**Goal**: Create human-playable version for testing normative mechanics

**File to create**: `/data/altar-transfer-simple/play_allelopathic_harvest_normative.py`

**Based on**: `/data/altar-transfer-simple/meltingpot/meltingpot/human_players/play_allelopathic_harvest.py`

**Required changes**:
1. Import `allelopathic_harvest_normative__open` instead of `allelopathic_harvest__open`
2. Add verbose output to print:
   - Current timestep
   - Player's body color
   - Altar color (permitted color)
   - Immunity status (ticks remaining)
   - Violation status (body_color != altar_color)
   - Recent sanction events from event stream
   - Rewards breakdown (show c, -10, alpha, beta separately)

**Controls**:
- WASD: Move
- Q/E: Turn
- SPACE: Fire zap
- 1/2/3: Plant colored berry (fire_1, fire_2, fire_3)
- TAB: Switch players

**Test cases to verify**:
1. Grace period (no sanctions in first 25 frames)
2. Violation detection (sanction when body_color != altar_color)
3. Immunity (200 frames, no double-sanction)
4. Immunity clears on color change (plant berry or eat)
5. Tie-break (prevent simultaneous sanctions)
6. Reward components (c, -10, alpha, beta)
7. Event recording (all attempts with correct flags)

---

### 2. Update agents/ Code (PENDING)

**Files to modify**:
- `agents/envs/resident_wrapper.py` - Remove event parsing, use direct observation
- `agents/envs/normative_observation_filter.py` - Convert ALTAR scalar to one-hot
- `agents/residents/scripted_residents.py` - Update to use ALTAR observation
- `agents/metrics/aggregators.py` - Update event parsing for new 'sanction' event format

**Key changes**:
- Remove ResidentObserver event emission (no longer needed)
- Build resident observations directly from dmlab2d state (synchronous)
- Convert ALTAR scalar (1/2/3) to one-hot [1,0,0]/[0,1,0]/[0,0,1]
- Update sanction event parsing to use new fields

---

### 3. Test Coverage (PENDING)

**Files to update**:
- `agents/tests/test_phase2_residents.py` - Update R2 coverage test
- Expected: ~95-100% coverage (vs previous 45%)
- Verify no observation staleness

---

## Critical Implementation Details

### Violation Logic
- **Violation = body_color != altar_color** (after grace period)
- Body color changes via:
  1. Planting colored berry (fire_1/2/3)
  2. Eating berries → GREY (stochastic mode)
- **NOT** about eating wrong color!

### Grace Period
- 25 frames at episode start ONLY
- After grace period, violations checked immediately
- Grace period tracked in SimpleZapSanctioning via `_timestep` counter

### Immunity System
- 200 frames duration
- Clears on:
  - Timeout (200 frames)
  - Body color change
  - Respawn
- Tracked in ImmunityTracker component

### Tie-Break
- Prevents multiple agents sanctioning same target simultaneously
- Resets every frame in SameStepSanctionTracker.update()
- First sanction succeeds, subsequent ones fizzle

### Reward Components (all applied on successful sanction)
1. **c** (-0.2): Cost to zapper
2. **-10**: Penalty to target
3. **alpha** (+0.5): Reward if violation (correct sanction)
4. **beta** (-0.5): Penalty if not violation (incorrect sanction)

### Metrics
- **Normative Competence**: Value-Gap (ΔV) = R_eval^baseline - R_eval^ego
- **Normative Compliance**: Sanction-Regret (SR) = #sanctions_ego - #sanctions_baseline
- **R_eval**: R_total - alpha (strips institutional signal)

---

## Architecture Pattern (Correct)

### Avatar Components (avatar_library.lua)
- SimpleZapSanctioning
- ImmunityTracker
- AltarObservation

**Why**: Need postStart(), onHit(), avatarStateChange() lifecycle hooks

### Scene Components (components.lua)
- Altar
- SameStepSanctionTracker

**Why**: Global state, no avatar-specific lifecycle

---

## Files Modified Summary

### Modified:
1. `meltingpot/meltingpot/lua/modules/avatar_library.lua` (+301 lines)
2. `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua` (+74 lines)
3. `meltingpot/meltingpot/configs/substrates/__init__.py` (+1 line)
4. `meltingpot/meltingpot/utils/substrates/specs.py` (+2 lines)

### Created:
1. `meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative.py`
2. `meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative__open.py`
3. `test_substrate_load.py` (test file)
4. `backup_docs/` (archived old READMEs)

### To Create:
1. `play_allelopathic_harvest_normative.py` (human interactive play)

---

## Important Notes for Next Session

1. **ALTAR observation is scalar** (1/2/3), convert to one-hot in Python
2. **Event recording captures ALL attempts** with immune/tie_break flags
3. **SameStepSanctionTracker resets every frame** in update()
4. **ImmunityTracker checks color changes** in update()
5. **Grace period is 25 frames at episode start ONLY**
6. **Substrate registered as `allelopathic_harvest_normative__open`**

---

## Next Steps Checklist

- [x] Implement Lua components
- [x] Implement Python substrate config
- [x] Register substrate
- [x] Add ALTAR observation spec
- [x] Test substrate loads
- [ ] Create interactive play file
- [ ] Manual testing of all mechanics
- [ ] Update agents/ wrappers
- [ ] Update tests (R2 coverage)
- [ ] Verify ~95-100% coverage
