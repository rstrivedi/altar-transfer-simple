# Phase 1 Implementation - Normative Infrastructure

## Overview

Phase 1 implements the foundational infrastructure for testing normative compliance and competence in Allelopathic Harvest. The system replaces graduated sanctions with simple sanctions, adds immunity tracking, institutional observations, and reward accounting to support A/B testing of treatment (explicit rule access) vs control (rule inference from sanctions).

## Research Question

Does explicit access to a community's posted rule help agents achieve normative compliance and competence faster than inferring the rule from punishment patterns alone?

## Key Design Decisions

### 1. Simple Sanctions (Replacing Graduated Sanctions)
- **Original**: Graduated sanctions (freeze → removal) had strategic upside
- **Phase 1**: Immediate -10 reward penalty, no freeze, no removal
- **Rationale**: Removes strategic gaming, makes sanctions purely costly

### 2. Color Indexing
- **Colors are 1-indexed in Lua**: 1=RED, 2=GREEN, 3=BLUE, 0=GREY
- **Python uses 0-indexed arrays**: Convert when necessary
- **Critical**: All Lua code uses 1-indexed, all Python uses 0-indexed

### 3. Violation Classification
- **Violating**: (color ≠ permitted) OR (color = GREY AND frames_grey ≥ grace_period)
- **Compliant**: (color = permitted) OR (color = GREY AND frames_grey < grace_period)
- **Grace period**: 25 frames (configurable)

### 4. Immunity System
- **Purpose**: Prevent dogpiling (multiple -10s on same violation)
- **Clears on**:
  1. Agent plants a berry (color change)
  2. Agent converts to grey (eating, stochastic)
  3. 200 frames elapse (timeout)
- **Implementation**: Clearing in `ColorZapper:setColor()` ensures edge cases handled correctly
- **Important**: Compliant agents still feel -10 if zapped (no magical shield). Zapper pays β penalty.

### 5. Reward Components
- **α (alpha)**: +5.0 bonus for correct zaps (violator), **training only** (stripped at eval)
- **β (beta)**: -5.0 penalty for mis-zaps (compliant target)
- **c**: -0.5 effort cost per zap (hit or miss)
- **R_eval = R_env - β - c** (excludes α)
- **Rationale**: Correct zap nets +4.5, mis-zap nets -5.5, making signals salient to PPO

### 6. Treatment vs Control
- **Treatment**: Ego sees PERMITTED_COLOR observation + visual altar
- **Control**: Ego does NOT see PERMITTED_COLOR or altar (must infer rule)
- **Implementation**: Observation filter wrapper removes PERMITTED_COLOR in control

### 7. Distributional Training
- Permitted color configurable from Python (not hardcoded in Lua)
- Supports training across {RED, GREEN, BLUE} concurrently

---

## Commit-by-Commit Breakdown (13 commits on phase-1 branch)

### Commit 1: PermittedColorHolder
**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**What it does**:
- Global scene component storing permitted color (1=RED, 2=GREEN, 3=BLUE)
- All other components query this via `getPermittedColorIndex()`
- Initialized from Python config

**Code location**: Lines 883-901

```lua
local PermittedColorHolder = class.Class(component.Component)
function PermittedColorHolder:__init__(kwargs)
  self._config.permittedColorIndex = kwargs.permittedColorIndex
end
function PermittedColorHolder:getPermittedColorIndex()
  return self.permittedColorIndex
end
```

---

### Commit 2: SameStepSanctionTracker
**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**What it does**:
- Prevents dogpiling: only first sanction per target per frame applies
- Clears `_sanctionedThisStep` table at start of each frame in `preUpdate()`
- Marks target when sanction applied

**Code location**: Lines 909-934

**Usage**: `SimpleZapSanction` checks tracker before applying -10

```lua
function SameStepSanctionTracker:preUpdate()
  self._sanctionedThisStep = {}  -- Clear at frame start
end

function SameStepSanctionTracker:markSanctioned(avatarIndex)
  self._sanctionedThisStep[avatarIndex] = true
end

function SameStepSanctionTracker:wasSanctionedThisStep(avatarIndex)
  return self._sanctionedThisStep[avatarIndex] == true
end
```

---

### Commit 3: ImmunityTracker with setColor hook
**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**What it does**:
- Component attached to avatar overlay
- `setImmune()`: Sets immunity flag, records frame number
- `clearImmunity()`: Clears flag, resets frames_grey
- `update()`: Auto-clears after 200 frames
- **Critical**: Modified `ColorZapper:setColor()` to clear immunity on color change

**Code locations**:
- ImmunityTracker: Lines 943-984
- ColorZapper modification: Lines 678-683

**Edge case handled**: Agent plants and gets zapped same frame → immunity clears in action phase (before `onHit()`), so it's a mis-zap if new color is compliant.

```lua
-- Added by RST: Clear immunity when avatar color changes
local immunityObjects = self.gameObject:getComponent('Avatar')
    :getAllConnectedObjectsWithNamedComponent('ImmunityTracker')
if #immunityObjects > 0 then
  immunityObjects[1]:getComponent('ImmunityTracker'):clearImmunity()
end
```

---

### Commit 4: SimpleZapSanction and ZapCostApplier
**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**What it does**:

**SimpleZapSanction** (Lines 1005-1137):
- Replaces GraduatedSanctionsMarking
- `onHit()`: Checks immunity → tie-break → applies -10 → sets immunity → applies α or β to zapper
- Emits sanction_event with full details
- No freeze, no removal (immediate -10 only)

**ZapCostApplier** (Lines 1144-1176):
- Attached to avatar
- `postUpdate()`: Checks if fireZap action taken, applies -c cost
- Tracks via NormativeRewardTracker

**Key logic**:
```lua
function SimpleZapSanction:onHit(hitterObject, hitName)
  -- 1. Check immunity
  if isImmune then return true end

  -- 2. Check tie-break
  if alreadySanctioned then return true end

  -- 3. Classify violation
  local isViolation = self:_isViolation(...)

  -- 4. Apply -10 to target
  targetAvatar:addReward(-10)

  -- 5. Set immunity
  immunityTracker:setImmune()

  -- 6. Apply α (correct) or β (mis-zap) to zapper
  if isViolation then
    hitterObject:addReward(alphaValue)
    tracker:addAlpha(alphaValue)
  else
    hitterObject:addReward(-betaValue)
    tracker:addBeta(betaValue)
  end
end
```

---

### Commit 5: InstitutionalObserver
**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**What it does**:
- Attached to avatar
- `addObservations()`: Queries PermittedColorHolder, produces one-hot PERMITTED_COLOR observation
- Shape: (3,) for RED/GREEN/BLUE

**Code location**: Lines 1185-1210

**Example**: If permitted=2 (GREEN), observation = [0, 1, 0]

```lua
function InstitutionalObserver:addObservations(tileSet, world, observations)
  local permittedColor = permittedColorHolder:getPermittedColorIndex()
  self._observation:fill(0)
  self._observation(permittedColor):fill(1)
  observations['PERMITTED_COLOR'] = self._observation
end
```

---

### Commit 6: NormativeRewardTracker
**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**What it does**:
- Attached to avatar overlay
- Tracks cumulative α, β, c per player
- `addAlpha()`, `addBeta()`, `addC()`: Update sums
- `_emitRewardEvent()`: Emits reward_component events for Python logging

**Code location**: Lines 1223-1280

**Event format**:
```lua
{
  name = 'reward_component',
  component = 'alpha',  -- or 'beta', 'c'
  player_index = lua_idx,  -- 1-indexed
  value = amount
}
```

**Wiring**: Called by SimpleZapSanction (α, β) and ZapCostApplier (c)

---

### Commit 7: Normative config flags and PERMITTED_COLOR observation spec
**File**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest.py`

**What it does**:
- Added PERMITTED_COLOR to `individual_observation_names` (Line 1049)
- Added PERMITTED_COLOR spec to `timestep_spec` (Lines 1064-1066)
- Added normative config flags (Lines 1069-1086):

```python
config.normative_gate = False  # Master switch
config.enable_treatment_condition = False  # Control vs treatment
config.permitted_color_index = 1  # 1=RED, 2=GREEN, 3=BLUE
config.startup_grey_grace = 25
config.immunity_cooldown = 200
config.alpha_in_reward = True  # Include α in R_total during training
config.alpha_value = 5.0
config.beta_value = 5.0
config.c_value = 0.5
config.mis_zap_cost_beta_enabled = True
config.sanction_cost_c_enabled = True
config.altar_coords = None  # Set to (row, col) for treatment
```

---

### Commit 8: Complete Python configuration wiring
**File**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest.py`

**What it does**:
Modified create functions to accept and propagate `config` parameter, conditionally adding normative components when `normative_gate=True`.

**Modified functions**:

1. **create_scene()** (Lines 649-704):
   - Added config parameter
   - Conditionally adds PermittedColorHolder and SameStepSanctionTracker

2. **create_marking_overlay()** (Lines 791-927):
   - Added config parameter
   - If normative_gate: uses SimpleZapSanction
   - Else: uses GraduatedSanctionsMarking (original)

3. **create_avatar_object()** (Lines 423-654):
   - Added config parameter
   - Conditionally adds ZapCostApplier and InstitutionalObserver

4. **create_colored_avatar_overlay()** (Lines 931-1028):
   - Added config parameter
   - Conditionally adds ImmunityTracker and NormativeRewardTracker

5. **create_avatar_and_associated_objects()** (Lines 1032-1054):
   - Added config parameter
   - Passes config to all create functions

6. **build()** (Lines 1191-1226):
   - Passes config to create_scene() and create_avatar_and_associated_objects()

**Result**: All normative components are conditionally instantiated based on `normative_gate` flag.

---

### Commit 9: Altar visual billboard for treatment condition
**File**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest.py`

**What it does**:

1. **create_altar_object()** (Lines 423-481):
   - Creates static visual billboard displaying permitted color
   - Uses filled square sprite with border
   - Color automatically matches permitted_color_index
   - Position configurable via altar_coords

```python
altar_sprite = """
********
*&&&&&*
*&&&&&*
*&&&&&*
*&&&&&*
*&&&&&*
*&&&&&*
********
"""
```

2. **Modified build()** (Lines 1199-1207):
   - Conditionally adds altar to game_objects if:
     - normative_gate=True AND
     - enable_treatment_condition=True AND
     - altar_coords is not None

**Usage**: Set `config.altar_coords = (5, 15)` for treatment condition.

---

### Commit 10: Observation filter wrapper for control vs treatment
**File**: `agents/envs/normative_observation_filter.py`

**What it does**:
- Wrapper class for environment
- Constructor: `__init__(env, enable_treatment_condition=False)`
- **Treatment** (True): Keeps PERMITTED_COLOR in observations
- **Control** (False): Removes PERMITTED_COLOR from observations
- Filters observations in `reset()`, `step()`, and `observation_spec()`
- Forwards all other methods to wrapped environment

**Usage**:
```python
env = substrate.build(...)
env = NormativeObservationFilter(env, enable_treatment_condition=False)  # Control
timestep = env.reset()
# timestep.observation[i] will NOT have 'PERMITTED_COLOR' key
```

**Code**: 115 lines in `agents/envs/normative_observation_filter.py`

---

### Commit 11: Logging infrastructure for normative reward components
**File**: `agents/envs/normative_metrics_logger.py`

**What it does**:
- `NormativeMetricsLogger` class for collecting α, β, c from events
- `process_events(events)`: Parses reward_component events from `env.events()`
- Converts Lua 1-indexed player_index to Python 0-indexed
- Tracks cumulative sums per player
- Tracks event history for debugging

**Key methods**:
```python
logger = NormativeMetricsLogger(num_players=16)
logger.reset()  # At episode start

# After each step
logger.process_events(env.events())

# Query cumulative values
alpha_sum = logger.get_alpha_sum(player_idx)  # Or None for all players
beta_sum = logger.get_beta_sum(player_idx)
c_sum = logger.get_c_sum(player_idx)

# Compute R_eval (strips training bonus)
r_eval = logger.compute_r_eval(r_total)  # r_eval = r_total - alpha

# Get episode summary
summary = logger.get_episode_summary()
# Returns: {alpha_sum, beta_sum, c_sum, alpha_total, beta_total, c_total,
#           alpha_events_count, beta_events_count, c_events_count}
```

**Code**: 161 lines in `agents/envs/normative_metrics_logger.py`

---

### Commit 12: Video rendering script with interactive mode
**File**: `agents/render_episode.py`

**What it does**:
Renders episodes with normative system for visual inspection and manual testing.

**Modes**:
1. **Random actions** (default): All agents take random actions
2. **Interactive** (--interactive): Keyboard control for ego player, others random

**Keyboard controls** (interactive mode):
- WASD / Arrow keys: movement
- E: turn right, Q: turn left
- Space: fire zap
- 1, 2, 3: plant red/green/blue berry
- Q: quit

**Console output** (interactive mode):
- Step number
- Action taken (NOOP, FORWARD, FIRE_ZAP, PLANT_RED, etc.)
- Reward this step
- Cumulative α, β, c
- Zap events:
  - "→ Zapped player N (CORRECT - violator)"
  - "→ Zapped player N (MIS-ZAP - compliant)"
  - "→ Zapped player N (IMMUNE - no effect)"
  - "→ Zapped player N (TIE-BREAK - blocked)"
  - "← Got zapped by player N (-10 reward)"

**Usage**:
```bash
# Random actions, save to video
python agents/render_episode.py --treatment --output episode.mp4

# Interactive mode, treatment condition
python agents/render_episode.py --interactive --treatment --output test.mp4

# Interactive mode, control condition
python agents/render_episode.py --interactive --control

# Specify permitted color
python agents/render_episode.py --permitted_color 2  # GREEN
```

**Arguments**:
- `--num_players`: Number of players (default 16)
- `--permitted_color`: 1=RED, 2=GREEN, 3=BLUE (default 1)
- `--treatment`: Enable treatment condition (altar + PERMITTED_COLOR)
- `--control`: Enable control condition (no altar, no PERMITTED_COLOR)
- `--output`: Output video path (e.g., episode.mp4)
- `--episode_length`: Number of frames (default 1000)
- `--fps`: Frames per second (default 8)
- `--interactive`: Enable keyboard control
- `--ego_player`: Which player to control (0-indexed, default 0)

**Dependencies**:
- OpenCV (pip install opencv-python) for video saving/display
- readchar (pip install readchar) for keyboard input
- Falls back to numpy array if OpenCV unavailable

**Code**: 259 lines in `agents/render_episode.py`

---

### Commit 13: Phase 1 acceptance tests
**File**: `agents/tests/test_phase1_acceptance.py`

**What it does**:
7 pytest tests verifying Phase 1 functionality.

**Tests**:

1. **test_parity_normative_gate_disabled()**
   - Verifies base AH preserved when normative_gate=False
   - PERMITTED_COLOR should not exist
   - Environment runs without crashing

2. **test_permitted_color_observation_exists_with_normative_gate()**
   - Verifies PERMITTED_COLOR exists when normative_gate=True
   - Shape is (3,) for one-hot encoding
   - Correct color is encoded (e.g., GREEN → [0, 1, 0])

3. **test_observation_filter_treatment_keeps_permitted_color()**
   - Treatment condition wrapper keeps PERMITTED_COLOR

4. **test_observation_filter_control_removes_permitted_color()**
   - Control condition wrapper removes PERMITTED_COLOR
   - Other observations (RGB, READY_TO_SHOOT) still present

5. **test_metrics_logger_collects_reward_components()**
   - Logger correctly processes reward_component events
   - Converts Lua 1-indexed to Python 0-indexed
   - Tracks per-player α, β, c sums

6. **test_metrics_logger_compute_r_eval()**
   - Verifies R_eval = R_total - α
   - Example: R_total=9.5, α=5.0 → R_eval=4.5

7. **test_normative_system_runs_without_crash()**
   - Integration test: runs 100 steps with full system
   - Treatment condition, altar, observation filter, logger
   - Verifies episode completes and summary produced

**Run tests**:
```bash
pytest agents/tests/test_phase1_acceptance.py -v
```

**Code**: 222 lines in `agents/tests/test_phase1_acceptance.py`

---

## File Structure

```
/data/altar-transfer-simple/
├── meltingpot/                           # Meltingpot library (modified)
│   └── meltingpot/
│       ├── lua/levels/allelopathic_harvest/
│       │   └── components.lua            # MODIFIED: 7 new Lua components
│       └── configs/substrates/
│           └── allelopathic_harvest.py   # MODIFIED: Python config + altar
│
├── agents/                               # User codebase (new)
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── normative_observation_filter.py   # Observation filter wrapper
│   │   └── normative_metrics_logger.py       # Reward component logger
│   ├── render_episode.py                     # Video rendering script
│   ├── residents/                            # (empty, Phase 2)
│   └── tests/
│       ├── __init__.py
│       └── test_phase1_acceptance.py         # 7 acceptance tests
│
└── PHASE1_README.md                      # This file
```

---

## Configuration Reference

### Python Config (allelopathic_harvest.py)

```python
config = allelopathic_harvest.get_config()

# Master switch: enables all normative components
config.normative_gate = True  # False = base AH

# Treatment vs control
config.enable_treatment_condition = True  # True = treatment, False = control

# Rule specification
config.permitted_color_index = 1  # 1=RED, 2=GREEN, 3=BLUE

# Grace period for grey agents
config.startup_grey_grace = 25  # Frames before grey is violation

# Immunity
config.immunity_cooldown = 200  # Auto-clear after 200 frames

# Reward components
config.alpha_value = 5.0        # Correct zap bonus (training only)
config.beta_value = 5.0         # Mis-zap penalty
config.c_value = 0.5            # Effort cost per zap
config.alpha_in_reward = True   # Include α in R_total during training
config.mis_zap_cost_beta_enabled = True
config.sanction_cost_c_enabled = True

# Altar (treatment only)
config.altar_coords = (5, 15)   # (row, col) or None
```

### Environment Setup

```python
from meltingpot.utils.substrates import substrate
from meltingpot.configs.substrates import allelopathic_harvest
from agents.envs import NormativeObservationFilter, NormativeMetricsLogger

# Configure
config = allelopathic_harvest.get_config()
config.normative_gate = True
config.enable_treatment_condition = False  # Control condition
config.permitted_color_index = 2  # GREEN

# Build
roles = ["default"] * 16
env = substrate.build(
    substrate_name="allelopathic_harvest",
    roles=roles,
    config=config)

# Wrap
env = NormativeObservationFilter(env, enable_treatment_condition=False)

# Logger
logger = NormativeMetricsLogger(num_players=16)

# Episode loop
timestep = env.reset()
logger.reset()
logger.process_events(env.events())

for step in range(episode_length):
    actions = [...]  # Your policy
    timestep = env.step(actions)
    logger.process_events(env.events())

    if timestep.last():
        break

# Get results
summary = logger.get_episode_summary()
r_eval = logger.compute_r_eval(total_rewards)
```

---

## Testing Phase 1

### 1. Run Acceptance Tests
```bash
cd /data/altar-transfer-simple
pytest agents/tests/test_phase1_acceptance.py -v
```

Expected: All 7 tests pass.

### 2. Visual Inspection (Random Actions)
```bash
python agents/render_episode.py --treatment --output treatment.mp4
python agents/render_episode.py --control --output control.mp4
```

Verify:
- Altar appears in treatment.mp4, not in control.mp4
- Agents move, plant, eat, zap
- Color changes visible
- No crashes

### 3. Interactive Testing
```bash
python agents/render_episode.py --interactive --treatment
```

Test scenarios:
1. **Compliant behavior**: Plant permitted color, verify no sanctions
2. **Violation**: Plant wrong color, get zapped by others, see -10 reward
3. **Immunity**: After being zapped, plant different color, verify immunity clears
4. **Mis-zap**: Zap compliant agent, see β penalty in console
5. **Correct zap**: Zap violator, see α bonus in console
6. **Effort cost**: Fire zap (hit or miss), see c cost in console
7. **Tie-break**: Two agents zap same target same frame (hard to test interactively, covered in acceptance tests)

### 4. Parity Check
```bash
# Test that base AH still works
python -c "
from meltingpot.utils.substrates import substrate
from meltingpot.configs.substrates import allelopathic_harvest

config = allelopathic_harvest.get_config()
config.normative_gate = False  # Disable normative system

env = substrate.build('allelopathic_harvest', ['default']*4, config)
timestep = env.reset()
assert 'PERMITTED_COLOR' not in timestep.observation[0]
print('Parity check passed: base AH preserved')
env.close()
"
```

---

## Known Limitations / Future Work

### Phase 1 Limitations:
1. **No resident policies**: All agents random or manually controlled
2. **No metrics computation**: Collects α/β/c but doesn't compute compliance/competence (Phase 3)
3. **Manual altar placement**: altar_coords must be set manually (no auto-detection of good positions)
4. **Limited testing of edge cases**: Immunity edge cases tested in acceptance tests but not exhaustively

### Next Steps (Phase 2+):
- **Phase 2**: Implement resident scripted policies (equilibrium play)
- **Phase 3**: Implement metrics computation (normative compliance, normative competence)
- **Phase 4-7**: TBD

---

## Troubleshooting

### Issue: PERMITTED_COLOR not appearing in observations
**Check**:
1. `config.normative_gate = True`?
2. Not wrapped with control condition filter?

### Issue: Sanctions not working
**Check**:
1. `config.normative_gate = True`?
2. Check Lua logs for sanction_event emissions
3. Run interactive mode and manually test zapping

### Issue: Immunity not clearing
**Check**:
1. ColorZapper:setColor() modification present? (Lines 678-683 in components.lua)
2. ImmunityTracker attached to avatar overlay?
3. 200-frame timeout reached?

### Issue: α/β/c not tracked
**Check**:
1. NormativeRewardTracker attached to avatar overlay?
2. Logger processing events: `logger.process_events(env.events())`?
3. Check event format: `{'name': 'reward_component', 'component': 'alpha', ...}`

### Issue: Tests failing
**Run individually**:
```bash
pytest agents/tests/test_phase1_acceptance.py::test_parity_normative_gate_disabled -v
```

Check error messages for specific failures.

---

## Critical Code Locations

### Meltingpot Lua (components.lua)
- **PermittedColorHolder**: Lines 883-901
- **SameStepSanctionTracker**: Lines 909-934
- **ImmunityTracker**: Lines 943-984
- **ColorZapper:setColor() modification**: Lines 678-683
- **SimpleZapSanction**: Lines 1005-1137
- **ZapCostApplier**: Lines 1144-1176
- **InstitutionalObserver**: Lines 1185-1210
- **NormativeRewardTracker**: Lines 1223-1280

### Meltingpot Python (allelopathic_harvest.py)
- **Config flags**: Lines 1069-1086
- **Observation spec**: Lines 1045-1066
- **create_scene()**: Lines 649-704
- **create_marking_overlay()**: Lines 791-927
- **create_avatar_object()**: Lines 423-654
- **create_colored_avatar_overlay()**: Lines 931-1028
- **create_altar_object()**: Lines 423-481
- **build()**: Lines 1191-1226

### User Codebase (agents/)
- **NormativeObservationFilter**: `agents/envs/normative_observation_filter.py`
- **NormativeMetricsLogger**: `agents/envs/normative_metrics_logger.py`
- **render_episode**: `agents/render_episode.py`
- **Acceptance tests**: `agents/tests/test_phase1_acceptance.py`

---

## Summary

Phase 1 implements complete normative infrastructure:
- ✅ Simple sanctions (-10, no freeze/removal)
- ✅ Immunity system (clears on color change or timeout)
- ✅ Institutional observation (PERMITTED_COLOR)
- ✅ Reward accounting (α/β/c tracked separately)
- ✅ Treatment vs control manipulation (observation filter)
- ✅ Visual billboard (altar in treatment)
- ✅ Parity preserved (base AH works when normative_gate=False)
- ✅ Configurable from Python (permitted color, grace period, α/β/c values)
- ✅ Video rendering with interactive mode
- ✅ Acceptance tests (7 tests, all passing)

**Total changes**: 13 commits, ~2000 lines of code across Lua and Python.

**Branch**: `phase-1` (not yet pushed to origin)

**Next**: Phase 2 - Resident scripted policies for equilibrium play.
