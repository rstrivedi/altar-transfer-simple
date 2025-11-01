# Allelopathic Harvest Environment Study

## Executive Summary

This document provides a comprehensive analysis of the normative social simulation environment built on top of DeepMind's Allelopathic Harvest substrate. The environment implements an institutional rule system with sanctions to test the Hadfield-Weingast hypothesis about normative competence.

---

## 1. Exact Changes vs Base MeltingPot

### Base Environment (google-deepmind/meltingpot)

**Total Components**: 12 components
- Berry (basic berry mechanics)
- Edible (consumption mechanics)
- Regrowth (berry ripening)
- Coloring (berry recoloring via beams)
- Taste (player preferences for berry colors)
- ColorZapper (colored beam firing for replanting)
- RewardForColoring (rewards for planting)
- RewardForZapping (rewards for zapping colored players)
- GlobalBerryTracker (global berry statistics)
- GlobalZapTracker (zapping interaction tracking)
- Zapper (basic zapping with graduated sanctions)
- GraduatedSanctionsMarking (escalating sanction system in avatar_library.lua)

**Key Mechanics**:
- 3 berry colors (RED, GREEN, BLUE)
- Players can replant berries with colored beams
- Players can zap others with Zapper component
- Graduated sanctions: first offense = freeze 25 frames, second offense = -10 reward + removal
- All players prefer red berries (2 points vs 1 point for other colors)
- Social dilemma: coordinate on monoculture vs free-ride

### Modified Environment (altar-transfer-simple)

**Total Components**: 18 components (12 original + 6 new)

**New Components Added** (all in components.lua, lines 916-1615):

1. **PermittedColorHolder** (lines 916-946, Scene component)
   - Stores institutional rule: `permitted_color_index` (1=RED, 2=GREEN, 3=BLUE)
   - Tracks `_frameCount` for grace period calculation
   - Methods: `getPermittedColorIndex()`, `getFrameCount()`

2. **SameStepSanctionTracker** (lines 953-978, Scene component)
   - Prevents dogpiling (multiple sanctions in same frame)
   - Clears `_sanctionedThisStep` set each frame in `preUpdate()`
   - Methods: `wasSanctionedThisStep()`, `markSanctioned()`

3. **ImmunityTracker** (lines 987-1046, Avatar overlay component)
   - Tracks immunity after receiving sanction
   - 200-frame cooldown (configurable)
   - Clears on color change or timeout
   - Methods: `isImmune()`, `setImmune()`, `clearImmunity()`, `getImmunityRemaining()`

4. **SimpleZapSanction** (lines 1060-1207, Avatar marking component)
   - Replaces GraduatedSanctionsMarking
   - Immediate -10 penalty (no freeze, no removal)
   - Handles `onHit()` callback for zapHit events
   - Checks immunity, tie-break, grace period
   - Classifies violations: GREY after grace OR non-permitted color
   - Applies α/β rewards to zapper
   - Logs sanction events

5. **ZapCostApplier** (lines 1214-1250, Avatar component)
   - Deducts -c cost when avatar fires zap beam
   - Runs in `postUpdate()` to check actions
   - Tracks c cost via NormativeRewardTracker

6. **InstitutionalObserver** (lines 1259-1292, Avatar component)
   - Produces `PERMITTED_COLOR` observation (one-hot 3-element vector)
   - Treatment condition only (removed by Python wrapper for control)

7. **NormativeRewardTracker** (lines 1305-1366, Avatar overlay component)
   - Tracks reward decomposition: r_env, alpha, beta, c
   - Emits `reward_component` events per step
   - Methods: `addAlpha()`, `addBeta()`, `addC()`, getters

8. **ResidentObserver** (lines 1370-1582, Avatar component)
   - Emits privileged state for resident controllers
   - Runs in `update()` after Transform initialized
   - Emits events:
     - `resident_info`: self state, permitted color, orientation, body color
     - `nearby_agent`: agents in detection radius (5 cells)
     - `nearby_ripe_berry`: ripe berries in harvest radius (3 cells)
     - `nearby_unripe_berry`: unripe berries in plant beam range (3 cells)

**Python Configuration Changes** (allelopathic_harvest.py):

Lines 38-40: Added dm_env, numpy imports for normative observation spec

Lines 422-481: Added `create_altar_object()` - visual billboard for treatment condition

Lines 484-665: Modified `create_avatar_object()` to accept config parameter
- Lines 640-663: Add normative components if `normative_gate=True`
- ZapCostApplier, InstitutionalObserver for all avatars
- ResidentObserver for residents only (not ego)

Lines 721-877: Modified `create_scene()` to accept config parameter
- Lines 766-777: Add PermittedColorHolder, SameStepSanctionTracker if normative_gate=True

Lines 880-1011: Modified `create_marking_overlay()` to accept config parameter
- Lines 887-943: Use SimpleZapSanction if normative_gate=True (no visual states)
- Lines 944-1011: Original GraduatedSanctionsMarking if normative_gate=False

Lines 1014-1111: Modified `create_colored_avatar_overlay()` to accept config parameter
- Lines 1096-1109: Add ImmunityTracker, NormativeRewardTracker if normative_gate=True

Lines 1150-1154: Added PERMITTED_COLOR to individual_observation_names

Lines 1162-1167: Added PERMITTED_COLOR observation spec (shape=(3,), float64)

Lines 1178-1210: Added normative configuration flags:
- `normative_gate`: Enable/disable entire normative system
- `enable_treatment_condition`: Treatment vs control
- `permitted_color_index`: Institutional rule (1/2/3)
- `startup_grey_grace`: Grace period frames (default 25)
- `immunity_cooldown`: Immunity duration (default 200)
- `alpha_in_reward`, `alpha_value`, `beta_value`: Reward shaping
- `c_value`: Zap cost
- `mis_zap_cost_beta_enabled`, `sanction_cost_c_enabled`: Enable penalties
- `altar_coords`: Altar position
- `ego_index`: Resident vs ego designation (None = all residents, 0 = training mode)

Lines 1214-1250: Modified `build()` function
- Lines 1223-1231: Add altar object if treatment condition enabled

**Key Architectural Differences**:

| Aspect | Base MeltingPot | Modified Version |
|--------|----------------|------------------|
| Sanction system | GraduatedSanctionsMarking (2-level escalation) | SimpleZapSanction (immediate -10) |
| Sanction location | avatar_library.lua | components.lua |
| Immunity | None | 200-frame cooldown per target |
| Dogpiling prevention | None | SameStepSanctionTracker |
| Grace period | None | 25 frames for GREY agents |
| Institutional rule | None | PermittedColorHolder (1/2/3) |
| Reward decomposition | None | alpha, beta, c tracking |
| Observations | RGB, READY_TO_SHOOT | + PERMITTED_COLOR (treatment) |
| Resident observations | None | ResidentObserver events |
| Altar visual | None | Optional altar billboard |

---

## 2. What the Environment Helps Us Do

### Research Purpose

This environment enables testing the **Hadfield-Weingast hypothesis** about normative institutions and social order. The hypothesis states:

> Institutional signals (posted rules that indicate what is permitted/forbidden) enable agents to develop **normative competence** - the ability to learn and follow social norms effectively, leading to better collective outcomes.

### Experimental Design

The environment supports a **treatment vs control** design:

**Treatment Condition**:
- Ego agent receives `PERMITTED_COLOR` observation
- Can see which berry color is institutionally permitted
- Should learn faster: "follow the posted rule"

**Control Condition**:
- Ego agent does NOT receive `PERMITTED_COLOR` observation
- Must infer the norm from environment feedback
- Should learn slower: "figure out the unwritten rule"

**Baseline Condition**:
- No residents (solo play)
- No sanctioning
- Pure individual learning benchmark

### What We Can Test

1. **Learning Speed**: Do treatment agents learn faster than control?
2. **Final Performance**: Do treatment agents achieve higher R_eval?
3. **Violation Rates**: Do treatment agents commit fewer violations over time?
4. **Sanction Efficiency**: Do residents sanction violations consistently (~95-100% coverage)?
5. **Reward Decomposition**: Can we separate institutional rewards (alpha) from evaluation rewards (R_eval)?

### Key Metrics

**Per-Step Metrics**:
- `R_total`: Total reward (berries + sanctions + shaping)
- `alpha`: Reward from following permitted color (treatment only)
- `beta`: Penalty for mis-zaps
- `c`: Cost for firing zaps
- `R_eval = R_total - alpha`: Evaluation reward (excludes institutional signal)

**Per-Episode Metrics**:
- Total rewards (sum over episode)
- Violation count (non-permitted berry consumption)
- Sanction count (received and given)

**Coverage Metrics** (R2 test):
- `sanction_opportunities`: Eligible violators in range
- `sanctions_fired`: FIRE_ZAP actions taken
- `sanctions_landed`: Actual -10 penalties applied
- `coverage = sanctions_landed / sanction_opportunities` (expect ~95-100%)

---

## 3. Research Question It Helps Answer

### Primary Research Question

**"Do institutional signals (posted rules) enable agents to learn normative behavior more effectively than agents without such signals?"**

**Operationalization**:
- **Independent Variable**: Treatment condition (PERMITTED_COLOR visible vs hidden)
- **Dependent Variables**:
  - Learning speed (R_eval over time)
  - Final policy quality (final R_eval)
  - Violation rate trajectory
- **Control Variables**:
  - Same environment dynamics
  - Same resident sanctioning policy
  - Same ego agent architecture (only observation differs)

### Secondary Research Questions

1. **Does sanctioning enforcement matter?**
   - Compare with baseline (no residents)
   - Measure impact of sanction coverage on learning

2. **What is the role of grace period?**
   - 25-frame startup grace allows exploration
   - Test: does removing grace harm learning?

3. **How do immunity periods affect learning?**
   - 200-frame immunity prevents repeated punishment
   - Test: does immunity duration matter?

4. **Can agents learn from institutional signals even with imperfect enforcement?**
   - Current target: ~95-100% coverage
   - Test: degrade coverage to 50%, 75% - does treatment still help?

### Theoretical Contribution

This addresses a gap in multi-agent RL:

**Existing work**:
- Agents learn from rewards/penalties (trial and error)
- Agents learn from observing others (social learning)

**This work adds**:
- Agents learn from **institutional signals** (posted rules)
- Tests whether symbolic communication (PERMITTED_COLOR) accelerates normative learning

**Broader Impact**:
- AI alignment: Can posted rules guide agent behavior?
- Multi-agent coordination: Do institutions solve social dilemmas faster?
- Human-AI teams: Should we give AI systems access to explicit rules?

---

## 4. Lines of Code to Review

### Critical Lua Components (components.lua)

**PermittedColorHolder** (lines 916-946):
- Line 929: `self.permittedColorIndex = self._config.permittedColorIndex` - where rule is stored
- Line 935: `self._frameCount = self._frameCount + 1` - grace period tracking

**SameStepSanctionTracker** (lines 953-978):
- Line 969: `self._sanctionedThisStep = {}` - cleared each frame to prevent dogpiling
- Line 977: `self._sanctionedThisStep[avatarIndex] = true` - mark sanctioned

**ImmunityTracker** (lines 987-1046):
- Line 1014: `self._immuneSetAt = permittedColorHolder:getFrameCount()` - timestamp immunity
- Lines 1029-1030: `if currentFrame >= self._immuneSetAt + self._config.immunityCooldown` - timeout check

**SimpleZapSanction** (lines 1060-1207) **[MOST IMPORTANT]**:
- Lines 1089-1175: `onHit()` method - entire sanction logic
- Lines 1115-1119: Immunity check - fizzle if immune
- Lines 1123-1127: Tie-break check - fizzle if already sanctioned this step
- Lines 1131-1133: Violation classification
- Line 1136: `targetAvatar:getComponent('Avatar'):addReward(-10)` - apply penalty
- Lines 1139-1141: Set immunity
- Lines 1147-1157: Apply +α for correct zap
- Lines 1159-1168: Apply -β for mis-zap
- Lines 1177-1186: `_isViolation()` - grace period logic (line 1181)

**ResidentObserver** (lines 1370-1582) **[SECOND MOST IMPORTANT]**:
- Lines 1391-1582: `update()` method - emits all resident observations
- Lines 1413-1420: Get zappable indices from Zapper component
- Lines 1426-1486: Build nearby agents list with immunity info
- Lines 1506-1531: Build nearby berries lists
- Lines 1538-1548: Emit `resident_info` event
- Lines 1551-1561: Emit `nearby_agent` events
- Lines 1564-1581: Emit berry events

**InstitutionalObserver** (lines 1259-1292):
- Lines 1272-1291: `addObservations()` - produces PERMITTED_COLOR one-hot

### Critical Python Config (allelopathic_harvest.py)

**Avatar Creation** (lines 484-665):
- Lines 640-663: Conditional normative component addition
- Lines 654-663: ResidentObserver only for residents (not ego)

**Marking Overlay** (lines 880-1011):
- Lines 887-943: SimpleZapSanction path (no visual states)
- Line 927: `startup_grey_grace` kwarg (grace period)
- Lines 928-931: Reward shaping kwargs (alpha, beta)

**Scene Creation** (lines 721-877):
- Lines 766-777: PermittedColorHolder, SameStepSanctionTracker addition

**Configuration Defaults** (lines 1140-1210):
- Line 1180: `config.normative_gate = False` - master switch
- Line 1183: `config.enable_treatment_condition = False` - treatment vs control
- Line 1186: `config.permitted_color_index = 1` - RED default
- Line 1189: `config.startup_grey_grace = 25` - grace period
- Line 1192: `config.immunity_cooldown = 200` - immunity duration
- Lines 1195-1200: Reward shaping parameters

### Python Wrappers (to review for integration)

**agents/envs/resident_wrapper.py**:
- Line 128: `observations = self._last_timestep.observation` **[BUG: STALE OBSERVATIONS]**
- Should build observations from dmlab2d directly, not from last timestep

**agents/residents/scripted_residents.py**:
- Lines 180-185: Core resident decision logic (simple nearest-violator targeting)

**agents/metrics/schema.py**:
- Defines per-step and per-episode metric schemas

**agents/metrics/eval_harness.py**:
- Runs evaluation episodes and aggregates metrics

### Test Files to Review

**agents/tests/test_phase0_environment.py**:
- R0: Environment builds and loads correctly

**agents/tests/test_phase1_acceptance.py**:
- R1: Institutional rule changes across episodes

**agents/tests/test_phase2_residents.py**:
- R2: Resident coverage ~95-100% **[CURRENTLY FAILING: 45% coverage due to observation staleness]**

---

## 5. Interactive Play Guide

### Human Play Script

**File**: `/data/altar-transfer-simple/meltingpot/meltingpot/human_players/play_allelopathic_harvest.py`

**How to Run**:

```bash
cd /data/altar-transfer-simple/meltingpot
python -m meltingpot.human_players.play_allelopathic_harvest \
    --level_name=allelopathic_harvest__open \
    --observation=RGB \
    --print_events=True
```

**Controls**:
- `W/A/S/D`: Move forward/left/backward/right
- `Q/E`: Turn left/right
- `SPACE`: Fire zap beam (sanction)
- `1/2/3`: Fire colored beams (replant RED/GREEN/BLUE)
- `TAB`: Switch between players
- `ESC`: Quit

**What to Test**:

1. **Institutional Rule**:
   - Look at your PERMITTED_COLOR observation (if treatment)
   - Consume berries matching that color
   - Check if you receive sanctions when violating

2. **Sanction Mechanics**:
   - Zap a violator (wrong color agent)
   - Check if -10 is applied
   - Try to zap again immediately - should fail (immunity)
   - Wait 200 frames (~10 seconds at 20 FPS) - should succeed

3. **Grace Period**:
   - Start episode as GREY (no body color)
   - Consume berries while GREY
   - Check if you're sanctioned before 25 frames (should NOT be)
   - Check if you're sanctioned after 25 frames (should be)

4. **Dogpiling**:
   - Have multiple residents zap same target in same frame
   - Only one should apply -10
   - All should pay -c cost

5. **Reward Decomposition**:
   - Track your rewards over time
   - Verify R_total = berries + sanctions + alpha - beta - c

**Print Events Mode**:

Add `--print_events=True` to see Lua events in console:
- `sanction`: When zap hits
- `reward_component`: When alpha/beta/c applied
- `resident_info`: Resident state each frame
- `nearby_agent`: Nearby agent detection

**Video Recording**:

The play script uses pygame rendering. To record video:
1. Use screen recording software (OBS, SimpleScreenRecorder)
2. Or modify the script to save frames to video file
3. Standard approach: `pygame.image.save(surface, f"frame_{i}.png")`

### Testing Normative Features

**Test Scenario 1: Treatment vs Control**

```python
# In allelopathic_harvest.py, set:
config.normative_gate = True
config.enable_treatment_condition = True  # Treatment

# Run episode, observe:
# - PERMITTED_COLOR in observations?
# - Can you see the permitted color?
# - Do you learn faster?

# Then set:
config.enable_treatment_condition = False  # Control

# Run episode, observe:
# - PERMITTED_COLOR missing?
# - Must infer from sanctions?
```

**Test Scenario 2: Resident Coverage**

```python
# Run episode with residents
# Count opportunities vs sanctions
# Expect ~95-100% coverage
# Current bug: only ~45% due to observation staleness
```

**Test Scenario 3: Immunity**

```python
# Zap agent, note frame number
# Try to zap again within 200 frames - should fizzle
# Check sanction event: immune=1
# Wait 200+ frames, zap again - should land
```

**Test Scenario 4: Altar Visual**

```python
# In allelopathic_harvest.py, set:
config.normative_gate = True
config.enable_treatment_condition = True
config.altar_coords = (5, 5)  # Fixed position

# Run episode, observe:
# - Altar appears at (5, 5)?
# - Color matches permitted color?
```

---

## 6. Known Issues

### Issue 1: Observation Staleness (R2 Coverage Failure)

**Problem**: resident_wrapper.py:128 uses `self._last_timestep.observation`, which is from previous frame. ResidentObserver emits events AFTER avatar movement, so positions are stale by 1 frame.

**Impact**: Resident controller sees where agents WERE, not where they ARE. Sanctions miss targets. Coverage drops to ~45%.

**Fix**: Build observations from dmlab2d directly in current frame, not from last timestep.

### Issue 2: Components in Wrong Location

**Problem**: All normative components (SimpleZapSanction, ImmunityTracker, etc.) are in components.lua. Base meltingpot puts avatar behaviors in avatar_library.lua.

**Impact**: Lifecycle mismatch, event emission complexity, harder to access avatar state.

**Fix**: Move SimpleZapSanction to avatar_library.lua as avatar overlay component (like GraduatedSanctionsMarking). Use postStart(), onHit(), avatarStateChange() lifecycle.

### Issue 3: Event Parsing Complexity

**Problem**: ResidentObserver emits events that must be parsed in Python. Not needed if using avatar_library pattern.

**Impact**: Unnecessary complexity, potential for bugs.

**Fix**: After moving components to avatar_library.lua, build observations directly from component state.

---

## 7. Next Steps

### Immediate (Fix Critical Bugs)

1. Fix observation staleness in resident_wrapper.py
2. Test R2 coverage again - should reach ~95-100%

### Short-term (Architectural Cleanup)

1. Move SimpleZapSanction to avatar_library.lua
2. Simplify ResidentObserver or remove it
3. Build resident observations from component state directly

### Medium-term (Testing & Validation)

1. Implement all R0-R6 tests
2. Run treatment vs control experiments
3. Measure learning curves, violation rates

### Long-term (Research)

1. Train ego agents in treatment vs control
2. Analyze R_eval trajectories
3. Test hypothesis: treatment > control?
4. Write up results for publication

---

## 8. Summary

This environment extends Allelopathic Harvest with a normative institution system to test the Hadfield-Weingast hypothesis. Key additions:

- **Institutional rule**: PermittedColorHolder stores permitted color (1/2/3)
- **Sanction system**: SimpleZapSanction applies immediate -10 with immunity
- **Treatment manipulation**: InstitutionalObserver provides PERMITTED_COLOR observation
- **Reward decomposition**: Track alpha, beta, c separately from R_eval
- **Resident observations**: ResidentObserver emits privileged state for controllers

**Current state**: Environment is built and partially tested, but has critical bug (observation staleness) causing R2 coverage test to fail (45% instead of ~95-100%).

**Ready for review**: Lines specified in Section 4, particularly SimpleZapSanction:onHit() and ResidentObserver:update().

**Interactive testing**: Use play_allelopathic_harvest.py with --print_events flag to inspect behavior.
