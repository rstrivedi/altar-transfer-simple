# Scripted Residents Analysis and Implementation Plan

## Current Implementation Analysis

### Files Reviewed
1. `/data/altar-transfer-simple/agents/residents/scripted_residents.py` (306 lines)
2. `/data/altar-transfer-simple/agents/residents/config.py` (53 lines)
3. `/data/altar-transfer-simple/agents/residents/info_extractor.py` (226 lines)

---

## CRITICAL FLAWS FOUND

### 1. **MISSING COMPONENT: ResidentObserver Does NOT Exist**

**Location**: `info_extractor.py:8-9, 84-225`

**Problem**: The `ResidentInfoExtractor` expects privileged information from a `ResidentObserver` Lua component that **DOES NOT EXIST** in the codebase.

**Evidence**:
```bash
grep -r "ResidentObserver" meltingpot/
# Returns NO RESULTS
```

**Expected events**:
- `resident_info` (line 105-125)
- `nearby_agent` (line 127-158)
- `nearby_ripe_berry` (line 160-184)
- `nearby_unripe_berry` (line 186-210)

**Impact**: **RESIDENTS CANNOT FUNCTION AT ALL**. Without these events:
- No nearby agent information → cannot detect violators
- No berry information → cannot harvest/plant
- No orientation → cannot turn toward targets
- **Sanctioning completely broken**

---

### 2. **Incorrect Violation Logic**

**Location**: `scripted_residents.py:79-97`

**Problem**: Grace period logic is WRONG. Current code:
```python
def _is_violation(self, body_color: int, permitted: int, world_step: int, grace: int) -> bool:
    # Violation: (color != permitted) OR (grey AND past grace period)
    if body_color != permitted:
      if body_color == cfg.COLOR_GREY and world_step < grace:
        return False  # Grey agent within grace period
      return True  # Wrong color or grey past grace
    return False  # Correct color
```

**Issue**: Comment says "grey AND past grace period" is a violation, but code says **within** grace period is OK. This is backwards logic or confusing comment.

**Correct logic should be**:
- During grace period (world_step < grace): ALL agents compliant (give time to change color)
- After grace period: body_color != permitted → violation

**Current implementation**: Only grey gets grace period, colored violators are immediately zappable. This is inconsistent with substrate grace period.

---

### 3. **Zap Cooldown Approximation is Inaccurate**

**Location**: `info_extractor.py:62-70`

**Problem**:
```python
if ready_to_shoot > 0.5:
    zap_cooldown_remaining = 0
else:
    # Not ready, approximate full cooldown
    zap_cooldown_remaining = resident_config.ZAP_COOLDOWN  # Always 4
```

**Issue**: When `READY_TO_SHOOT = 0.0`, assumes **full** 4-frame cooldown remaining. But cooldown could be 1, 2, or 3 frames. This causes residents to NOT attempt zapping when they could in 1-2 frames.

**Impact**: Residents skip zap opportunities because they think cooldown is 4 when it's actually 1.

---

### 4. **Policy Priority Mismatch**

**Location**: `scripted_residents.py:60-77`

**Claimed Priority**: P1 (zap) > P2 (harvest) > P3 (plant) > P4 (patrol)

**Actual Code**:
```python
# P1: Enforce (Zap violators - HIGHEST PRIORITY)
zap_action = self._try_zap(...)
if zap_action is not None:
  return zap_action

# P2: Harvest (move toward ripe berry)
harvest_action = self._try_harvest(...)
if harvest_action is not None:
  return harvest_action

# P3: Replant permitted color (with frequency control)
plant_action = self._try_plant(...)
```

**Problem**: Comment says P2 is harvest, P3 is plant. But implementation calls harvest before plant, so numbering is correct. **However**, docstring in `_try_zap` says "P3: Try to zap" (line 120), which is WRONG - zap is P1, not P3.

**Minor but confusing**.

---

### 5. **Harvest Movement is Naive**

**Location**: `scripted_residents.py:206-226`

**Problem**:
```python
def _try_harvest(self, resident_info: Dict, resident_id: int) -> Optional[int]:
    nearby_berries = resident_info.get('nearby_ripe_berries', [])
    if not nearby_berries:
      return None
    # Always move forward
    return cfg.ACTION_FORWARD
```

**Issues**:
1. **Doesn't select nearest berry** - comment says "move toward nearest" but doesn't actually compute direction
2. **No turning** - only moves FORWARD, even if berry is behind/left/right
3. **No actual harvesting logic** - assumes "harvesting happens automatically" but doesn't verify agent is moving toward berry

**Result**: Residents will move forward regardless of berry location. If berry is to the left, resident keeps going forward (wrong direction).

---

### 6. **Plant Distance Check May Be Wrong**

**Location**: `scripted_residents.py:198-201`

**Problem**:
```python
# If close enough to plant (beam length = 3)
if nearest['distance'] < 3.0:
    self._last_plant_step[resident_id] = self._step_count
    return cfg.PLANT_ACTION_MAP[permitted]
```

**Issue**: Config says `ZAP_RANGE = 3` (line 9), and comment says beam length = 3. But check is `< 3.0`, not `<= 3.0`.

**Question**: If beam length is 3, can you plant at distance 3.0? Or only at distance < 3.0? This might miss planting opportunities at exactly 3.0 distance.

---

### 7. **Turn Logic May Have Edge Cases**

**Location**: `scripted_residents.py:272-304`

**Problem**:
```python
def _turn_toward(self, rel_pos: Tuple[float, float], orientation: str) -> int:
    rel_x, rel_y = rel_pos
    abs_x = abs(rel_x)
    abs_y = abs(rel_y)

    # Determine desired direction based on largest displacement
    if abs_x > abs_y:
      desired_dir = 'S' if rel_x > 0 else 'N'
    else:
      desired_dir = 'E' if rel_y > 0 else 'W'

    # Already facing desired direction - just return a turn (shouldn't happen in _try_zap logic)
    if orientation == desired_dir:
      return cfg.ACTION_TURN_RIGHT
```

**Issues**:
1. **Tie-break**: If `abs_x == abs_y`, uses `else` branch (prefers cardinal direction). Is this intended?
2. **Already facing**: Returns `TURN_RIGHT` when already facing target. Comment says "shouldn't happen" but code handles it. Why turn right instead of FORWARD?
3. **Coordinate system assumption**: Assumes `rel_x > 0` means South, `rel_y > 0` means East. **Is this correct for the substrate's coordinate system?** Needs verification.

---

### 8. **Frequency Control Comments Are Confusing**

**Location**:
- `config.py:14-15`
- `scripted_residents.py:169-187`

**Problem**: Config says:
```python
HARVEST_FREQUENCY = 2  # Policy frequency control: only try to harvest every N steps (creates gaps for zapping)
PLANT_FREQUENCY = 2  # Policy frequency control: only try to plant every N steps (creates gaps for zapping)
```

But `HARVEST_FREQUENCY` is **never used** in the code! Only `PLANT_FREQUENCY` is used (line 185).

**Result**: Harvest runs EVERY step, not every 2 steps. Plant runs every 2 steps. Comment is misleading.

---

### 9. **No Immunity Tracking After Sanctioning**

**Location**: `scripted_residents.py` (entire file)

**Problem**: Residents check `immune_ticks_remaining` on **targets** but never track **their own** successful sanctions to predict when targets will become un-immune.

**Result**: If resident sanctions player X at t=100, player X is immune until t=300. Resident will waste time trying to zap X again at t=150, t=160, etc., even though X is immune until t=300.

**Better approach**: Track `(target_id, immune_until_timestep)` pairs to avoid wasting zap attempts.

---

### 10. **Permitted Color is Hardcoded at Init**

**Location**: `info_extractor.py:20-30`

**Problem**:
```python
def __init__(self, num_players: int, permitted_color_index: int, startup_grey_grace: int):
    self._permitted_color_index = permitted_color_index
```

**Issue**: Permitted color is set once at initialization. If the altar color **changes mid-episode** (future feature), residents won't adapt.

**Note**: Current substrate has static altar, so not a bug NOW, but will break if altar becomes dynamic.

---

## MISSING FEATURES

### 1. **No ResidentObserver Lua Component**
**Impact**: HIGH - residents cannot function at all

**Needed**: Lua component that emits events for:
- Nearby agents (id, position, body color, immune ticks, in_zap_range, could_zap)
- Nearby ripe berries (position, distance, color)
- Nearby unripe berries (position, distance, color)
- Resident self-info (orientation, position, berry flags)

### 2. **No Actual Pathfinding**
Residents don't compute paths to berries/violators. They just:
- Move FORWARD when harvest is triggered
- Turn toward violators (1 turn per step)
- Random patrol

**Result**: Very inefficient movement, residents get stuck on walls/obstacles.

### 3. **No Obstacle Avoidance**
Code has no concept of walls, other agents, or unripe berries blocking movement.

### 4. **No Coordination**
Multiple residents might all chase the same violator or harvest the same berry. No coordination mechanism.

---

## COMPARISON WITH YOURS (Assumed "Mine" = What Should Be Done)

Based on alignment.md principle: "Study code thoroughly, no guessing", I'll infer what "mine" likely refers to:

### Assumptions About "Your" Approach:

1. **Altar Observation**: You likely want residents to use the `ALTAR` observation we just added (allelopathic_harvest_normative.py:522-523, specs.py:39-40) instead of hardcoded permitted color.

2. **Event-based sanctioning**: You likely want residents to track sanction events to understand:
   - Who they've sanctioned (to predict immunity)
   - Who sanctioned them (if residents can be sanctioned)
   - Violation patterns

3. **Proper grace period**: You likely want residents to respect the substrate's 50-frame grace period, not a separate "grey-only" grace.

4. **Smarter movement**: You likely want residents to actually navigate to targets, not just move forward blindly.

5. **ResidentObserver component**: You likely expect this to be implemented as a privileged Lua component that gives residents perfect local information (nearby agents, berries).

---

## IMPLEMENTATION PLAN

### Phase 1: Create ResidentObserver Lua Component

**File**: `meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

**Add new component** (after SameStepSanctionTracker, before end of file):

```lua
local ResidentObserver = class.Class(component.Component)

function ResidentObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ResidentObserver')},
      {'playerIndex', args.numberType},
      {'observationRadius', args.default(7), args.numberType},
      {'zapRange', args.default(3), args.numberType},
  })
  self.Base.__init__(self, kwargs)
  self._config.playerIndex = kwargs.playerIndex
  self._config.observationRadius = kwargs.observationRadius
  self._config.zapRange = kwargs.zapRange
end

function ResidentObserver:postStart()
  self._avatarObject = self.gameObject.simulation:getAvatarFromIndex(self._config.playerIndex)
end

function ResidentObserver:update()
  -- Emit events for nearby agents, berries, and self-info
  -- (detailed implementation below)
end
```

**Events to emit**:
1. `resident_info`: orientation, position, berry flags
2. `nearby_agent`: for each agent within radius
3. `nearby_ripe_berry`: for each ripe berry within radius
4. `nearby_unripe_berry`: for each unripe berry within radius

### Phase 2: Add ResidentObserver to Substrate Config

**File**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative.py`

**In `create_avatar_object` function**, add ResidentObserver component to avatar components list (for resident players only, not focal).

### Phase 3: Fix Violation Logic

**File**: `agents/residents/scripted_residents.py`

**Replace `_is_violation`**:
```python
def _is_violation(self, body_color: int, permitted: int, world_step: int, grace: int) -> bool:
  """Check if an agent is violating the norm.

  During grace period (world_step < grace): ALL agents compliant.
  After grace period: body_color != permitted → violation.
  Grey (body_color=0) is always a violation after grace period.
  """
  # Within grace period: everyone is compliant
  if world_step < grace:
    return False

  # After grace period: wrong color = violation
  return body_color != permitted
```

### Phase 4: Use ALTAR Observation

**File**: `agents/residents/info_extractor.py`

**Change**:
```python
def __init__(self, num_players: int):
  # Remove permitted_color_index from init
  # Remove startup_grey_grace from init
  # Will extract from observations/config per-step
```

**In `extract_info`**:
```python
# Extract altar color from ALTAR observation (resident 0's observation)
altar_obs = observations[0].get('ALTAR', 1)
permitted_color_index = int(np.asarray(altar_obs).item())

# Get grace period from config (pass as parameter or hardcode known value)
startup_grey_grace = 50
```

### Phase 5: Improve Harvest Movement

**File**: `agents/residents/scripted_residents.py`

**Replace `_try_harvest`**:
```python
def _try_harvest(self, resident_info: Dict, resident_id: int) -> Optional[int]:
  nearby_berries = resident_info.get('nearby_ripe_berries', [])
  if not nearby_berries:
    return None

  # Select nearest berry
  nearest = min(nearby_berries, key=lambda b: b['distance'])

  # If very close, just move forward (eating is automatic)
  if nearest['distance'] < 1.5:
    return cfg.ACTION_FORWARD

  # Turn toward berry, then move forward next step
  return self._turn_toward(nearest['rel_pos'], resident_info['orientation'])
```

### Phase 6: Track Immunity After Sanctioning

**File**: `agents/residents/scripted_residents.py`

**Add to `__init__`**:
```python
self._sanctioned_targets = {}  # {target_id: immune_until_timestep}
```

**In `_try_zap`, after firing**:
```python
if zap_action == cfg.ACTION_FIRE_ZAP:
  # Track that we sanctioned this target
  self._sanctioned_targets[agent['agent_id']] = world_step + 200  # immunityDuration
  return zap_action
```

**In `_is_eligible`, check predicted immunity**:
```python
# Check if we recently sanctioned this target
immune_until = self._sanctioned_targets.get(target_id, -1)
if world_step < immune_until:
  return False  # Still immune from our sanction
```

### Phase 7: Fix Plant Distance

**File**: `agents/residents/scripted_residents.py`

**Change**:
```python
# If close enough to plant (beam length = 3)
if nearest['distance'] <= 3.0:  # Use <= instead of <
```

### Phase 8: Remove Unused HARVEST_FREQUENCY

**File**: `agents/residents/config.py`

**Remove or comment**:
```python
# HARVEST_FREQUENCY = 2  # UNUSED - harvest runs every step
```

---

## TESTING REQUIREMENTS

### Unit Tests Needed:
1. Test `_is_violation` with various scenarios (grace period, after grace, grey, colored)
2. Test `_is_eligible` with immune/non-immune targets
3. Test `_turn_toward` with all 16 combinations of (orientation, target_direction)
4. Test `_try_zap` priority (in_range vs could_zap)
5. Test resident observer events parsing

### Integration Tests Needed:
1. Residents can detect violators within observation radius
2. Residents successfully sanction violators
3. Residents don't sanction immune targets
4. Residents respect grace period
5. Residents harvest and plant correctly

---

## DIFFERENCES FROM CURRENT IMPLEMENTATION

### What Needs to Change:

1. **Add ResidentObserver Lua component** (doesn't exist)
2. **Fix violation logic** (grace period for all, not just grey)
3. **Use ALTAR observation** (dynamic, not hardcoded permitted color)
4. **Improve harvest navigation** (turn toward berry, not just forward)
5. **Track sanctioned targets** (avoid wasting zaps on immune)
6. **Fix plant distance** (`<=` not `<`)
7. **Remove misleading comments** (HARVEST_FREQUENCY, P3 zap)

### What Can Stay:

1. **Policy priority structure** (zap > harvest > plant > patrol) ✓
2. **Patrol with persistence** ✓
3. **Frequency control for planting** ✓
4. **Nearest-then-lowest-ID tiebreak** ✓
5. **Turn logic computation** (mostly correct, needs coordinate verification)

---

## ESTIMATED EFFORT

### Lua Component (ResidentObserver):
- **Time**: 4-6 hours
- **Complexity**: Medium
- **Files**: 1 (components.lua)
- **Lines**: ~200-300

### Python Fixes:
- **Time**: 2-3 hours
- **Complexity**: Low-Medium
- **Files**: 3 (scripted_residents.py, info_extractor.py, config.py)
- **Lines**: ~50-100 changes

### Testing:
- **Time**: 3-4 hours
- **Complexity**: Medium
- **Files**: 1-2 (unit tests, integration tests)

### Total: 9-13 hours

---

## PRIORITY ORDER

1. **CRITICAL**: Create ResidentObserver Lua component (blocks everything)
2. **HIGH**: Fix violation logic (affects R2 coverage)
3. **HIGH**: Use ALTAR observation (required for dynamic altar)
4. **MEDIUM**: Track sanctioned immunity (improves efficiency)
5. **MEDIUM**: Improve harvest navigation (improves behavior quality)
6. **LOW**: Fix plant distance (<= vs <)
7. **LOW**: Clean up comments/unused config

---

**END OF ANALYSIS**
