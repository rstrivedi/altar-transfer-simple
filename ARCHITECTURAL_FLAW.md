# Critical Architectural Flaw: Wrong Component Location

## TL;DR

**All normative components should be in `avatar_library.lua`, NOT `components.lua`.**

This architectural mistake is the root cause of:
- Observation staleness (R2 coverage failure: 45% instead of ~95-100%)
- Event parsing complexity
- Lifecycle timing issues

---

## Evidence: Where Sanctioning Lives in Reference Implementations

### Base MeltingPot (google-deepmind/meltingpot)

**File**: `/home/ubuntu/altar/altar-transfer/meltingpot/lua/modules/avatar_library.lua`

```
Line 622:  local Zapper = class.Class(component.Component)
Line 1060: local GraduatedSanctionsMarking = class.Class(component.Component)
```

**Location**: `avatar_library.lua` (avatar overlay components, NOT substrate components)

### altar-transfer (Reference Implementation)

**File**: `/home/ubuntu/altar/altar-transfer/meltingpot/lua/modules/avatar_library.lua`

```
Line 1236: local NormativeSanctionsMarking = class.Class(component.Component)
```

**Location**: `avatar_library.lua` (avatar overlay component)

**Key Lifecycle Methods**:

```lua
function NormativeSanctionsMarking:postStart()
  -- Line 1324: GET DIRECT AVATAR REFERENCE
  self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  -- Direct access to avatar, no event emission needed
end

function NormativeSanctionsMarking:onHit(hittingGameObject, hitName)
  -- Line 1366: IMMEDIATE HIT CALLBACK
  -- Called directly when zap hits, no latency
  -- Line 1374: Direct component access
  local zappedColorId = thisAvatar.gameObject:getComponent('ColorZapper').colorId
  -- Line 1382: Direct scene component access
  local meta_altar_color = scene:getComponent("MetaAltar"):getColor()
  -- Line 1406-1407: Apply rewards immediately
  hittingAvatar:addReward(sourceReward)
  thisAvatar:addReward(logic.targetReward)
end

function NormativeSanctionsMarking:avatarStateChange(behavior)
  -- Line 1340: RESPAWN/DEATH CALLBACK
  if behavior == 'respawn' then
    -- Handle respawn, reset immunity, etc.
  elseif behavior == 'die' then
    -- Handle death
  end
end
```

### Our Project (WRONG)

**File**: `/data/altar-transfer-simple/meltingpot/meltingpot/lua/levels/allelopathic_harvest/components.lua`

```
Line 987:  local ImmunityTracker = class.Class(component.Component)
Line 1060: local SimpleZapSanction = class.Class(component.Component)
Line 1370: local ResidentObserver = class.Class(component.Component)
```

**Location**: `components.lua` (substrate-level components, WRONG for avatar behaviors)

**Missing Lifecycle Methods**:
- NO `postStart()` - cannot get direct avatar reference
- HAS `onHit()` but it's awkward (lines 1089-1175)
- NO `avatarStateChange()` - cannot handle respawn/death cleanly

---

## Why This Matters: Component Lifecycle

### avatar_library.lua Components (Avatar Overlays)

**Purpose**: Components that attach to individual avatars and follow their lifecycle

**Lifecycle Hooks**:
1. `__init__()` - construction with kwargs
2. `reset()` - called at episode start
3. **`postStart()`** - called AFTER avatar is created and positioned
   - **Critical**: Can get avatar object reference via `sim:getAvatarFromIndex()`
   - Stores `self._avatarObject` for direct access
4. **`onHit(hittingGameObject, hitName)`** - called IMMEDIATELY when hit by beam
   - **Critical**: Synchronous callback, no latency
   - Direct access to hitter and target avatars
5. **`avatarStateChange(behavior)`** - called when avatar respawns/dies
   - **Critical**: Clean lifecycle management
6. `update()` - called each frame
7. `addObservations()` - called to build observations

**Key Advantage**: Direct avatar access via `self._avatarObject`

### components.lua Components (Substrate/Scene Components)

**Purpose**: Components that manage substrate-level state (berries, global counters, scene rules)

**Lifecycle Hooks**:
1. `__init__()` - construction
2. `reset()` - called at episode start
3. `update()` - called each frame
4. NO `postStart()` - cannot get avatar references easily
5. NO `onHit()` callback - must use other mechanisms
6. NO `avatarStateChange()` - cannot handle respawn/death

**Key Limitation**: No direct avatar access, must use `simulation:getAvatarFromIndex()` repeatedly

---

## The Impact of Wrong Location

### Problem 1: Cannot Use onHit() Properly

**In avatar_library.lua** (correct):
```lua
function SimpleZapSanctioning:onHit(hittingGameObject, hitName)
  -- Called IMMEDIATELY when zap hits
  -- Direct access to both avatars
  local targetColor = self._avatarObject:getComponent('ColorZapper').colorId
  local zapperAvatar = hittingGameObject:getComponent('Avatar')
  -- Check, apply, done - all synchronous
  zapperAvatar:addReward(10)
  self._avatarObject:getComponent('Avatar'):addReward(-10)
  return true  -- Block beam
end
```

**In components.lua** (our broken version):
```lua
function SimpleZapSanction:onHit(hitterObject, hitName)
  -- Lines 1089-1175: AWKWARD ACCESS
  -- Must fetch avatar via simulation API every time
  local targetAvatar = self.gameObject.simulation:getAvatarFromIndex(targetIndex)
  -- No direct self._avatarObject reference
  -- More API calls, more latency
end
```

### Problem 2: No postStart() = No Avatar Reference

**In avatar_library.lua** (correct):
```lua
function SimpleZapSanctioning:postStart()
  -- Get avatar reference ONCE
  self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)
  -- Now have direct access forever
end

function SimpleZapSanctioning:someMethod()
  -- Fast: use cached reference
  local color = self._avatarObject:getComponent('ColorZapper').colorId
end
```

**In components.lua** (our broken version):
```lua
-- NO postStart() available
-- Must fetch avatar every single time we need it
function SimpleZapSanction:someMethod()
  -- Slow: must query simulation API
  local avatar = self.gameObject.simulation:getAvatarFromIndex(self._config.playerIndex)
  local color = avatar:getComponent('ColorZapper').colorId
end
```

### Problem 3: Observation Staleness

**Root cause flow**:

1. ResidentObserver is in components.lua (line 1370)
2. It runs in `update()` method (line 1391)
3. By the time update() runs, avatars have already moved
4. It emits events with CURRENT positions (line 1538-1581)
5. Python wrapper uses `self._last_timestep.observation` (resident_wrapper.py:128)
6. But `_last_timestep` is from PREVIOUS frame
7. So resident controller sees positions from frame N-1
8. When it decides to FIRE_ZAP, target has moved
9. Sanction misses → 45% coverage instead of ~95-100%

**If it were in avatar_library.lua**:

1. Component has direct avatar access via `self._avatarObject`
2. Python wrapper can query dmlab2d's avatar state SYNCHRONOUSLY
3. No need for event emission and parsing
4. No frame lag
5. ~95-100% coverage

### Problem 4: No avatarStateChange()

**What we need**:
- Clear immunity on respawn
- Reset violation tracking on death
- Reconnect overlay to avatar

**In avatar_library.lua** (correct):
```lua
function SimpleZapSanctioning:avatarStateChange(behavior)
  if behavior == 'respawn' then
    self:clearImmunity()
    self.gameObject:teleport(self._avatarObject:getPosition())
  elseif behavior == 'die' then
    self.gameObject:setState(self._waitState)
  end
end
```

**In components.lua** (our broken version):
```lua
-- NO avatarStateChange() hook available
-- Must detect respawn manually in update()
-- Fragile, error-prone
```

---

## Comparison Table

| Feature | avatar_library.lua (Correct) | components.lua (Our Mistake) |
|---------|------------------------------|------------------------------|
| **Direct avatar reference** | ✅ via postStart() | ❌ Must query simulation API repeatedly |
| **onHit() callback** | ✅ Immediate, synchronous | ⚠️ Available but awkward |
| **avatarStateChange()** | ✅ Clean respawn/death handling | ❌ Not available |
| **Observation timing** | ✅ Synchronous with avatar state | ❌ Event emission lag |
| **Performance** | ✅ Fast (cached reference) | ❌ Slow (repeated queries) |
| **Lifecycle coupling** | ✅ Follows avatar lifecycle | ❌ Independent of avatar |
| **Code complexity** | ✅ Simple, direct | ❌ Complex, indirect |

---

## Why Did We Make This Mistake?

Looking at our code:

1. **We saw berries, coloring, taste in components.lua**
   - These ARE correct: they're substrate-level mechanics

2. **We assumed ALL components go in components.lua**
   - Wrong: avatar behaviors belong in avatar_library.lua

3. **We didn't study avatar_library.lua carefully enough**
   - If we had, we would have seen Zapper and GraduatedSanctionsMarking there

4. **We didn't check altar-transfer's architecture**
   - If we had, we would have seen NormativeSanctionsMarking in avatar_library.lua

---

## The Fix

### What Should Move to avatar_library.lua

**Must move** (avatar-level behaviors):
- SimpleZapSanction → SimpleZapSanctioning (follow naming convention)
- ImmunityTracker (attached to avatar overlay)
- ZapCostApplier (tracks avatar actions)
- InstitutionalObserver (produces avatar observations)
- NormativeRewardTracker (tracks avatar rewards)
- ResidentObserver (observes from avatar perspective)

**Should stay in components.lua** (scene/substrate-level):
- PermittedColorHolder (global state)
- SameStepSanctionTracker (cross-avatar coordination)

### What the Rewrite Looks Like

**File**: `meltingpot/meltingpot/lua/modules/avatar_library.lua`

```lua
local SimpleZapSanctioning = class.Class(component.Component)

function SimpleZapSanctioning:__init__(kwargs)
  -- Parse kwargs
end

function SimpleZapSanctioning:reset()
  -- Initialize state
end

function SimpleZapSanctioning:postStart()
  -- GET AVATAR REFERENCE
  local sim = self.gameObject.simulation
  self._avatarObject = sim:getAvatarFromIndex(self._config.playerIndex)
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  avatarComponent:connect(self.gameObject)
end

function SimpleZapSanctioning:onHit(hittingGameObject, hitName)
  if hitName ~= 'zapHit' then return false end

  -- Direct avatar access (fast!)
  local targetColor = self._avatarObject:getComponent('ColorZapper').colorId

  -- Get scene components
  local scene = self.gameObject.simulation:getSceneObject()
  local permittedColor = scene:getComponent('PermittedColorHolder'):getPermittedColorIndex()

  -- Check immunity (from overlay)
  local immunityObjects = self._avatarObject:getComponent('Avatar'):getAllConnectedObjectsWithNamedComponent('ImmunityTracker')
  if #immunityObjects > 0 and immunityObjects[1]:getComponent('ImmunityTracker'):isImmune() then
    return true  -- Fizzle
  end

  -- Check tie-break
  local sameStepTracker = scene:getComponent('SameStepSanctionTracker')
  if sameStepTracker:wasSanctionedThisStep(self._config.playerIndex) then
    return true  -- Fizzle
  end

  -- Apply -10 immediately
  self._avatarObject:getComponent('Avatar'):addReward(-10)

  -- Set immunity
  if #immunityObjects > 0 then
    immunityObjects[1]:getComponent('ImmunityTracker'):setImmune()
  end

  -- Mark sanctioned
  sameStepTracker:markSanctioned(self._config.playerIndex)

  -- Apply α/β to zapper
  local isViolation = (targetColor ~= permittedColor)
  if isViolation then
    hittingGameObject:getComponent('Avatar'):addReward(self._config.alphaValue)
  else
    hittingGameObject:getComponent('Avatar'):addReward(-self._config.betaValue)
  end

  return true
end

function SimpleZapSanctioning:avatarStateChange(behavior)
  if behavior == 'respawn' then
    -- Clear immunity on respawn
    local immunityObjects = self._avatarObject:getComponent('Avatar'):getAllConnectedObjectsWithNamedComponent('ImmunityTracker')
    if #immunityObjects > 0 then
      immunityObjects[1]:getComponent('ImmunityTracker'):clearImmunity()
    end
    -- Reconnect overlay
    self._avatarObject:getComponent('Avatar'):connect(self.gameObject)
  elseif behavior == 'die' then
    self.gameObject:setState(self._config.waitState)
  end
end

function SimpleZapSanctioning:addObservations(tileSet, world, observations)
  -- Add PERMITTED_COLOR observation directly
  -- No event emission needed
end
```

---

## Implications for ResidentWrapper

**Current (broken)**:
```python
# resident_wrapper.py:128
observations = self._last_timestep.observation  # STALE!
```

**After fix**:
```python
# Build observations from dmlab2d directly
observations = self._build_resident_observations()

def _build_resident_observations(self):
  # Query avatar components directly via dmlab2d
  # No event parsing needed
  # Synchronous with current frame
  return {
    'permitted_color': <from scene component>,
    'nearby_agents': <from avatar positions>,
    'immunity_status': <from ImmunityTracker component>,
  }
```

---

## Timeline for Fix

### Phase 1: Move Components to avatar_library.lua
1. Create SimpleZapSanctioning in avatar_library.lua
2. Implement postStart(), onHit(), avatarStateChange()
3. Update Python config to reference avatar_library.SimpleZapSanctioning

### Phase 2: Simplify Resident Observations
1. Remove ResidentObserver (no longer needed)
2. Build observations directly from component state
3. Fix resident_wrapper.py to use current frame data

### Phase 3: Test
1. Run R2 coverage test
2. Expect ~95-100% coverage (not 45%)
3. Verify no observation staleness

---

## Lessons Learned

1. **Study reference implementations carefully**
   - Don't just look at WHAT they do
   - Understand WHERE they put each component and WHY

2. **Respect framework patterns**
   - avatar_library.lua exists for a reason
   - Use it for avatar-level behaviors

3. **Check component lifecycle requirements**
   - Need postStart()? → avatar_library.lua
   - Need onHit()? → avatar_library.lua
   - Need avatarStateChange()? → avatar_library.lua
   - Just global state? → components.lua

4. **Event emission is a code smell**
   - If you're emitting events to communicate avatar state, probably wrong architecture
   - Direct component access is cleaner and faster

---

## Conclusion

**You were absolutely right to question this.**

Zapper, GraduatedSanctionsMarking (base meltingpot), and NormativeSanctionsMarking (altar-transfer) are ALL in avatar_library.lua for a reason:
- They need direct avatar access
- They need onHit() callbacks
- They need avatarStateChange() lifecycle hooks

Our mistake of putting everything in components.lua is the root cause of:
- Observation staleness (R2 coverage 45%)
- Event parsing complexity
- Lifecycle management issues

**The fix is clear: move avatar behaviors to avatar_library.lua where they belong.**
