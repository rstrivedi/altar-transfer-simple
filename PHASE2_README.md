# Phase 2 Implementation - Scripted Resident Agents

## Overview

Phase 2 implements deterministic scripted resident agents that play at equilibrium in the normative allelopathic harvest environment. Residents enforce the posted rule, harvest/replant permitted color, and patrol - **identically in control and treatment conditions**. This isolates the Hadfield-Weingast mechanism: clear classification + congruent decentralized sanctions → competence & compliance.

## Research Design

**Objective**: Provide selective, decentralized punishment keyed to the institution's classification, while the **only** A/B difference remains ego's access to the rule observation.

**Key Invariant**: Residents behave identically in control and treatment. The experimental manipulation is ONLY whether ego sees PERMITTED_COLOR observation and altar.

## Critical Design Decisions

### 1. Resident Behavior (Equilibrium Play)
Residents play optimally at equilibrium:
- **Do not violate**: Residents always plant permitted color, never violate the norm
- **Enforce violations**: Sanction any agent (including ego) that violates
- **Achieve monoculture**: Focus on planting permitted color to reach high monoculture fraction (≥85%)
- **Harvest opportunistically**: Eat any ripe berry (not selective, but monoculture means most are permitted color)

### 2. Privileged Information Access
Residents need accurate state information to enforce effectively. We use **ResidentObserver** Lua component to provide:
- Positions of all agents
- Body colors (0=GREY, 1=RED, 2=GREEN, 3=BLUE)
- Immunity status (frames remaining)
- Nearby ripe berries
- Standing on unripe patch

**Critical separation**:
- **Baseline mode** (`ego_index=None`): All 16 agents are residents → all get ResidentObserver
- **Training mode** (`ego_index=0`): Agent 0 is ego (no ResidentObserver), agents 1-15 are residents (have ResidentObserver)
- Residents can see ego's privileged state (via ResidentObserver querying all avatars)
- Ego cannot see residents' privileged state (no ResidentObserver component)

### 3. Violation Classification (Match Phase 1)
- **Violating**: `(body_color ≠ permitted)` OR `(body_color == GREY AND world_step ≥ grace_period)`
- **Compliant**: `(body_color == permitted)` OR `(body_color == GREY AND world_step < grace_period)`
- **Grace period**: 25 frames (startup_grey_grace)

### 4. Eligibility for Sanctioning
- **Eligible**: `Violating(target) AND immune_ticks_remaining == 0`
- Residents must check immunity before attempting to zap
- If target is immune, do not zap (wait for immunity to clear)

### 5. Physics (From Phase 1)
- Simple sanctions: immediate -10, no freeze/removal
- Per-target immunity: 200 frames (clears on color change or timeout)
- Same-step tie-break: at most one -10 per target per frame
- Zap costs: -c (effort), +α (correct, training only), -β (mis-zap)
- Zap cooldown: 4 frames
- Zap range: 3 cells
- Plant cooldown: 2 frames

---

## Resident Policy (Deterministic Priority Order)

### P0. Definitions
```python
permitted = permitted_color_index  # 1=RED, 2=GREEN, 3=BLUE
grace = startup_grey_grace  # 25 frames

def is_violation(target):
    return (target.body_color != permitted) or \
           (target.body_color == GREY and world_step >= grace)

def is_eligible(target):
    return is_violation(target) and target.immune_ticks_remaining == 0
```

### P1. Enforce (Zap)
**Priority**: Highest
```
If any eligible target within zap range AND zap_cooldown_remaining == 0:
    - Select nearest eligible target (tie-break: lowest agent_id)
    - Action: FIRE_ZAP (index 7)
Else:
    - Continue to P2
```

**Note**: Do not zap immune targets even if they are violating. Wait for immunity to clear.

### P2. Replant Permitted
**Priority**: Second
```
If standing_on_unripe == True:
    - Action: PLANT permitted color (index 8/9/10 for RED/GREEN/BLUE)
Else:
    - Continue to P3
```

**Rationale**: Residents must always be ready to plant permitted color to avoid becoming grey violators. High planting priority ensures monoculture achievement.

### P3. Harvest (Any Ripe Berry)
**Priority**: Third
```
If has_ripe_berry_in_radius == True:
    - Move toward nearest ripe berry (greedy: turn toward, then FORWARD)
    - Action: movement action
Else:
    - Continue to P4
```

**Note**: Harvest ANY ripe berry, not just permitted color. R4 expects ≥95% harvests are permitted color naturally due to monoculture.

### P4. Patrol
**Priority**: Lowest (fallback)
```
- Pick direction from {FORWARD, TURN_LEFT, TURN_RIGHT}
- Persist for 8 frames before changing direction
- Action: patrol movement
```

**Rationale**: Ensure coverage so residents encounter violators. Avoid oscillation/stuck behavior.

---

## Implementation Status

### ✅ Completed Components

#### 1. **Resident Config** (`agents/residents/config.py`)
**Commit**: 23d9402

Tunables:
- `ZAP_RANGE = 3`, `ZAP_COOLDOWN = 4`
- `HARVEST_RADIUS = 3`, `PLANT_COOLDOWN = 2`
- `PATROL_PERSISTENCE = 8` frames
- `CHOICE_TIEBREAK = "nearest_then_lowest_id"`
- `DEFAULT_SEED = 42`
- Action/color constants and mappings

#### 2. **ResidentObserver Lua Component** (`meltingpot/lua/.../components.lua`)
**Commit**: 6dca512

**What it does**:
- Queries all avatars' positions via Transform component
- Queries body colors via ColorZapper.getColorId()
- Queries immunity via ImmunityTracker.getImmunityRemaining()
- Detects ripe berries at current position
- Detects unripe patches at current position
- Emits `resident_info` event (permitted_color, berry flags)
- Emits `nearby_agent` events (agent_id, rel_pos, body_color, immune_ticks)

**Attached to**:
- All agents when `ego_index = None` (baseline mode)
- Only residents (not ego) when `ego_index = 0` (training mode)

**Added methods**:
- `ImmunityTracker:getImmunityRemaining()` → frames of immunity left

#### 3. **ResidentInfoExtractor** (`agents/residents/info_extractor.py`)
**Commit**: 6dca512

**What it does**:
- Parses `resident_info` and `nearby_agent` events from ResidentObserver
- Extracts `zap_cooldown_remaining` from READY_TO_SHOOT observation
- Converts Lua 1-indexed to Python 0-indexed
- Builds structured info dict:
```python
{
  'world_step': int,
  'permitted_color_index': int,  # 1, 2, or 3
  'startup_grey_grace': int,  # 25
  'residents': {
    agent_id: {
      'pos': (x, y),
      'zap_cooldown_remaining': int,
      'nearby_agents': [
        {
          'agent_id': int,
          'rel_pos': (dx, dy),
          'body_color': int,  # 0=GREY, 1=RED, 2=GREEN, 3=BLUE
          'immune_ticks_remaining': int
        }
      ],
      'has_ripe_berry_in_radius': bool,
      'standing_on_unripe': bool
    }
  }
}
```

#### 4. **Config Changes** (`meltingpot/configs/.../allelopathic_harvest.py`)
**Commit**: 6dca512

Added:
- `config.ego_index = None` parameter
  - `None`: All-residents mode (all 16 get ResidentObserver)
  - `0`: Training mode (only 1-15 get ResidentObserver)

Modified `create_avatar_object()`:
- Conditionally attach ResidentObserver based on ego_index
- If ego_index=None → attach to all
- If ego_index=0 → skip agent 0

---

### 🚧 Remaining Components

#### 3. **ResidentController** (`agents/residents/scripted_residents.py`)
**Status**: Next to implement

**Interface**:
```python
class ResidentController:
    def reset(seed: int, cfg: dict) -> None
    def act(resident_id: int, info: dict) -> int  # Returns action index
```

**Implementation**:
- P1: Check eligible targets, zap nearest
- P2: If standing on unripe, plant permitted color
- P3: If ripe berry nearby, move toward it (greedy)
- P4: Patrol with persistence (RNG seeded)

**Action Selection**:
- Zap: action = 7 (FIRE_ZAP)
- Plant: action = 8/9/10 (FIRE_ONE/TWO/THREE based on permitted color)
- Move toward berry: Greedy turn + forward
- Patrol: Random from {FORWARD=1, TURN_LEFT=5, TURN_RIGHT=6}, persist 8 frames

#### 4. **ResidentWrapper** (`agents/envs/resident_wrapper.py`)
**Status**: Not started

**Purpose**: Wrapper that calls ResidentController for designated agents

**Interface**:
```python
class ResidentWrapper:
    def __init__(env, resident_indices, ego_index, resident_controller, extractor)
    def reset() -> timestep
    def step(ego_action) -> timestep  # Combines ego + resident actions
```

**Logic**:
- Extract info via ResidentInfoExtractor
- Get resident actions from ResidentController
- Combine with ego action
- Step environment

#### 5. **Acceptance Tests R1-R8** (`agents/tests/phase2_residents_tests.py`)
**Status**: Not started

**Tests**:
- **R1**: Selectivity - residents never zap compliant agents
- **R2**: Coverage - ≥80% of violators sanctioned within 10 frames
- **R3**: No dogpiling - no duplicate -10 on immune targets, residents don't attempt
- **R4**: Plant/harvest purity - 100% plant permitted, ≥95% harvest permitted
- **R5**: Arm invariance - identical decisions in control vs treatment
- **R6**: Same-step tie-break - only one -10 lands when multiple zaps
- **R7**: No hidden dependencies - grep for freeze/removal references
- **R8**: Monoculture achievement - ≥85% permitted berries at t=2000

#### 6. **PHASE2_README.md**
**Status**: This file (in progress)

---

## Configuration Reference

### Baseline Mode (All-Residents)
```python
config = allelopathic_harvest.get_config()
config.normative_gate = True
config.permitted_color_index = 1  # RED
config.ego_index = None  # All-residents mode

roles = ["default"] * 16
env = substrate.build("allelopathic_harvest", roles, config)

# All 16 agents get ResidentObserver
# Use for baseline experiments
```

### Training Mode (Ego + Residents)
```python
config = allelopathic_harvest.get_config()
config.normative_gate = True
config.permitted_color_index = 1  # RED
config.ego_index = 0  # Agent 0 is ego

# Treatment
config.enable_treatment_condition = True
config.altar_coords = (5, 15)

# Control
# config.enable_treatment_condition = False

roles = ["default"] * 16
env = substrate.build("allelopathic_harvest", roles, config)

# Agent 0: No ResidentObserver (ego)
# Agents 1-15: Have ResidentObserver (residents)
```

### Using Residents
```python
from agents.residents.info_extractor import ResidentInfoExtractor
from agents.residents.scripted_residents import ResidentController  # TODO
from agents.envs.resident_wrapper import ResidentWrapper  # TODO

# Setup
extractor = ResidentInfoExtractor(
    num_players=16,
    permitted_color_index=config.permitted_color_index,
    startup_grey_grace=config.startup_grey_grace)

controller = ResidentController()
controller.reset(seed=42, cfg={...})

# Wrap environment
env = ResidentWrapper(
    env=base_env,
    resident_indices=[1,2,3,...,15],
    ego_index=0,
    resident_controller=controller,
    info_extractor=extractor)

# Use like normal env
timestep = env.reset()
timestep = env.step(ego_action)  # Residents act automatically
```

---

## Testing Strategy

### Test Map Approach
For R1-R7, use small episodes (200-500 steps) with 3-4 agents:
```python
config.episode_timesteps = 200
roles = ["default"] * 3  # Minimal setup
```

For R8 (monoculture), full episode:
```python
config.episode_timesteps = 2000  # Full episode
roles = ["default"] * 16  # All residents
```

### Determinism
All tests use fixed seeds for reproducibility:
```python
controller.reset(seed=42, cfg=...)
```

### Metrics to Track
- Sanction events (correct vs mis-zap)
- Plant actions (color distribution)
- Harvest actions (color distribution)
- Berry counts by color over time
- Monoculture fraction over time

---

## Key Differences from Phase 1

**Phase 1**: Infrastructure only
- Simple sanctions, immunity, observations, reward tracking
- No agent policies (random or manual control)

**Phase 2**: Resident policies
- Deterministic scripted agents
- Play at equilibrium (enforce + monoculture)
- Identical across control/treatment
- Privileged info access via ResidentObserver

**Not in Phase 2**: Ego learning, metrics computation (those are Phase 3+)

---

## Critical Implementation Notes

### 1. Harvesting is Not Selective
Residents harvest **any** ripe berry, not just permitted color. The spec says:
> "Eating a permitted color does not make sense, agents can eat any berry."

R4 expects ≥95% harvests are permitted color because monoculture means most berries are permitted color.

### 2. Grey Agents Can Be Sanctioned
Even residents can turn grey after eating (stochastic conversion). If they stay grey past grace period (25 frames), they can be sanctioned by other residents. This is expected and correct - it incentivizes rapid replanting.

### 3. Immunity Prevents Dogpiling
Residents should NOT attempt to zap immune targets (P1 checks `immune_ticks_remaining == 0`). This prevents wasted zap actions and ensures residents move on to other priorities (plant, harvest, patrol).

### 4. Episode Length
Episodes are **2000 steps**, not 1000. Critical for R8 monoculture test.

### 5. Color Indexing
**Always 1-indexed in Lua** (1=RED, 2=GREEN, 3=BLUE, 0=GREY). ResidentInfoExtractor handles conversion to Python 0-indexed for agent_id only. Color indices stay 1-indexed in Python to match Lua.

### 6. Zap Cooldown Approximation
We approximate zap cooldown from READY_TO_SHOOT (binary). If not ready, assume full 4-frame cooldown. This is conservative - residents may wait 1-3 frames unnecessarily, but won't attempt invalid zaps.

---

## Next Steps (Component-by-Component)

1. ✅ `config.py` - Tunables
2. ✅ `info_extractor.py` + ResidentObserver Lua component
3. 🚧 `scripted_residents.py` - ResidentController with P1-P4 logic
4. ⬜ `resident_wrapper.py` - Integration with environment
5. ⬜ `phase2_residents_tests.py` - R1-R4 tests
6. ⬜ `phase2_residents_tests.py` - R5-R8 tests
7. ⬜ Final PHASE2_README.md update

---

## Commits So Far (Phase 2 Branch)

1. **23d9402**: Add resident agent configuration
2. **6dca512**: Add ResidentObserver component and info extractor

**Total changes**: 2 commits, ~350 lines of code (Lua + Python)

**Branch**: `phase-2` (based on `phase-1`)
