# Resident Implementation Plan - NEW Approach

## Understanding of Resident Requirements

### Behavioral Requirements
Based on your description, residents must:

1. **Always be compliant** (play at equilibrium)
   - Body color = altar color at all times
   - Always plant berries of permitted (altar) color

2. **Sanction violators correctly**
   - Violation = grey (color_id=0) OR body_color ≠ altar_color
   - Must NOT sanction immune agents (one sanction per violation)
   - Must sanction ego when violating (high recall, no misses)
   - Must NOT sanction ego when compliant (high precision, no false positives)

3. **Testing focus**
   - Residents normally all compliant
   - Need specific tests where ego violates to verify sanctioning works
   - Need to measure both precision and recall of sanctioning

---

## Available Observations in New Environment

Based on study of `allelopathic_harvest_normative.py` and `avatar_library.lua`:

### Individual Observations (per agent)
1. **RGB** (shape: varies) - Egocentric visual view
   - Shows what the agent sees in their FOV
   - Contains visual representation of agents, berries, walls

2. **READY_TO_SHOOT** (shape: scalar) - Zap cooldown status
   - 1.0 when zap is ready (cooldown = 0)
   - 0.0 when zap is on cooldown (cooldown > 0)

3. **ALTAR** (shape: scalar) - Permitted color
   - Exposes altar color directly
   - 1=RED, 2=GREEN, 3=BLUE (Lua 1-indexed)

### Debug Observations (if _ENABLE_DEBUG_OBSERVATIONS = True)
4. **AVATAR_IDS_IN_VIEW** (shape: [num_players]) - Binary vector
   - 1 if avatar i is visible in FOV, 0 otherwise

5. **AVATAR_IDS_IN_RANGE_TO_ZAP** (shape: [num_players]) - Binary vector
   - 1 if avatar i can be zapped, 0 otherwise
   - Uses Zapper.getZappablePlayerIndices()

6. **LOCATION** - Position and orientation
7. **COLOR_ID** - Own body color
8. **MOST_TASTY_BERRY_ID** - Own taste preference

### Global Observations (if debug enabled)
9. **WORLD.RGB** - Global top-down view
10. **BERRIES_BY_TYPE** (shape: [3]) - Total berries of each color
11. **RIPE_BERRIES_BY_TYPE** (shape: [3]) - Ripe berries of each color
12. **UNRIPE_BERRIES_BY_TYPE** (shape: [3]) - Unripe berries of each color

### What's NOT Available in Observations
- Other agents' body colors (ColorZapper.colorId)
- Other agents' immunity status (ImmunityTracker.isImmune())
- Berry positions (x, y coordinates)
- Agent positions (except via LOCATION debug obs)
- Orientation of other agents

---

## How to Get Information in New Environment

### Option 1: Parse RGB Observation
The RGB observation contains visual information. We could:
- Parse the egocentric RGB view to detect:
  - Agent colors (grey=125, red=200, green=200 in respective channels)
  - Berry positions and colors
  - Walls and obstacles

**Pros**: No events, uses existing observations
**Cons**: Complex computer vision, inaccurate, computationally expensive

### Option 2: Direct Component Access (Python-side)
Since residents are scripted in Python, we can directly access Lua components:
- Access `simulation.getAvatarFromIndex(i)` to get other avatars
- Query `avatar.getComponent('ColorZapper').colorId` for body color
- Query `avatar.getComponent('ImmunityTracker').isImmune()` for immunity
- Query `berryObject.getComponent('Berry').colorId` for berry colors

**Pros**: Direct, accurate, no parsing needed
**Cons**: Requires Python-Lua bridge access (need to verify if accessible)

### Option 3: Add New Observation Components
Create new Lua observation components that expose:
- **NEARBY_AGENTS** - List of nearby agents with (id, color, immunity, position)
- **NEARBY_BERRIES** - List of nearby berries with (position, color, ripe/unripe)

**Pros**: Clean architecture, reusable
**Cons**: Requires Lua code changes, similar to old ResidentObserver approach

### Option 4: Use AVATAR_IDS_IN_RANGE_TO_ZAP + Direct Access
Combine debug observations with direct component access:
- Use AVATAR_IDS_IN_RANGE_TO_ZAP to know who's zappable
- Directly access ColorZapper.colorId and ImmunityTracker for those agents
- Use RGB to estimate positions if needed

**Pros**: Hybrid approach, leverages existing code
**Cons**: Still requires component access verification

---

## CRITICAL QUESTION: Python-Lua Bridge Access

**Do scripted residents (Python) have access to the dm_env/Lua simulation object?**

Looking at current code structure:
- `agents/residents/scripted_residents.py` receives `info` dict from `ResidentInfoExtractor`
- `ResidentInfoExtractor` receives `timestep` from dm_env

**Need to check**: Can we access `timestep.observation` AND the underlying simulation object to query components?

If NO direct access, we MUST use RGB parsing or create new observation components.
If YES, we can use Option 2 or 4.

---

## Recommended Approach (Pending Verification)

### Approach A: If Direct Component Access Available

**Implementation**:
1. Residents receive `env` or `simulation` reference at init
2. For each timestep:
   - Get altar color from ALTAR observation
   - Get own color from ColorZapper.colorId (or COLOR_ID debug obs)
   - Iterate through all players (1 to num_players):
     - Get avatar: `sim.getAvatarFromIndex(i)`
     - Get color: `avatar.getComponent('ColorZapper').colorId`
     - Get immunity: `avatar.getAllConnectedObjectsWithNamedComponent('ImmunityTracker')[0].getComponent('ImmunityTracker').isImmune()`
     - Check if violating: `color != altar_color`
     - Check if zappable: `AVATAR_IDS_IN_RANGE_TO_ZAP[i] == 1`
     - If violating AND not immune AND in range → zap
   - For harvesting/planting:
     - Parse RGB to find berries OR
     - Use GlobalBerryTracker to know berry counts
     - Move randomly while prioritizing altar-colored berries

**Pros**: No Lua changes, uses existing infrastructure
**Cons**: Requires verification that component access works from Python

### Approach B: If NO Direct Access (Pure Observation-based)

**Implementation**:
1. Enable `_ENABLE_DEBUG_OBSERVATIONS = True` for residents
2. Use available observations:
   - ALTAR for permitted color
   - COLOR_ID for own color
   - AVATAR_IDS_IN_RANGE_TO_ZAP for who can be zapped
   - RGB for visual parsing
3. Parse RGB to estimate:
   - Other agents' colors (color palette matching)
   - Berry positions
4. **Cannot detect immunity** - must assume immune if sanction just applied

**Pros**: No Lua changes
**Cons**: Cannot detect immunity accurately, RGB parsing complex

### Approach C: Add Minimal Observation Components (Recommended if A fails)

**Create new observation components** (similar to old approach but cleaner):

**New Lua Component**: `NearbyAgentsObservation`
- Attached to avatar
- Exposes observation: `NEARBY_AGENTS` (shape: [max_visible, 4])
  - For each visible agent: [player_id, body_color, is_immune, distance]
  - Sorted by distance
- Uses avatar's FOV and observation range

**New Lua Component**: `NearbyBerriesObservation`
- Exposes: `NEARBY_RIPE_BERRIES` and `NEARBY_UNRIPE_BERRIES`
- Shape: [max_visible, 3] - [rel_x, rel_y, color_id]

**Python Implementation**:
```python
class ResidentController:
    def act(self, obs):
        altar_color = obs['ALTAR']
        ready_to_shoot = obs['READY_TO_SHOOT']
        nearby_agents = obs['NEARBY_AGENTS']  # shape: [N, 4]

        # Find violators in range
        for agent_data in nearby_agents:
            player_id, body_color, is_immune, distance = agent_data
            if body_color != altar_color and not is_immune:
                # Turn toward and zap
                return ACTION_FIRE_ZAP

        # Harvest/plant logic using berry observations
        ...
```

**Pros**: Clean, accurate, no parsing
**Cons**: Requires Lua changes (but minimal, just observations)

---

## Implementation Steps

### Phase 1: Verify Component Access (DO THIS FIRST)
1. Check if Python wrapper can access `env._impl` or simulation object
2. Test if `sim.getAvatarFromIndex(i).getComponent('ColorZapper').colorId` works from Python
3. Document what's accessible

### Phase 2: Choose Implementation Approach
Based on Phase 1 results:
- If component access works → Use Approach A
- If not → Use Approach C (add observation components)

### Phase 3: Implement Resident Controller
**Core logic** (same for all approaches, only data source differs):

```python
class ResidentController:
    def __init__(self):
        self._rng = None
        self._patrol_state = {}

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self._patrol_state = {}

    def act(self, resident_id, observations):
        """Select action for resident.

        Priority:
        1. Maintain compliance (always correct color)
        2. Sanction violators (if in range and not immune)
        3. Harvest ripe berries
        4. Plant permitted color
        5. Patrol
        """
        altar_color = observations['ALTAR']
        ready_to_shoot = observations['READY_TO_SHOOT']

        # P1: Ensure own compliance
        own_color = self._get_own_color(observations)
        if own_color != altar_color:
            # Plant correct color to become compliant
            return self._plant_action(altar_color)

        # P2: Sanction violators
        if ready_to_shoot > 0.5:
            violators = self._find_violators(observations, altar_color)
            if violators:
                return self._zap_nearest_violator(violators)

        # P3: Harvest ripe berries
        harvest_action = self._try_harvest(observations)
        if harvest_action is not None:
            return harvest_action

        # P4: Plant permitted color
        plant_action = self._try_plant(observations, altar_color)
        if plant_action is not None:
            return plant_action

        # P5: Patrol
        return self._patrol(resident_id)

    def _find_violators(self, obs, altar_color):
        """Find agents in range who are violating and not immune."""
        violators = []
        nearby_agents = self._get_nearby_agents(obs)  # Data source varies by approach

        for agent_info in nearby_agents:
            player_id = agent_info['player_id']
            body_color = agent_info['body_color']
            is_immune = agent_info['is_immune']
            in_zap_range = agent_info['in_zap_range']

            # Check violation: grey OR wrong color
            is_violating = (body_color == 0) or (body_color != altar_color)

            if is_violating and not is_immune and in_zap_range:
                violators.append(agent_info)

        return violators
```

### Phase 4: Testing
Create test scenarios:
1. **R_SANCTION_PRECISION**: Ego compliant, verify residents don't sanction
2. **R_SANCTION_RECALL**: Ego violating, verify residents DO sanction
3. **R_IMMUNITY_RESPECT**: Ego immune, verify residents don't re-sanction
4. **R_COMPLIANCE**: Verify residents always have correct color

---

## Key Differences from Old Approach

### Old (Broken) Approach:
- ResidentObserver Lua component emitting events
- ResidentInfoExtractor parsing events
- Event-based architecture (prone to missing events)
- Complex event matching logic

### New Approach (Recommended):
- Direct observations or component access
- No event parsing (except optionally for debugging)
- Simpler, more reliable
- Uses existing observation infrastructure

### What Stays the Same:
- Policy priority: sanction > harvest > plant > patrol
- Violation logic: body_color != altar_color
- Immunity tracking: don't sanction immune agents
- Python-side controller logic

---

## Next Steps

**IMMEDIATE**: I need to verify which approach is feasible by checking:
1. Can Python access Lua simulation object?
2. Can Python query ColorZapper.colorId for other agents?
3. Can Python query ImmunityTracker.isImmune() for other agents?

**Please advise**: Should I:
- A) Investigate component access from Python (read wrapper code)
- B) Assume Approach C and start implementing observation components
- C) You tell me which approach is feasible given the codebase architecture

Once we determine the feasible approach, I can implement the resident controller with the correct data access pattern.
