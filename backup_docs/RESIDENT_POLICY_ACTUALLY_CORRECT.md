# Resident Policy - ACTUALLY CORRECT Logic

## Understanding the Problem

**Goal**: Achieve 95%+ monoculture WITHIN EACH EPISODE (2000 steps)

**How planting works**:
1. Move around (explore/search)
2. When NEAR an unripe berry → plant action converts it to altar color
3. Continue moving to find more unripe berries
4. Repeat until 95%+ of berries are altar color

**You must MOVE to find unripe berries to plant on.**

---

## Correct Priority Order

### P1: SANCTIONING
**Condition**: Agent in range AND violating AND not immune AND ready

```python
if ready_to_shoot > 0.5 and world_step >= GRACE_PERIOD:
    for lua_idx in range(1, num_players + 1):
        if lua_idx == player_index:
            continue

        array_idx = lua_idx - 1

        # In range?
        if ids_in_range[array_idx] == 0:
            continue

        # Violating?
        color = agent_colors[array_idx]
        is_violating = (color == 0) or (color != altar_color)

        # Immune?
        is_immune = (immunity_status[array_idx] == 1)

        # Sanction
        if is_violating and not is_immune:
            return FIRE_ZAP
```

---

### P2: PLANTING (When Near Unripe Berry)
**Condition**: Near an unripe berry (plantable ground)

```python
# Check if we're near an unripe berry we can plant on
can_plant = self._check_can_plant(obs)

# If we can plant, do so with high probability (90%)
if can_plant and self._rng.random() < 0.9:
    return PLANT_MAP[altar_color]
```

**Key**: This only happens when NEAR an unripe berry. Otherwise we move.

---

### P3: MOVEMENT/EXPLORATION (Default)
**Condition**: Nothing else to do (no sanctioning, no planting)

```python
# Move around to find unripe berries to plant on
return self._explore_action()
```

**Most of the time we're moving**, looking for unripe berries.

---

## How to Detect "Can Plant"?

### Option A: Check Observation for Unripe Berry Nearby

If we have an observation like:
- `HAS_UNRIPE_BERRY_NEARBY` (binary)
- `UNRIPE_BERRY_FLAGS` (berry types visible)

```python
def _check_can_plant(self, obs):
    if 'HAS_UNRIPE_BERRY_NEARBY' in obs:
        return obs['HAS_UNRIPE_BERRY_NEARBY'] > 0.5
    return False
```

### Option B: Heuristic (Like HandcodedMonoculturePolicy)

Plant periodically to ensure we hit unripe berries:

```python
def _check_can_plant(self, obs):
    # Plant every N steps (assume we encounter berries while moving)
    return self._step_count % 10 < 3  # Plant 30% of time
```

This assumes that as we move around, we're near unripe berries 30% of the time.

### Option C: Always Try to Plant (Let Game Decide)

```python
def _check_can_plant(self, obs):
    # Always return True, plant action will fizzle if no berry nearby
    return True
```

If we plant when no berry is nearby, the action just does nothing (no harm).

---

## Complete Policy Logic

```python
class ResidentPolicy(policy.Policy):
    """Resident policy for 95%+ monoculture per episode."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._step_count = 0

        # Actions
        self.FORWARD = 1
        self.STEP_LEFT = 3
        self.STEP_RIGHT = 4
        self.TURN_LEFT = 5
        self.TURN_RIGHT = 6
        self.FIRE_ZAP = 7
        self.FIRE_RED = 8
        self.FIRE_GREEN = 9
        self.FIRE_BLUE = 10

        # Plant actions
        self.PLANT_MAP = {
            1: self.FIRE_RED,
            2: self.FIRE_GREEN,
            3: self.FIRE_BLUE,
        }

        # Parameters
        self.GRACE_PERIOD = 50
        self.PLANT_PROBABILITY = 0.9  # Plant 90% of time when near berry
        self.PLANT_FREQUENCY = 10  # Attempt plant every N steps (heuristic)

    def initial_state(self) -> Any:
        return None

    def step(self, timestep: dm_env.TimeStep, prev_state: Any) -> Tuple[int, Any]:
        obs = timestep.observation
        self._step_count += 1

        # Extract observations
        altar_color = int(obs['ALTAR'])
        ready_to_shoot = float(obs['READY_TO_SHOOT'])
        agent_colors = np.asarray(obs['AGENT_COLORS'])
        immunity_status = np.asarray(obs['IMMUNITY_STATUS'])
        ids_in_range = np.asarray(obs['AVATAR_IDS_IN_RANGE_TO_ZAP'])
        player_index = int(obs['PLAYER_INDEX'])

        # === P1: SANCTIONING ===
        if ready_to_shoot > 0.5 and self._step_count >= self.GRACE_PERIOD:
            for lua_idx in range(1, len(agent_colors) + 1):
                if lua_idx == player_index:
                    continue

                array_idx = lua_idx - 1

                if ids_in_range[array_idx] == 0:
                    continue

                color = agent_colors[array_idx]
                is_violating = (color == 0) or (color != altar_color)
                is_immune = (immunity_status[array_idx] == 1)

                if is_violating and not is_immune:
                    return self.FIRE_ZAP, None

        # === P2: PLANTING (when near unripe berry) ===
        can_plant = self._check_can_plant(obs)
        if can_plant and self._rng.random() < self.PLANT_PROBABILITY:
            return self.PLANT_MAP[altar_color], None

        # === P3: MOVEMENT (default - find berries) ===
        return self._explore_action(), None

    def _check_can_plant(self, obs):
        """Check if we can plant (near unripe berry)."""
        # Option A: Use observation if available
        if 'HAS_UNRIPE_BERRY_NEARBY' in obs:
            return obs['HAS_UNRIPE_BERRY_NEARBY'] > 0.5

        # Option B: Heuristic - plant periodically (assume we hit berries)
        return self._step_count % self.PLANT_FREQUENCY < 3

    def _explore_action(self) -> int:
        """Movement to find berries and spread across map."""
        action_probs = [
            (self.FORWARD, 0.5),
            (self.TURN_LEFT, 0.15),
            (self.TURN_RIGHT, 0.15),
            (self.STEP_LEFT, 0.1),
            (self.STEP_RIGHT, 0.1),
        ]

        rand = self._rng.random()
        cumsum = 0
        for action, prob in action_probs:
            cumsum += prob
            if rand < cumsum:
                return action

        return self.FORWARD

    def close(self) -> None:
        pass
```

---

## Behavior Pattern (Single Episode)

**Episode: 2000 steps, goal 95%+ monoculture**

```
Step 1-10: Move around, plant when near berry
  - Move FORWARD
  - See unripe berry (step 3) → PLANT_altar_color
  - Move FORWARD
  - See unripe berry (step 6) → PLANT_altar_color
  - ...

Step 11-20: Continue moving and planting
  - Turn LEFT
  - Move FORWARD
  - See unripe berry (step 13) → PLANT_altar_color
  - See violator in range → FIRE_ZAP
  - Move FORWARD
  - See unripe berry (step 16) → PLANT_altar_color
  - ...

... repeat for 2000 steps ...

Result:
- Planted altar color ~600 times (30% of steps * 2000)
- With 15 residents: ~9000 plants per episode
- Achieves 95%+ monoculture by step 2000
```

---

## Why This Achieves 95%+ Monoculture PER EPISODE

**Math per episode**:
- 2000 steps
- ~30% of steps we're near unripe berry (heuristic: step % 10 < 3)
- 90% of those we plant
- Effective planting: 2000 * 0.3 * 0.9 = 540 plants per resident
- 15 residents: 540 * 15 = 8,100 plants of altar color per episode

**Initial state**: ~33% each color (random)
**After 8,100 altar plants**: 95%+ altar color

**Each episode achieves monoculture independently.**

---

## Movement is Essential

**Without movement**: Can't find unripe berries → can't plant → no monoculture

**With movement**:
- Residents spread across map (50% forward, 30% turn, 20% step)
- Encounter unripe berries naturally
- Plant when encountered
- Cover whole map over 2000 steps

---

## Key Differences

**WRONG (previous version)**:
```python
if random() < 0.85:
    return PLANT  # Plant 85% of all steps
```
❌ Wastes plant actions when not near berries

**CORRECT (this version)**:
```python
if can_plant and random() < 0.9:
    return PLANT  # Plant 90% when NEAR berry

return MOVE  # Otherwise move (default)
```
✅ Move to find berries, plant when found

---

## Next Steps

1. **Decide on planting detection**:
   - Add `HAS_UNRIPE_BERRY_NEARBY` observation (accurate)?
   - Use heuristic `step % 10 < 3` (simple, like HandcodedMonoculturePolicy)?

2. **Implement Lua observations**:
   - AGENT_COLORS
   - IMMUNITY_STATUS
   - PLAYER_INDEX
   - (Optional) HAS_UNRIPE_BERRY_NEARBY

3. **Implement ResidentPolicy**

Which approach for planting detection should we use?
