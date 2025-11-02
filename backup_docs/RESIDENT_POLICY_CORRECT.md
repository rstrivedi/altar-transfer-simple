# Resident Policy - CORRECT Logic

## Priority Order (CORRECT)

### P1: SANCTIONING (Highest Priority)
**Condition**: Agent in view AND violating AND not immune AND ready to shoot

```python
if ready_to_shoot > 0.5 and world_step >= GRACE_PERIOD:
    for lua_idx in range(1, num_players + 1):
        if lua_idx == player_index:
            continue  # Skip self

        array_idx = lua_idx - 1

        # Check if in range
        if ids_in_range[array_idx] == 0:
            continue

        # Check violation: grey OR wrong color
        color = agent_colors[array_idx]
        is_violating = (color == 0) or (color != altar_color)

        # Check immunity
        is_immune = (immunity_status[array_idx] == 1)

        # Sanction
        if is_violating and not is_immune:
            return FIRE_ZAP
```

---

### P2: PLANTING (Second Priority)
**Condition**: Planting opportunity exists (unripe berry in view)

```python
# Check if there's an unripe berry nearby to plant on
if has_unripe_berry_in_view(obs):
    return PLANT_MAP[altar_color]
```

**Key**: Always plant altar color when opportunity exists. This naturally maintains compliance.

**NO checking own color** - body color will converge to altar color through frequent planting.

---

### P3: EXPLORE (Fallback)
**Condition**: Nothing else to do

```python
return random_patrol_action()
```

---

## Detecting Planting Opportunities

We need to know if there's an unripe berry in view. Options:

### Option A: Add UnripeBerryObservation (Lua)

```lua
local UnripeBerryObservation = class.Class(component.Component)

function UnripeBerryObservation:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.HAS_UNRIPE_BERRY_IN_VIEW',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        -- Check if any unripe berry exists in agent's view
        -- (implementation depends on how berries are tracked)
        return has_unripe_berry and 1.0 or 0.0
      end
  }
end
```

### Option B: Simple Heuristic (Like HandcodedMonoculturePolicy)

```python
def _check_can_plant(self, obs):
    """Check if agent should attempt planting."""
    # Simple heuristic: plant 30% of the time
    return self._step_count % 10 < 3
```

### Option C: Use Existing Observations

Check if there's already an observation for berries (like `HAS_UNRIPE_BERRY` or berry flags).

---

## Complete Policy Logic

```python
def step(self, timestep: dm_env.TimeStep, prev_state: Any) -> Tuple[int, Any]:
    """Select action based on observations."""
    obs = timestep.observation
    self._step_count += 1

    # === Extract Observations ===
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

            # Check if in range
            if ids_in_range[array_idx] == 0:
                continue

            # Check violation
            color = agent_colors[array_idx]
            is_violating = (color == 0) or (color != altar_color)

            # Check immunity
            is_immune = (immunity_status[array_idx] == 1)

            # Sanction
            if is_violating and not is_immune:
                return self.FIRE_ZAP, None

    # === P2: PLANTING ===
    # Check if planting opportunity exists
    can_plant = self._check_can_plant(obs)
    if can_plant:
        return self.PLANT_MAP[altar_color], None

    # === P3: EXPLORE ===
    return self._patrol(), None

def _check_can_plant(self, obs):
    """Check if planting opportunity exists."""
    # Option A: Use observation
    if 'HAS_UNRIPE_BERRY_IN_VIEW' in obs:
        return obs['HAS_UNRIPE_BERRY_IN_VIEW'] > 0.5

    # Option B: Heuristic fallback
    return self._step_count % 10 < 3  # Plant 30% of the time
```

---

## Why This Is Better

**OLD (Wrong)**:
```python
# Check own color first
if my_color != altar_color:
    return PLANT  # Fix compliance

# Then sanction
if can_sanction:
    return ZAP

# Then patrol
return PATROL
```

**Problems**:
1. Wastes time checking own color
2. Might not plant if already "compliant" (but color can drift)
3. Reactive rather than proactive

**NEW (Correct)**:
```python
# Sanction if opportunity
if can_sanction:
    return ZAP

# Plant if opportunity (always altar color)
if can_plant:
    return PLANT_altar_color  # Naturally maintains compliance

# Explore
return PATROL
```

**Benefits**:
1. Prioritizes sanctioning (enforcement first)
2. Plants altar color whenever possible (proactive compliance)
3. Simpler logic, no color checking
4. Body color converges to altar color naturally

---

## Observations Needed

**Already have**:
- `ALTAR` - permitted color
- `READY_TO_SHOOT` - zap cooldown
- `AVATAR_IDS_IN_RANGE_TO_ZAP` - who can I zap

**Need to add**:
- `AGENT_COLORS` - all agents' colors (for violation detection)
- `IMMUNITY_STATUS` - all agents' immunity (for sanction eligibility)
- `PLAYER_INDEX` - my own index (to skip self)
- `HAS_UNRIPE_BERRY_IN_VIEW` - planting opportunity (optional, can use heuristic)

---

## Multi-Env Compatibility

**Single-Env (always RED)**:
```
Step 1: ALTAR=1 → if can_plant: PLANT_RED
Step 2: ALTAR=1 → if can_sanction: ZAP
Step 3: ALTAR=1 → if can_plant: PLANT_RED
...
```

**Multi-Env (varies)**:
```
Episode 1, Step 1: ALTAR=1 → if can_plant: PLANT_RED
Episode 1, Step 2: ALTAR=1 → if can_sanction: ZAP
Episode 2, Step 1: ALTAR=3 → if can_plant: PLANT_BLUE
Episode 2, Step 2: ALTAR=3 → if can_sanction: ZAP
...
```

**Same policy, reads ALTAR and adapts.**

---

## Next Question

**How should we detect planting opportunities?**

1. **Add Lua observation** `HAS_UNRIPE_BERRY_IN_VIEW` (accurate but requires Lua code)
2. **Use heuristic** like `step_count % 10 < 3` (simple, works okay)
3. **Check if observation already exists** in substrate config

Which approach should we use?
