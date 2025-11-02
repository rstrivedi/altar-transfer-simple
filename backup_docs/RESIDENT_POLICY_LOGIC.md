# Resident Policy - Detailed Logic

## Observations Used (NO EVENTS)

```python
obs = timestep.observation

# What norm is active?
altar_color = int(obs['ALTAR'])  # 1=RED, 2=GREEN, 3=BLUE

# Can I zap?
ready_to_shoot = float(obs['READY_TO_SHOOT'])  # 1.0=ready, 0.0=cooling

# Who can I zap RIGHT NOW?
ids_in_range = np.asarray(obs['AVATAR_IDS_IN_RANGE_TO_ZAP'])  # [num_players] binary vector

# All agents' states
agent_colors = np.asarray(obs['AGENT_COLORS'])  # [num_players] - 0=GREY, 1=RED, 2=GREEN, 3=BLUE
immunity_status = np.asarray(obs['IMMUNITY_STATUS'])  # [num_players] - 1=immune, 0=not

# Who am I?
player_index = int(obs['PLAYER_INDEX'])  # Lua 1-indexed
```

**NO event parsing. Everything from observations.**

---

## Action Priority (Strict Ordering)

### P1: COMPLIANCE (Highest Priority)
**Goal**: Ensure my own body color matches altar color

```python
own_color = agent_colors[player_index - 1]  # Lua 1-indexed → array 0-indexed

if own_color != altar_color:
    # IMMEDIATELY plant correct color to become compliant
    return PLANT_MAP[altar_color]  # FIRE_RED / FIRE_GREEN / FIRE_BLUE
```

**Why highest priority?**
- Body color comes from planting
- If I'm non-compliant, I'm a violator
- Must fix ASAP before doing anything else

---

### P2: SANCTIONING (Second Priority)
**Goal**: Sanction eligible violators within zap range

```python
# Can only sanction if ready
if ready_to_shoot < 0.5:
    continue to P3

# Check grace period (first 50 frames, sanctions fizzle)
if world_step < GRACE_PERIOD:
    continue to P3

# Find eligible targets to sanction
for lua_idx in range(1, num_players + 1):
    array_idx = lua_idx - 1

    # Skip self
    if lua_idx == player_index:
        continue

    # Check if in zap range RIGHT NOW
    if ids_in_range[array_idx] == 0:
        continue  # Not in range, skip

    # Check violation
    color = agent_colors[array_idx]
    is_violating = (color == 0) or (color != altar_color)  # GREY or WRONG COLOR

    # Check immunity
    is_immune = (immunity_status[array_idx] == 1)

    # Sanction if: violating AND not immune AND in range
    if is_violating and not is_immune:
        return FIRE_ZAP  # Zap this violator
```

**Key points**:
- Uses `AVATAR_IDS_IN_RANGE_TO_ZAP` to know who's in range
- Checks violation: grey (0) OR wrong color
- Checks immunity to avoid re-sanctioning
- Zaps first eligible violator found

---

### P3: MAINTENANCE PLANTING (Third Priority)
**Goal**: Keep planting permitted color frequently to maintain compliance

```python
# Even if already compliant, keep planting to ensure body stays correct color
frames_since_plant += 1

if frames_since_plant >= PLANT_FREQUENCY:  # e.g., every 2 frames
    frames_since_plant = 0
    return PLANT_MAP[altar_color]
```

**Why needed?**
- Body color decays or changes from eating berries
- Need to keep planting frequently to stay compliant
- Creates "compliance maintenance" behavior

---

### P4: PATROL (Fallback)
**Goal**: Random movement when nothing else to do

```python
# Random movement
if patrol_frames_remaining <= 0:
    patrol_action = random.choice([FORWARD, TURN_LEFT, TURN_RIGHT])
    patrol_frames_remaining = PATROL_PERSISTENCE

patrol_frames_remaining -= 1
return patrol_action
```

---

## Full Policy Step Logic

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

    own_color = agent_colors[player_index - 1]

    # === P1: COMPLIANCE ===
    if own_color != altar_color:
        return self.PLANT_MAP[altar_color], None

    # === P2: SANCTIONING ===
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

    # === P3: MAINTENANCE PLANTING ===
    self._frames_since_plant += 1
    if self._frames_since_plant >= self.PLANT_FREQUENCY:
        self._frames_since_plant = 0
        return self.PLANT_MAP[altar_color], None

    # === P4: PATROL ===
    return self._patrol(), None
```

---

## What About Events?

**Events are ONLY for telemetry/metrics, NOT decision making.**

```python
# In wrapper/environment code (NOT in policy):
events = env.events()  # Get raw events
parsed = parse_events(events)  # Parse for logging

for event in parsed:
    if event['name'] == 'sanction':
        # Log to metrics
        metrics['sanctions_given'] += 1
        # Log to wandb, tensorboard, etc.

# Events are NEVER passed to policy.step()
# Policy sees ONLY observations
```

---

## Observations Needed (Final List)

**Already have**:
- `ALTAR` - permitted color
- `READY_TO_SHOOT` - zap cooldown

**Need to add**:
- `AVATAR_IDS_IN_RANGE_TO_ZAP` - who can I zap (binary vector)
- `AGENT_COLORS` - all agents' colors
- `IMMUNITY_STATUS` - all agents' immunity
- `PLAYER_INDEX` - my own index

**Do NOT need**:
- Events for decision making
- RGB parsing for agent detection (we have AGENT_COLORS)
- Complex info extraction (just read observations)

---

## Why This Works for Multi-Env

**Single-Env (always RED)**:
```
Episode 1: ALTAR = 1 → plant RED, sanction non-RED
Episode 2: ALTAR = 1 → plant RED, sanction non-RED
Episode 3: ALTAR = 1 → plant RED, sanction non-RED
```

**Multi-Env (random RED/GREEN/BLUE)**:
```
Episode 1: ALTAR = 1 → plant RED, sanction non-RED
Episode 2: ALTAR = 3 → plant BLUE, sanction non-BLUE
Episode 3: ALTAR = 2 → plant GREEN, sanction non-GREEN
```

**Same policy, reads ALTAR each episode, adapts behavior.**

---

## Summary

**What resident does**:
1. Check own color → if wrong, plant correct color
2. Check for violators in range → if exists, zap
3. Plant frequently to maintain compliance
4. Patrol when idle

**What resident observes**:
- ALTAR (norm)
- AGENT_COLORS (who's violating)
- IMMUNITY_STATUS (who can't be sanctioned)
- AVATAR_IDS_IN_RANGE_TO_ZAP (who's in range)
- READY_TO_SHOOT (can I zap)
- PLAYER_INDEX (who am I)

**What resident DOES NOT use**:
- Events for decisions (events only for metrics)
- RGB parsing for agent colors (have AGENT_COLORS)
- Complex info extraction (direct observation access)

---

## Event Parser Status

**REMOVE** for decision making.

**KEEP** only for metrics/logging in wrappers:
```python
# In SB3Wrapper or similar:
events = env.events()
parsed = parse_events(events)  # For logging only
recorder.record_events(parsed)  # Metrics tracking

# Policy NEVER sees events
action, state = policy.step(timestep, prev_state)
```

If event parser is ONLY used by residents for decisions → **DELETE IT**.
If used for metrics → **KEEP IT** but only in wrapper/recorder code, not in policy.
