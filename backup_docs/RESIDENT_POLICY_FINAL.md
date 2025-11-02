# Resident Policy - FINAL Logic (95%+ Monoculture)

## Requirements

1. **Achieve 95%+ monoculture** → Plant altar color FREQUENTLY
2. **Earn rewards** → Move around to eat ripe berries
3. **Enforce norm** → Sanction violators when opportunity arises

## Priority Order

### P1: SANCTIONING (Always Check First)
**Condition**: Ready AND agent in range AND violating AND not immune

```python
if ready_to_shoot > 0.5 and world_step >= GRACE_PERIOD:
    for lua_idx in range(1, num_players + 1):
        if lua_idx == player_index:
            continue

        array_idx = lua_idx - 1

        # In range?
        if ids_in_range[array_idx] == 0:
            continue

        # Violating? (grey OR wrong color)
        color = agent_colors[array_idx]
        is_violating = (color == 0) or (color != altar_color)

        # Immune?
        is_immune = (immunity_status[array_idx] == 1)

        # Sanction
        if is_violating and not is_immune:
            return FIRE_ZAP
```

---

### P2: FREQUENT PLANTING (80-90% of steps)
**Goal**: Achieve 95%+ monoculture by planting altar color very frequently

```python
# Plant 80-90% of the time to achieve monoculture
if self._rng.random() < 0.85:  # 85% chance to plant
    return PLANT_MAP[altar_color]
```

**Why so frequent?**
- Need 95%+ monoculture = 95%+ of berries are altar color
- Planting is the ONLY way to achieve this
- The more we plant, the faster monoculture converges
- HandcodedMonoculturePolicy uses 0.9 plant probability

---

### P3: EXPLORE/MOVEMENT (10-20% of steps)
**Goal**: Move around to find and eat ripe berries (earn rewards)

```python
# Explore: move around to find berries to eat
return self._explore_action()
```

**Movement strategy**:
```python
def _explore_action(self) -> int:
    """Exploration: move to find berries."""
    # Mix of forward movement and turning
    action_probs = [
        (FORWARD, 0.6),      # Move forward most often
        (TURN_LEFT, 0.15),   # Turn occasionally
        (TURN_RIGHT, 0.15),
        (STEP_LEFT, 0.05),
        (STEP_RIGHT, 0.05),
    ]

    rand = self._rng.random()
    cumsum = 0
    for action, prob in action_probs:
        cumsum += prob
        if rand < cumsum:
            return action

    return FORWARD
```

---

## Complete Policy Logic

```python
class ResidentPolicy(policy.Policy):
    """Resident policy optimized for 95%+ monoculture."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._step_count = 0

        # Actions
        self.NOOP = 0
        self.FORWARD = 1
        self.STEP_LEFT = 3
        self.STEP_RIGHT = 4
        self.TURN_LEFT = 5
        self.TURN_RIGHT = 6
        self.FIRE_ZAP = 7
        self.FIRE_RED = 8
        self.FIRE_GREEN = 9
        self.FIRE_BLUE = 10

        # Plant actions by altar color
        self.PLANT_MAP = {
            1: self.FIRE_RED,
            2: self.FIRE_GREEN,
            3: self.FIRE_BLUE,
        }

        # Parameters
        self.GRACE_PERIOD = 50  # frames
        self.PLANT_PROBABILITY = 0.85  # Plant 85% of time for monoculture

    def initial_state(self) -> Any:
        return None

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

        # === P1: SANCTIONING (Always check first) ===
        if ready_to_shoot > 0.5 and self._step_count >= self.GRACE_PERIOD:
            for lua_idx in range(1, len(agent_colors) + 1):
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
                    return self.FIRE_ZAP, None

        # === P2: FREQUENT PLANTING (85% of time) ===
        if self._rng.random() < self.PLANT_PROBABILITY:
            return self.PLANT_MAP[altar_color], None

        # === P3: EXPLORE (15% of time) ===
        return self._explore_action(), None

    def _explore_action(self) -> int:
        """Exploration action to find berries."""
        action_probs = [
            (self.FORWARD, 0.6),
            (self.TURN_LEFT, 0.15),
            (self.TURN_RIGHT, 0.15),
            (self.STEP_LEFT, 0.05),
            (self.STEP_RIGHT, 0.05),
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

## Why This Achieves 95%+ Monoculture

**Math**:
- 85% of steps plant altar color
- Each plant converts an unripe berry to altar color
- Over 2000 steps episode: ~1700 plants of altar color
- With 15 residents: ~25,500 plants per episode
- This creates massive monoculture dominance

**Comparison to HandcodedMonoculturePolicy**:
- altar-transfer uses `plant_probability=0.9`
- But only attempts planting 30% of the time (`step_count % 10 < 3`)
- Effective planting: 0.9 * 0.3 = 27% of steps
- We use 85% directly → much higher planting rate

---

## Multi-Env Compatibility

**Single-Env (ALTAR always RED)**:
```
Step 1: Check sanction → none → plant (85%) → PLANT_RED
Step 2: Check sanction → none → plant (85%) → PLANT_RED
Step 3: Check sanction → none → explore (15%) → FORWARD
Step 4: Check sanction → found! → FIRE_ZAP
...
```

**Multi-Env (ALTAR varies)**:
```
Episode 1 (ALTAR=RED):
  Step 1: plant (85%) → PLANT_RED
  Step 2: sanction → FIRE_ZAP
  Step 3: plant (85%) → PLANT_RED

Episode 2 (ALTAR=GREEN):
  Step 1: plant (85%) → PLANT_GREEN
  Step 2: explore (15%) → FORWARD
  Step 3: plant (85%) → PLANT_GREEN

Episode 3 (ALTAR=BLUE):
  Step 1: plant (85%) → PLANT_BLUE
  ...
```

**Same policy, adapts to ALTAR observation each episode.**

---

## Observations Needed

**Must have**:
- `ALTAR` - current norm (1/2/3)
- `READY_TO_SHOOT` - can I zap? (0.0/1.0)
- `AVATAR_IDS_IN_RANGE_TO_ZAP` - who can I zap? (binary vector)
- `AGENT_COLORS` - what color is each agent? (for violation detection)
- `IMMUNITY_STATUS` - is each agent immune? (for sanction eligibility)
- `PLAYER_INDEX` - which agent am I? (to skip self)

**Do NOT need**:
- Unripe berry detection (we plant randomly with high probability)
- RGB parsing (we have AGENT_COLORS)
- Events for decisions (only for metrics)

---

## Key Differences from Previous Versions

**Version 1 (WRONG)**:
```python
if my_color != altar_color:
    return PLANT  # Check own color first
```
❌ Wastes time, reactive

**Version 2 (WRONG)**:
```python
if can_plant:  # Need to detect planting opportunity
    return PLANT
```
❌ Complex, requires berry detection

**Version 3 (CORRECT)**:
```python
if random() < 0.85:  # Plant frequently
    return PLANT_altar_color
```
✅ Simple, achieves monoculture, naturally maintains compliance

---

## Expected Behavior

**Monoculture convergence**:
- Episode 1-10: 40-60% monoculture (mixed berries)
- Episode 11-50: 70-85% monoculture (dominance emerging)
- Episode 51+: 90-95%+ monoculture (equilibrium)

**Sanctioning effectiveness**:
- Residents sanction any non-compliant agent in range
- Grace period respected (first 50 frames)
- Immune agents avoided (no double sanctioning)

**Reward accumulation**:
- Residents move around 15% of time
- Find and eat ripe berries (step on them)
- Earn rewards to train alongside ego

---

## Implementation Priority

1. **Add Lua observations** (Phase 1)
2. **Implement ResidentPolicy with 85% plant rate** (Phase 2)
3. **Test monoculture convergence** (should reach 95%+)
4. **Tune PLANT_PROBABILITY if needed** (0.8-0.95 range)

Should I start implementing Phase 1 (Lua observation components)?
