# FiLM Ablation Study: Mechanism Validation

**Status**: Deferred until after main Phase 4/5 experiments complete
**Date**: 2025-11-14
**Purpose**: Validate that FiLM learns color-specific modulation (not just benefiting from any signal)

---

## 1. MOTIVATION

### The Problem

Current experiment compares:
- **Treatment**: Has `permitted_color` observation → FiLM([1,0,0] or [0,1,0] or [0,0,1])
- **Control**: No `permitted_color` observation → FiLM([0,0,0])

**If Treatment beats Control, what did we learn?**

**Possibility 1**: FiLM learns color-specific modulation
- γ(RED) ≠ γ(GREEN) ≠ γ(BLUE)
- Policy has distinct feature modulation for each community
- ✓ This is what we want!

**Possibility 2**: Any non-zero signal helps (FiLM is broken)
- γ(RED) ≈ γ(GREEN) ≈ γ(BLUE) ≈ γ(random noise)
- FiLM just provides gradient flow or exploration bias
- ✗ Would invalidate the mechanism!

**Possibility 3**: Benefit comes from architecture, not signal content
- Maybe just having FiLM modules (even with zeros) helps
- Or maybe concatenation would work just as well
- Would change our interpretation

### The Solution: Ablation Suite

Test **4 configurations** to disentangle:
1. **Correct-FiLM**: Treatment (current) - altar color → FiLM
2. **Null-FiLM**: Control (current) - zeros → FiLM
3. **Random-FiLM**: NEW - random noise → FiLM
4. **No-FiLM**: NEW - altar color → concatenation (no FiLM)

---

## 2. COMPLETE ABLATION MATRIX

| Ablation | Obs Access | FiLM Input | Architecture | Treatment Arm? |
|----------|------------|------------|--------------|----------------|
| **Correct-FiLM** | ✅ permitted_color | Altar color [1,0,0]/[0,1,0]/[0,0,1] | FiLM modulation | YES |
| **Null-FiLM** | ❌ None | Zeros [0,0,0] | FiLM modulation | NO |
| **Random-FiLM** | ✅ permitted_color | Random noise per episode | FiLM modulation | YES |
| **No-FiLM-Concat** | ✅ permitted_color | N/A (concat to features) | Direct concatenation | YES |

### Detailed Architectures

#### **Correct-FiLM** (Current Treatment)
```python
# Observations
obs = {'rgb': (88,88,3), 'ready_to_shoot': (1,), 'permitted_color': (3,)}

# Forward pass
h = CNN(rgb) + Linear([h_visual, ready_to_shoot])  # → (256,)
conditioning = permitted_color  # [1,0,0] for RED, [0,1,0] for GREEN, [0,0,1] for BLUE
h_tilde = γ(conditioning) ⊙ h + β(conditioning)  # Global FiLM

# Action heads
game_logits = MLP(h_tilde)  # → (10,) non-zap actions
sanction_h = Linear(h_tilde) → FiLM_local(conditioning, sanction_h) → SiLU → Linear  # → (1,) zap action

# Value head
value = MLP(h_tilde)  # → (1,)
```

**FiLM learns**:
- γ_RED, β_RED for RED community
- γ_GREEN, β_GREEN for GREEN community
- γ_BLUE, β_BLUE for BLUE community

Each should modulate features differently to produce color-specific behavior.

---

#### **Null-FiLM** (Current Control)
```python
# Observations
obs = {'rgb': (88,88,3), 'ready_to_shoot': (1,)}  # NO permitted_color

# Forward pass
h = CNN(rgb) + Linear([h_visual, ready_to_shoot])  # → (256,)
conditioning = zeros([0,0,0])  # Fixed null token
h_tilde = γ(zeros) ⊙ h + β(zeros)  # ≈ h at initialization (identity)

# Rest same as Correct-FiLM
```

**FiLM behavior**:
- Initialized to identity: γ(0) = 1, β(0) = 0
- Can learn to deviate from identity, but conditioning is always zeros
- No color-specific modulation possible

---

#### **Random-FiLM** (New Ablation)
```python
# Observations
obs = {'rgb': (88,88,3), 'ready_to_shoot': (1,), 'permitted_color': (3,)}

# Forward pass
h = CNN(rgb) + Linear([h_visual, ready_to_shoot])  # → (256,)

# ABLATION: Ignore permitted_color, use random noise instead
conditioning = randn([3])  # Sample at episode start, fixed for episode
# e.g., [0.234, -0.671, 0.912]

h_tilde = γ(conditioning) ⊙ h + β(conditioning)

# Rest same as Correct-FiLM
```

**FiLM behavior**:
- Gets different random vector each episode
- Cannot learn color-specific modulation (no consistent mapping)
- Tests if FiLM needs CORRECT altar color or just ANY varying signal

**Variants**:
- **random_episodic**: Sample once per episode, constant within episode
- **random_step**: Sample every step (even more chaotic)
- **fixed_random**: Same random vector all episodes (e.g., [0.5, 0.3, 0.8])

---

#### **No-FiLM-Concat** (New Ablation)
```python
# Observations
obs = {'rgb': (88,88,3), 'ready_to_shoot': (1,), 'permitted_color': (3,)}

# Forward pass
h = CNN(rgb) + Linear([h_visual, ready_to_shoot])  # → (256,)

# ABLATION: Concatenate permitted_color directly (no FiLM)
h_concat = concat([h, permitted_color])  # → (259,)

# Action heads (take concatenated input)
game_logits = MLP(h_concat)  # → (10,)
sanction_logit = MLP(h_concat)  # → (1,)

# Value head
value = MLP(h_concat)  # → (1,)
```

**Architecture differences**:
- No FiLM modulation at all
- Permitted color added as extra features
- Simpler architecture (standard MLP)
- Tests if FiLM's multiplicative gating is necessary

---

## 3. WHAT EACH COMPARISON TESTS

### **Comparison 1: Correct-FiLM vs Null-FiLM**

**Question**: Does having institutional signal help?

**Expected**: Correct-FiLM >> Null-FiLM

**This is the main hypothesis test** (current experiment).

---

### **Comparison 2: Correct-FiLM vs Random-FiLM** ⭐ CRITICAL

**Question**: Does FiLM learn color-specific modulation, or does any varying signal help?

**Expected**: Correct-FiLM >> Random-FiLM

**If Correct ≈ Random**:
- 🚨 **FiLM is broken!**
- Policy not using altar color semantics
- Benefit comes from something else (e.g., gradient flow, exploration)
- Would invalidate our interpretation

**If Correct >> Random**:
- ✅ **FiLM works as intended!**
- Policy learns distinct γ/β for each color
- Validates mechanism

---

### **Comparison 3: Correct-FiLM vs No-FiLM-Concat**

**Question**: Is FiLM's multiplicative modulation better than additive concatenation?

**Expected**: Unclear (both could work)

**Possible outcomes**:

**Correct ≈ No-FiLM-Concat**:
- Both architectures sufficient
- Signal access is key, architecture less important
- FiLM provides no architectural advantage

**Correct >> No-FiLM-Concat**:
- FiLM's multiplicative gating superior
- Feature modulation more effective than concatenation
- Validates architectural choice

**Correct < No-FiLM-Concat**:
- Concatenation simpler and works better
- FiLM might be overparameterized
- Should reconsider architecture

---

### **Comparison 4: Random-FiLM vs Null-FiLM**

**Question**: Does any non-zero signal help compared to zeros?

**Expected**: Random ≈ Null (both meaningless)

**If Random > Null**:
- Suggests FiLM benefits from ANY varying signal
- Would be concerning (mechanism not color-specific)

---

### **Comparison 5: No-FiLM-Concat vs Null-FiLM**

**Question**: Does concatenation architecture work?

**Expected**: No-FiLM-Concat >> Null-FiLM

**This validates that having signal access helps** (sanity check).

---

## 4. MULTI-COMMUNITY TEST (Phase 5)

Random-FiLM ablation is **especially diagnostic in multi-community mode**:

**Setup**:
- Each episode samples RED, GREEN, or BLUE
- Policy must learn distinct behaviors for each

**Correct-FiLM-Multi**:
```
Episode 1: Sample RED   → FiLM([1,0,0]) → Learn "plant red" policy
Episode 2: Sample GREEN → FiLM([0,1,0]) → Learn "plant green" policy
Episode 3: Sample BLUE  → FiLM([0,0,1]) → Learn "plant blue" policy
```
Should learn 3 distinct policies conditioned on FiLM input.

**Random-FiLM-Multi**:
```
Episode 1: Sample RED   → FiLM([0.23, -0.67, 0.91]) → ???
Episode 2: Sample GREEN → FiLM([-0.12, 0.44, 0.78]) → ???
Episode 3: Sample BLUE  → FiLM([0.91, 0.05, -0.33]) → ???
```
Random conditioning has no consistent mapping to community. Should fail!

**Expected**:
- Correct-FiLM-Multi >> Random-FiLM-Multi (strong evidence)
- Random-FiLM-Multi ≈ Null-FiLM-Multi (both can't condition)

**This is the strongest test** that FiLM learns color-specific modulation.

---

## 5. IMPLEMENTATION APPROACH

### Option 1: Environment Wrapper (Recommended)

Create `FiLMAblationWrapper` that modifies observations:

```python
# agents/envs/film_ablation_wrapper.py

import numpy as np
from typing import Optional

class FiLMAblationWrapper:
    """Wrapper to modify permitted_color for FiLM ablations.

    Supports:
        - 'correct': Use actual altar color (treatment)
        - 'null': Remove permitted_color (control)
        - 'random_episodic': Random vector per episode
        - 'random_step': Random vector per step
        - 'fixed_random': Fixed random vector [0.5, 0.3, 0.8]
        - 'permuted': Wrong color mapping (RED→GREEN, GREEN→BLUE, BLUE→RED)
    """

    def __init__(self, env, ablation_mode: str = 'correct', seed: Optional[int] = None):
        """Initialize ablation wrapper.

        Args:
            env: Base environment to wrap
            ablation_mode: Type of ablation
            seed: Random seed for reproducibility
        """
        self.env = env
        self.ablation_mode = ablation_mode
        self.rng = np.random.RandomState(seed)

        # For episodic random: store current episode's vector
        self.current_random_vector = None

        # For permutation ablation
        self.color_permutation = {
            0: np.array([0., 1., 0.], dtype=np.float32),  # RED → GREEN
            1: np.array([0., 0., 1.], dtype=np.float32),  # GREEN → BLUE
            2: np.array([1., 0., 0.], dtype=np.float32),  # BLUE → RED
        }

        # For fixed random
        self.fixed_random_vector = np.array([0.5, 0.3, 0.8], dtype=np.float32)

    def reset(self, **kwargs):
        """Reset environment and apply ablation."""
        obs, info = self.env.reset(**kwargs)

        # Generate new random vector for episodic ablation
        if self.ablation_mode == 'random_episodic':
            self.current_random_vector = self.rng.randn(3).astype(np.float32)

        obs = self._modify_observation(obs)
        return obs, info

    def step(self, action):
        """Step environment and apply ablation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._modify_observation(obs)
        return obs, reward, terminated, truncated, info

    def _modify_observation(self, obs):
        """Modify permitted_color based on ablation mode."""

        # Correct mode: no modification
        if self.ablation_mode == 'correct':
            return obs

        # Null mode: remove permitted_color entirely
        if self.ablation_mode == 'null':
            obs_copy = obs.copy()
            if 'permitted_color' in obs_copy:
                del obs_copy['permitted_color']
            return obs_copy

        # All other modes require permitted_color to exist
        if 'permitted_color' not in obs:
            return obs  # Control arm, no modification possible

        obs_copy = obs.copy()

        # Random episodic: use stored random vector
        if self.ablation_mode == 'random_episodic':
            obs_copy['permitted_color'] = self.current_random_vector.copy()

        # Random step: new random vector every step
        elif self.ablation_mode == 'random_step':
            obs_copy['permitted_color'] = self.rng.randn(3).astype(np.float32)

        # Fixed random: same vector every episode
        elif self.ablation_mode == 'fixed_random':
            obs_copy['permitted_color'] = self.fixed_random_vector.copy()

        # Permuted: wrong color mapping
        elif self.ablation_mode == 'permuted':
            color_idx = np.argmax(obs['permitted_color'])
            obs_copy['permitted_color'] = self.color_permutation[color_idx]

        else:
            raise ValueError(f"Unknown ablation_mode: {self.ablation_mode}")

        return obs_copy

    def __getattr__(self, name):
        """Forward all other attributes to wrapped environment."""
        return getattr(self.env, name)
```

**Usage in sb3_wrapper.py**:
```python
from agents.envs.film_ablation_wrapper import FiLMAblationWrapper

def make_vec_env_treatment(..., ablation_mode='correct'):
    # ... create base env ...

    # Wrap with ablation wrapper
    env = FiLMAblationWrapper(env, ablation_mode=ablation_mode, seed=seed)

    return env
```

**Config**:
```yaml
# configs/treatment_random.yaml
env:
  film_ablation: 'random_episodic'  # NEW parameter
  permitted_color_index: 1
  # ... rest same
```

---

### Option 2: Policy-Level Modification

Modify `FiLMTwoHeadPolicy._get_conditioning()`:

```python
class FiLMTwoHeadPolicy(ActorCriticPolicy):
    def __init__(self, ..., film_ablation='correct'):
        super().__init__(...)
        self.film_ablation = film_ablation

        # For random ablations
        if film_ablation == 'fixed_random':
            self.register_buffer(
                'fixed_random_vector',
                torch.tensor([0.5, 0.3, 0.8])
            )

        # For episodic random
        self.current_random_vector = None
        self.episode_step = 0

    def _get_conditioning(self, observations):
        """Get conditioning with ablation support."""
        batch_size = observations['rgb'].shape[0]
        device = observations['rgb'].device

        # === Ablation: Random episodic ===
        if self.film_ablation == 'random_episodic':
            # Reset at episode start (heuristic: when step resets to 0)
            if self.episode_step == 0 or self.current_random_vector is None:
                self.current_random_vector = torch.randn(3, device=device)
            self.episode_step += 1
            return self.current_random_vector.unsqueeze(0).expand(batch_size, -1)

        # === Ablation: Random step ===
        if self.film_ablation == 'random_step':
            return torch.randn(batch_size, 3, device=device)

        # === Ablation: Fixed random ===
        if self.film_ablation == 'fixed_random':
            return self.fixed_random_vector.unsqueeze(0).expand(batch_size, -1)

        # === Correct (default) ===
        if 'permitted_color' in observations:
            return observations['permitted_color']
        else:
            return torch.zeros(batch_size, 3, device=device)
```

**Problem with this approach**: Harder to detect episode boundaries in policy. Wrapper approach is cleaner.

---

### Option 3: Separate Policy Class for No-FiLM-Concat

Create `ConcatTwoHeadPolicy` for concatenation ablation:

```python
# agents/train/concat_policy.py

class ConcatTwoHeadPolicy(ActorCriticPolicy):
    """Two-head policy with concatenation (no FiLM).

    Architecture:
        - Trunk: CNN + vector concat → h (256,)
        - Concatenate permitted_color: [h, permitted_color] → (259,)
        - Game head: concat_h → MLP → 10 logits
        - Sanction head: concat_h → MLP → 1 logit
        - Value head: concat_h → MLP → scalar

    No FiLM modulation. Simpler architecture for comparison.
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 trunk_dim=256, sanction_hidden_dim=128,
                 ent_coef_game=0.01, ent_coef_sanction=0.02, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.trunk_dim = trunk_dim
        self.sanction_hidden_dim = sanction_hidden_dim
        self.ent_coef_game = ent_coef_game
        self.ent_coef_sanction = ent_coef_sanction

        # Build custom architecture
        self._build()

    def _build(self):
        """Build concatenation architecture."""
        # Features extractor (CNN + vector)
        self.features_extractor = DictFeaturesExtractor(
            self.observation_space,
            trunk_dim=self.trunk_dim
        )

        # Check if permitted_color exists
        self.has_permitted_color = 'permitted_color' in self.observation_space.spaces
        concat_dim = self.trunk_dim + (3 if self.has_permitted_color else 0)

        # Game head (non-zap actions)
        self.game_head = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 non-zap actions
        )

        # Sanction head (zap action)
        self.sanction_head = nn.Sequential(
            nn.Linear(concat_dim, self.sanction_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.sanction_hidden_dim, 1)  # 1 zap action
        )

        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, deterministic=False):
        """Forward pass through concatenation architecture."""
        # Extract trunk features
        h = self.extract_features(obs)  # (batch, trunk_dim)

        # Concatenate permitted_color if available
        if self.has_permitted_color and 'permitted_color' in obs:
            h_concat = torch.cat([h, obs['permitted_color']], dim=1)
        else:
            h_concat = h

        # Action heads
        game_logits = self.game_head(h_concat)  # (batch, 10)
        sanction_logit = self.sanction_head(h_concat)  # (batch, 1)

        # Combine into full action distribution
        full_logits = torch.cat([game_logits, sanction_logit], dim=1)  # (batch, 11)

        # Sample action
        distribution = Categorical(logits=full_logits)
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        # Value estimate
        values = self.value_net(h_concat)  # (batch, 1)

        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs
```

**Usage**:
```python
# For No-FiLM ablation
model = PPO(ConcatTwoHeadPolicy, env, policy_kwargs={...})

# For FiLM ablations
model = PPO(FiLMTwoHeadPolicy, env, policy_kwargs={...})
```

---

## 6. EXPERIMENTAL DESIGN

### Phase 1: Single-Community Ablations (RED)

**Conditions** (3 seeds each):
1. Correct-FiLM (treatment)
2. Null-FiLM (control)
3. Random-FiLM (treatment-random-episodic)
4. No-FiLM-Concat (treatment-concat)

**Total**: 4 conditions × 3 seeds = 12 runs

**Command examples**:
```bash
# Correct-FiLM (baseline)
python train_ppo.py treatment --config configs/treatment.yaml --seed 42

# Null-FiLM (baseline)
python train_ppo.py control --config configs/control.yaml --seed 42

# Random-FiLM (ablation)
python train_ppo.py treatment --config configs/treatment_random.yaml --seed 42
# Where treatment_random.yaml has: env.film_ablation = 'random_episodic'

# No-FiLM-Concat (ablation)
python train_ppo.py treatment --config configs/treatment_concat.yaml --seed 42
# Uses ConcatTwoHeadPolicy instead of FiLMTwoHeadPolicy
```

---

### Phase 2: Multi-Community Ablations (if Phase 1 validates)

**Conditions** (3 seeds each):
1. Correct-FiLM-Multi (treatment-multi)
2. Null-FiLM-Multi (control-multi)
3. Random-FiLM-Multi (treatment-random-multi) ⭐ **Critical test**

**Total**: 3 conditions × 3 seeds = 9 runs

**Expected**: Random-FiLM-Multi should FAIL badly (can't learn distinct policies per community)

---

### Phase 3: Additional Ablations (if needed)

**Permutation test**:
```bash
# Permuted color mapping
python train_ppo.py treatment --config configs/treatment_permuted.yaml
# env.film_ablation = 'permuted'
```

**Per-step random**:
```bash
# Random every step (chaos)
python train_ppo.py treatment --config configs/treatment_random_step.yaml
# env.film_ablation = 'random_step'
```

---

## 7. EXPECTED RESULTS

### Single-Community (Phase 1)

**Expected ranking**:
```
Correct-FiLM > No-FiLM-Concat >> Random-FiLM ≈ Null-FiLM
```

**Interpretation**:
- Correct-FiLM > No-FiLM-Concat: FiLM modulation better than concatenation
- Correct-FiLM >> Random-FiLM: FiLM learns color-specific (mechanism validated)
- Random-FiLM ≈ Null-FiLM: Random signal doesn't help (sanity check)

**Alternative outcomes**:

**If Correct ≈ Random**:
- 🚨 FiLM not using color semantics (broken!)
- Need to investigate why

**If No-FiLM-Concat > Correct-FiLM**:
- Concatenation simpler and better
- Reconsider FiLM architecture

---

### Multi-Community (Phase 2)

**Expected ranking**:
```
Correct-FiLM-Multi >> Null-FiLM-Multi ≈ Random-FiLM-Multi
```

**Critical test**:
- Random-FiLM-Multi should perform VERY poorly
- Cannot learn 3 distinct policies with random conditioning
- **Strong evidence** FiLM learns color-specific modulation

**If Random-FiLM-Multi ≈ Correct-FiLM-Multi**:
- 🚨 Major problem with FiLM implementation!
- Policy not conditioning on signal correctly

---

## 8. METRICS TO TRACK

### Training Metrics (W&B)

**Existing**:
- Mean reward
- Monoculture fraction
- Sanction rates
- FiLM diagnostics (γ/β norms)

**Additional for ablations**:
- **Per-color monoculture** (in multi-community):
  - `episode/RED/monoculture_fraction`
  - `episode/GREEN/monoculture_fraction`
  - `episode/BLUE/monoculture_fraction`
  - Should be ~0.95 for Correct-FiLM-Multi
  - Should be ~0.33 for Random-FiLM-Multi (random planting)

### FiLM Diagnostics

**For Correct-FiLM** (multi-community):
- Plot γ([1,0,0]), γ([0,1,0]), γ([0,0,1]) over training
- Should see divergence (different modulation per color)

**For Random-FiLM** (multi-community):
- Plot γ(random) over training
- Should see no consistent pattern

---

## 9. IMPLEMENTATION CHECKLIST

### Code Changes Needed

- [ ] Create `agents/envs/film_ablation_wrapper.py`
  - Implement all ablation modes (correct, null, random_episodic, random_step, fixed_random, permuted)
  - Add to `sb3_wrapper.py` make_vec_env functions

- [ ] Create `agents/train/concat_policy.py`
  - Implement ConcatTwoHeadPolicy (no FiLM, just concatenation)
  - Add registration in train_ppo.py

- [ ] Update `agents/train/train_ppo.py`
  - Add `--ablation-mode` argument
  - Add policy class selection logic

- [ ] Create config files
  - `configs/treatment_random.yaml` (film_ablation: random_episodic)
  - `configs/treatment_concat.yaml` (use ConcatTwoHeadPolicy)
  - `configs/treatment_permuted.yaml` (film_ablation: permuted)
  - Multi-community variants

- [ ] Update `agents/train/callbacks.py`
  - Add ablation_mode to W&B tags
  - Log per-color metrics for multi-community ablations

### Testing

- [ ] Unit test FiLMAblationWrapper
  - Verify correct mode doesn't modify obs
  - Verify null mode removes permitted_color
  - Verify random modes generate proper distributions
  - Verify permutation mapping

- [ ] Unit test ConcatTwoHeadPolicy
  - Verify architecture dimensions
  - Verify forward pass
  - Verify no FiLM modules exist

- [ ] Integration test
  - Run smoke test with each ablation mode (100k steps)
  - Verify training completes without errors
  - Verify W&B logging works

### Documentation

- [ ] Update README with ablation instructions
- [ ] Update alignment.md with ablation status
- [ ] Create analysis notebook for comparing ablations

---

## 10. TIMELINE

**After main Phase 4/5 experiments complete**:

1. Week 1: Implementation
   - Create wrapper and policy classes
   - Create configs
   - Test locally

2. Week 2: Single-community ablations
   - Run 4 conditions × 3 seeds (12 runs)
   - Analyze results
   - Decide if mechanism validated

3. Week 3: Multi-community ablations (if validated)
   - Run 3 conditions × 3 seeds (9 runs)
   - Analyze results
   - Write up findings

**Total**: ~3 weeks, 21 training runs

---

## 11. REFERENCES

### Key Files
- `agents/train/film_policy.py` - Current FiLM implementation
- `agents/train/train_ppo.py` - Training script
- `agents/envs/sb3_wrapper.py` - Environment creation
- `agents/envs/normative_observation_filter.py` - Current observation filtering

### Related Docs
- `alignment.md` - Main progress tracking
- `UNDERSTANDING.md` - Architecture overview

### FiLM Papers
- Perez et al. (2018) "FiLM: Visual Reasoning with a General Conditioning Layer"
- Dumoulin et al. (2018) "Feature-wise transformations"

---

## SUMMARY

**Purpose**: Validate that FiLM learns color-specific modulation (not just benefiting from any signal)

**Critical Test**: Correct-FiLM vs Random-FiLM
- If Correct >> Random → FiLM works correctly ✓
- If Correct ≈ Random → FiLM is broken ✗

**Implementation**: Environment wrapper to modify permitted_color + separate policy for concatenation ablation

**Timeline**: After main experiments, ~3 weeks for full ablation suite

**Priority**: High (validates mechanism), but deferred until main hypothesis confirmed
