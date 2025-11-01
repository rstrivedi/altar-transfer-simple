# Phase 4 — Single-Agent RL Integration & Training

**Added by RST**

Phase 4 implements a complete **single-agent RL training pipeline** using **Stable-Baselines3 PPO** with a custom **FiLM-conditioned two-head policy**. This phase trains an ego agent (agent 0) in the presence of 15 scripted residents (Phase 2) under identical physics (Phase 1), with the **only** A/B difference being the ego's access to the **institutional observation** (PERMITTED_COLOR).

---

## Overview

**Goal**: Test the Hadfield-Weingast hypothesis via RL training:
> **Public classification + decentralized punishment → normative competence & compliance**

**Key Design Principle**: Isolate the causal effect of the institutional signal by keeping **all** other factors identical between treatment and control arms.

### What Phase 4 Delivers

1. **Gymnasium Wrapper** (`sb3_wrapper.py`): Exposes ego as single-agent environment for SB3 PPO
2. **FiLM Two-Head Policy** (`film_policy.py`): Custom policy with institutional conditioning
3. **Training Script** (`train_ppo.py`): End-to-end PPO training with callbacks
4. **Evaluation CLI** (`eval_cli.py`): Offline evaluation using Phase 3 harness
5. **Callbacks** (`callbacks.py`): W&B logging, checkpointing, periodic evaluation
6. **Configs** (`configs/`): Treatment, control, and smoke test configurations
7. **Tests** (`phase4_tests.py`): Acceptance tests T1-T6
8. **Documentation**: This README

---

## Architecture

### Policy Network (FiLM-Conditioned Two Heads)

```
INPUTS:
  - RGB (88, 88, 3) uint8
  - READY_TO_SHOOT (1,) float32
  - TIMESTEP (1,) float32 [normalized: t/episode_len ∈ [0,1]]
  - [Treatment only] PERMITTED_COLOR (3,) float32 one-hot

PROCESSING:
  1. CNN(RGB) → h_visual (256,)
  2. Concat [h_visual, READY_TO_SHOOT, TIMESTEP] → h_raw (258,)
  3. Linear projection: h_raw → h (256,)
  4. [Optional] LSTM(h) → h_lstm (256,)
  5. Global FiLM:
     - γ = Linear_γ(C_onehot) → (256,)  # Init: weights=0, bias=1
     - β = Linear_β(C_onehot) → (256,)  # Init: weights=0, bias=0
     - h̃ = γ ⊙ h + β

POLICY HEADS (from h̃):
  6. Game head: h̃ → MLP → g (10 logits for non-zap actions [0,1,2,3,4,5,6,8,9,10])
  7. Sanction head (with Local FiLM):
     - Base: h̃ → Linear → ℓ_zap^base (scalar)
     - Residual: h̃ → Proj(256→128) → FiLM_s(C_onehot) → SiLU → u^⊤ → scalar
     - s = ℓ_zap^base + scalar_residual
  8. Compose L[11]: L[7] = s, L[non_zap] = g

VALUE HEAD (SHARED, from h̃):
  9. Value head: h̃ → MLP → V(s) (scalar)

OUTPUT:
  - Action distribution: Categorical(logits=L)  # 11-way discrete
  - Value estimate: V(s)  # Single shared critic
```

### FiLM Conditioning

**Treatment arm**:
- `C_onehot` comes from environment (PERMITTED_COLOR observation)
- FiLM modules modulate features based on posted rule

**Control arm**:
- `C_onehot = zeros(3)` (fixed null token, not learnable)
- FiLM modules exist (architectural parity) but receive no institutional signal

**Why FiLM?**
- Parameter-efficient conditional modulation
- Identity initialization (no effect at start)
- Concentrates institutional influence on zapping (sanction head)
- Allows same perceptual features to be reweighted under different rules

### Action Space Mapping

Substrate provides `Discrete(11)` action space:
```
0: NOOP
1: FORWARD
2: BACKWARD
3: STEP_LEFT
4: STEP_RIGHT
5: TURN_LEFT
6: TURN_RIGHT
7: FIRE_ZAP        ← Sanction head (s)
8: FIRE_ONE        ← Game head (g)
9: FIRE_TWO        ← Game head (g)
10: FIRE_THREE     ← Game head (g)
```

Policy outputs 11 logits by composing:
- **Game head** (g): 10 logits for non-zap actions
- **Sanction head** (s): 1 logit for zap action (index 7)

---

## File Structure

```
agents/
├── envs/
│   └── sb3_wrapper.py         # Gymnasium wrapper for SB3 PPO
├── train/
│   ├── film_policy.py         # FiLM-conditioned two-head policy
│   ├── train_ppo.py           # Main training script
│   ├── eval_cli.py            # Offline evaluation CLI
│   ├── callbacks.py           # W&B logging, checkpointing, eval
│   └── configs/
│       ├── treatment.yaml     # Treatment arm config
│       ├── control.yaml       # Control arm config
│       └── smoke_test.yaml    # Quick test config
└── tests/
    └── phase4_tests.py        # Acceptance tests T1-T6
```

**New files (~2,800 lines total)**:
- `sb3_wrapper.py` (465 lines)
- `film_policy.py` (496 lines)
- `train_ppo.py` (343 lines)
- `eval_cli.py` (265 lines)
- `callbacks.py` (293 lines)
- `configs/*.yaml` (210 lines)
- `phase4_tests.py` (280 lines)
- `README_PHASE4.md` (this file)

---

## Usage

### Training

**Treatment arm** (with PERMITTED_COLOR):
```bash
python agents/train/train_ppo.py \
    --arm treatment \
    --config agents/train/configs/treatment.yaml \
    --output-dir ./outputs/treatment
```

**Control arm** (without PERMITTED_COLOR):
```bash
python agents/train/train_ppo.py \
    --arm control \
    --config agents/train/configs/control.yaml \
    --output-dir ./outputs/control
```

**Smoke test** (quick validation, 100k steps):
```bash
python agents/train/train_ppo.py \
    --arm treatment \
    --config agents/train/configs/smoke_test.yaml \
    --total-timesteps 100000
```

**Custom hyperparameters**:
```bash
python agents/train/train_ppo.py \
    --arm treatment \
    --total-timesteps 10000000 \
    --n-envs 32 \
    --seed 12345
```

### Evaluation

**Offline evaluation** (using Phase 3 harness):
```bash
python agents/train/eval_cli.py \
    --checkpoint checkpoints/treatment/ppo_treatment_step_1000000.zip \
    --arm treatment \
    --n-episodes 50 \
    --video-episodes 0,1,2 \
    --output-dir eval_results
```

**Evaluation with custom config**:
```bash
python agents/train/eval_cli.py \
    --checkpoint checkpoints/control/ppo_control_final.zip \
    --arm control \
    --config agents/train/configs/control.yaml \
    --seeds 42,43,44,45,46
```

### Testing

**Run all acceptance tests**:
```bash
pytest agents/tests/phase4_tests.py -v
```

**Run specific test**:
```bash
pytest agents/tests/phase4_tests.py::test_t2_space_parity -v
```

**Include slow tests** (T6 learning smoke):
```bash
pytest agents/tests/phase4_tests.py -v --run-slow
```

---

## Hyperparameters

### Default PPO Hyperparameters (from Phase 4 spec)

```yaml
training:
  total_timesteps: 5000000
  n_envs: 16
  learning_rate: 0.0003
  n_steps: 256
  batch_size: 2048
  n_epochs: 10
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5
```

### Policy Architecture

```yaml
policy:
  trunk_dim: 256
  sanction_hidden_dim: 128
  ent_coef_game: 0.01      # Entropy for game head
  ent_coef_sanction: 0.02  # Entropy for sanction head (higher initially)
  recurrent: false         # Set to true for LSTM variant
```

### Environment Economics (same across arms)

```yaml
env:
  permitted_color_index: 1  # 1=RED, 2=GREEN, 3=BLUE
  alpha: 0.5   # Train-time bonus (STRIPPED in eval)
  beta: 0.5    # Mis-zap penalty
  c: 0.2       # Zap cost
  immunity_cooldown: 200
  startup_grey_grace: 25
  episode_timesteps: 2000
```

### VecNormalize

```yaml
vec_normalize:
  norm_obs: false    # Don't normalize RGB pixels
  norm_reward: true  # Normalize rewards (clip=10.0)
```

---

## Integration with Phase 3

Phase 4 uses Phase 3's evaluation harness for offline A/B testing:

1. **During training**: Checkpoints saved every 50k steps
2. **Offline evaluation**: Load checkpoint → run `eval_harness.run_evaluation()`
3. **Phase 3 protocol**:
   - Baseline: Resident-in-ego-slot (all 16 residents)
   - Treatment: Ego + residents with PERMITTED_COLOR
   - Control: Ego + residents without PERMITTED_COLOR
4. **Metrics reported**: ΔV, SR_events, compliance%, violations/1k, R_eval

**Note**: Phase 3's `run_evaluation()` expects a policy callable:
```python
def ego_policy(obs):
    return action  # integer 0-10
```

The `eval_cli.py` script handles the conversion from SB3 checkpoint to this format.

---

## Acceptance Tests

### T1 - SB3 env check
✅ `check_env()` passes for both treatment and control arms

### T2 - Space parity
✅ Treatment includes `permitted_color` key, control doesn't
✅ Action spaces identical (Discrete(11))

### T3 - Reward identity
✅ `r_train = r_env + alpha - beta - c`
✅ Strip test: `R_eval = R_train - alpha`

### T4 - Residents integration
✅ Residents enforce when ego is idle (NOOPs)
✅ Phase 2 integration verified

### T5 - Tie-break & immunity
✅ At most one -10 per target per step
✅ No second -10 within K steps on same target

### T6 - Learning smoke test
⏭️ Skipped by default (requires 1M training steps)
Run manually with `smoke_test.yaml` config

---

## W&B Logging

**Setup**:
```bash
export WANDB_API_KEY=your_key_here
```

**What gets logged**:
- Standard PPO metrics (reward, episode length, losses)
- VecNormalize stats (if enabled)
- Tensorboard logs auto-synced to W&B

**TODO** (future enhancements):
- Head-wise entropy (game vs sanction)
- FiLM diagnostics (γ, β norms)
- Action distribution (zap rate)
- Compliance metrics during training

---

## Key Design Decisions

### 1. Architectural Parity
Both treatment and control have **identical** FiLM modules. Control uses a **fixed null token** (zeros) instead of receiving PERMITTED_COLOR from the environment. This ensures any performance difference is attributable to **access to the institutional signal**, not to extra model capacity.

### 2. FiLM Initialization
All FiLM modules initialized to **identity** (γ=1, β=0):
- Ensures no effect at start of training
- Policy learns when/how to use institutional signal
- Prevents interference with base feature learning

### 3. Two-Head Split
Separating **game actions** (move/turn/plant) from **sanction action** (zap) allows us to:
- Isolate institutional effect on enforcement
- Apply head-wise entropy annealing
- Measure treatment effects on zapping without entangling navigation

### 4. Single Shared Value Head
One critic V(s) estimates expected return regardless of action:
- Standard for PPO (one MDP, one return)
- Avoids credit-splitting artifacts
- Simplifies advantage computation

### 5. Timestep Observation
Added normalized timestep `t/episode_len ∈ [0,1]` to observations:
- Helps agent learn temporal patterns (grace period, immunity windows)
- Complements LSTM for partial observability
- Simple linear normalization

### 6. Reward Normalization
VecNormalize normalizes **rewards only** (not RGB observations):
- Stabilizes training with varying reward scales
- Clip=10.0 prevents extreme values
- Stats saved with checkpoints for eval consistency

---

## Future Work & TODOs

### Immediate TODOs (marked in code)

**1. Timestep Tracking in Eval** (`eval_cli.py`):
- Currently hardcoded to 0.0
- Should track actual timestep during evaluation
- Options: (A) stateful policy, (B) modify Phase 3 harness, (C) leave as 0.0

**2. Head-wise Metrics in W&B** (`callbacks.py`):
- Log separate entropy for game vs sanction heads
- Log FiLM parameter norms (γ, β)
- Requires accessing policy internals during training

**3. Phase 3 Eval Integration** (`callbacks.py`):
- EvalCallback currently just saves checkpoint
- Should call `run_evaluation_from_checkpoint()` automatically
- Requires policy loader implementation

### Future Enhancements

**1. Entropy Annealing**:
- Implement per-head entropy annealing schedules
- Reduce `ent_coef_sanction` as selectivity improves
- Based on violations/1k or SR_events metrics

**2. Curriculum Learning**:
- Start with β=0 (no mis-zap penalty)
- Gradually enable β after basic coverage achieved
- Alpha annealing after compliance plateaus

**3. LSTM Variant**:
- Currently implemented but not extensively tested
- May improve performance on longer episodes
- Handle partial observability (immunity windows)

**4. Ablation Studies** (from Phase 4 spec):
- **Info-only**: α=β=0, c>0 (pure institutional observation)
- **Shaping-only**: Mask PERMITTED_COLOR, keep α/β/c
- Expected: info-only < shaping-only < treatment

**5. Multi-seed Training**:
- Train multiple seeds per arm for robustness
- Statistical comparison of treatment vs control
- Handle variance in learning curves

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'agents'`

**Solution**: Run from repository root, or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/altar-transfer-simple"
```

### SB3 Check Warnings

**Problem**: `check_env()` warns about observation space

**Solution**: This is expected for Dict observation spaces. As long as tests pass, it's safe to ignore.

### VecNormalize Loading

**Problem**: "VecNormalize stats not found"

**Solution**: VecNormalize is optional. If you didn't train with `norm_reward=true`, you don't need the stats file.

### WANDB_API_KEY Missing

**Problem**: "WARNING: WANDB_API_KEY not set"

**Solution**: W&B logging is optional. Export your key or disable W&B:
```bash
export WANDB_API_KEY=your_key_here
# OR
python agents/train/train_ppo.py --arm treatment --no-wandb
```

### Slow Training

**Problem**: Training very slow (< 100 steps/sec)

**Solution**:
- Reduce `n_envs` (fewer subprocesses)
- Disable telemetry during training (`enable_telemetry=False` in wrapper)
- Use smaller `batch_size` or `n_epochs`
- Check CPU utilization (SubprocVecEnv is CPU-bound)

### Out of Memory

**Problem**: CUDA OOM or system memory exhausted

**Solution**:
- Reduce `n_envs` or `batch_size`
- Use CPU-only training (remove CUDA)
- Close other processes

### Evaluation Mismatch

**Problem**: eval_cli.py results differ from training

**Solution**:
- Check that alpha is stripped in eval (`alpha=0.0` in eval config)
- Verify VecNormalize stats loaded correctly
- Use deterministic policy (`deterministic=True` in eval_cli)

---

## Notes for Next Session

### What Works
✅ Gymnasium wrapper integrates cleanly with SB3
✅ FiLM two-head policy compiles and trains
✅ Treatment/control configs have A/B parity
✅ Phase 3 eval harness integration via eval_cli
✅ Acceptance tests T1-T5 pass

### What Needs Testing
⚠️ Long training runs (5M+ steps)
⚠️ LSTM recurrent variant
⚠️ Multi-seed reproducibility
⚠️ W&B logging and callbacks
⚠️ Eval callback integration with Phase 3

### Known Limitations
- Timestep tracking in eval is TODO (currently 0.0)
- Head-wise metrics not logged to W&B yet
- EvalCallback doesn't call Phase 3 harness yet
- T6 learning smoke test requires manual run

### Next Steps (Phase 5?)
1. Run full training (treatment + control, 5M steps each)
2. Compare results: ΔV, SR_events, compliance%
3. If treatment > control: validates Hadfield-Weingast hypothesis
4. If not: investigate (curriculum? longer training? hyperparams?)
5. Implement ablations (info-only, shaping-only)
6. Multi-seed statistical analysis

---

## Summary

Phase 4 delivers a complete single-agent RL training pipeline for testing the Hadfield-Weingast hypothesis. The core innovation is the **FiLM-conditioned two-head policy** that explicitly conditions on the institutional signal (treatment) or learns without it (control), while maintaining architectural parity.

**Total additions**: ~2,800 lines of code across 9 files
**Integration**: Phases 1 (physics), 2 (residents), 3 (eval harness)
**Ready for**: Long training runs and A/B comparison

---

*Phase 4 implemented by RST, 2025*
