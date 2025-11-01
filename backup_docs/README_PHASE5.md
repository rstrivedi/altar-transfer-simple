# Phase 5 — Multi-Community Distributional Competence

**Added by RST**

Phase 5 extends Phase 4's single-agent RL training to **multi-community distributional competence**. Instead of training separate policies for each community (RED/GREEN/BLUE), Phase 5 trains a **single policy** that learns to enforce **all three** institutional rules by **randomly sampling** the community at each episode reset.

---

## Overview

**Goal**: Train a single FiLM policy that exhibits distributional competence across all three communities:
> **Single policy π generalizes to enforce any posted rule θ ∈ {RED, GREEN, BLUE}**

**Key Innovation**: **Independent per-worker random sampling** ensures unbiased mixture gradient optimization without schedule confounding.

### What Phase 5 Delivers

1. **Multi-community mode** (`sb3_wrapper.py`): Independent per-worker community sampling
2. **Community tagging** (`schema.py`, `recorder.py`): Track which community each episode belongs to
3. **Distributional metrics** (`aggregators.py`): Per-color and distributional (avg/worst/best) metrics
4. **Distributional evaluation** (`eval_harness.py`): Evaluate policy across all three communities
5. **Multi-community training** (`train_ppo.py`): `--multi-community` flag for Phase 5 mode
6. **Phase 5 callbacks** (`callbacks.py`): Per-color W&B logging and faceted plots
7. **Phase 5 configs** (`configs/*_multi.yaml`): Treatment, control, smoke test for multi-community
8. **Acceptance tests** (`phase5_tests.py`): Tests D1-D6
9. **Documentation**: This README

---

## Architecture

### Independent Sampling Design

**Core Principle**: Each worker independently samples community at each episode reset using its own RNG.

```
Training run (one seed):
  seed = 42

  Worker 0: RandomState(42 + 0) → samples RED, GREEN, RED, BLUE, ...
  Worker 1: RandomState(42 + 1) → samples BLUE, RED, GREEN, RED, ...
  Worker 2: RandomState(42 + 2) → samples GREEN, BLUE, BLUE, RED, ...
  ...
  Worker 31: RandomState(42 + 31) → samples RED, RED, GREEN, BLUE, ...

  NO GLOBAL COORDINATION (each worker samples independently)
```

**Why independent sampling?**
1. **Objective fidelity**: Optimizes E_θ~μ[J(π|θ)] where μ = {1/3 RED, 1/3 GREEN, 1/3 BLUE}
2. **No schedule confounding**: Agent MUST read PERMITTED_COLOR (can't memorize temporal pattern)
3. **No catastrophic forgetting**: Gradients naturally stabilize without ping-pong updates
4. **Stationary data distribution**: On-policy learning assumption holds
5. **Throughput**: No synchronization overhead between workers

**Law of Large Numbers**:
- 32 workers × ~15 episodes per rollout = ~480 episodes per training iteration
- Expected distribution: ~160 RED, ~160 GREEN, ~160 BLUE
- Balance check: max/min ratio ≈ 1.0

### Community Tagging

Each episode is tagged with:
- `community_tag`: 'RED', 'GREEN', or 'BLUE' (string)
- `community_idx`: 1, 2, or 3 (int)

Tags flow through:
1. `sb3_wrapper.py`: Sample community, set `env.config['permitted_color_index']`
2. `recorder.py`: Receive community tags, attach to each `StepMetrics`
3. `aggregators.py`: Group episodes by community_tag, compute per-color metrics
4. `eval_harness.py`: Tag episodes during distributional evaluation

### Distributional Metrics

**Per-community metrics** (RED, GREEN, BLUE):
- Value gap (ΔV): mean, median, std
- Sanction regret (SR): mean, median
- Compliance%, violations/1k
- Correct sanction rate, immunity rate

**Distributional summary**:
- `avg_value_gap`: E_θ~μ[ΔV(π|θ)] (average across all communities)
- `worst_value_gap`: max_θ(ΔV) (worst-case community)
- `best_value_gap`: min_θ(ΔV) (best-case community)
- `worst_community`: 'RED'/'GREEN'/'BLUE'
- `best_community`: 'RED'/'GREEN'/'BLUE'
- `balance_check_ratio`: max(counts)/min(counts) (should be ≈1.0)

**Interpretation**:
- **avg_value_gap < Phase 4 ΔV**: Policy generalizes across communities
- **worst_value_gap ≈ avg_value_gap**: No severe underperformance on any community
- **balance_check_ratio ≈ 1.0**: Sampling is balanced (no bias)

---

## File Structure

```
agents/
├── envs/
│   └── sb3_wrapper.py              # Extended with multi_community_mode
├── metrics/
│   ├── schema.py                   # Added community_tag, DistributionalRunMetrics
│   ├── recorder.py                 # Extended with community tracking
│   ├── aggregators.py              # Added aggregate_distributional_metrics()
│   └── eval_harness.py             # Added run_distributional_evaluation()
├── train/
│   ├── train_ppo.py                # Added --multi-community flag
│   ├── callbacks.py                # Extended for per-color logging
│   └── configs/
│       ├── treatment_multi.yaml    # Phase 5 treatment config
│       ├── control_multi.yaml      # Phase 5 control config
│       └── smoke_test_multi.yaml   # Phase 5 smoke test
└── tests/
    └── phase5_tests.py             # Acceptance tests D1-D6
```

**Modified files** (~800 lines added):
- `sb3_wrapper.py` (+120 lines): Multi-community mode, independent sampling
- `schema.py` (+50 lines): Community tags, DistributionalRunMetrics
- `recorder.py` (+15 lines): Community tag parameters
- `aggregators.py` (+140 lines): Distributional aggregation
- `eval_harness.py` (+180 lines): Distributional evaluation
- `train_ppo.py` (+30 lines): Multi-community flag
- `callbacks.py` (+180 lines): Per-color logging
- `configs/*_multi.yaml` (+210 lines): Phase 5 configs
- `phase5_tests.py` (+386 lines): Acceptance tests
- `README_PHASE5.md` (this file)

---

## Usage

### Training

**Treatment arm** (Phase 5 multi-community):
```bash
python agents/train/train_ppo.py treatment \
    --multi-community \
    --config agents/train/configs/treatment_multi.yaml \
    --output-dir ./outputs/treatment_multi
```

**Control arm** (Phase 5 multi-community):
```bash
python agents/train/train_ppo.py control \
    --multi-community \
    --config agents/train/configs/control_multi.yaml \
    --output-dir ./outputs/control_multi
```

**Smoke test** (quick validation, 200k steps):
```bash
python agents/train/train_ppo.py treatment \
    --multi-community \
    --config agents/train/configs/smoke_test_multi.yaml
```

**Custom seed and timesteps**:
```bash
python agents/train/train_ppo.py treatment \
    --multi-community \
    --total-timesteps 10000000 \
    --seed 12345
```

### Evaluation

**Distributional evaluation** (using Phase 5 harness):
```bash
python -c "
from agents.metrics.eval_harness import run_distributional_evaluation
from agents.train.eval_cli import load_policy_from_checkpoint

# Load policy
policy_fn = load_policy_from_checkpoint(
    'checkpoints/treatment_multi/ppo_treatment_step_5000000.zip',
    arm='treatment'
)

# Run distributional evaluation
results = run_distributional_evaluation(
    ego_policy=policy_fn,
    config={'permitted_color_index': 1, 'alpha': 0.5, 'beta': 0.5, 'c': 0.2},
    num_episodes_per_community=20,
    seed=42,
)

# Print results
print('Baseline:', results['baseline'])
print('Treatment:', results['treatment'])
print('Control:', results['control'])
"
```

**Note**: Full eval CLI integration is TODO. For now, use the Python snippet above or implement a dedicated `eval_cli_multi.py`.

### Testing

**Run all Phase 5 tests**:
```bash
pytest agents/tests/phase5_tests.py -v
```

**Run specific test**:
```bash
pytest agents/tests/phase5_tests.py::test_d1_independent_sampling -v
pytest agents/tests/phase5_tests.py::test_d3_balance_check -v
```

**Include slow tests**:
```bash
pytest agents/tests/phase5_tests.py -v --run-slow
```

---

## Hyperparameters

### Phase 5 Training (vs Phase 4)

| Parameter          | Phase 4 (Single) | Phase 5 (Multi) | Rationale                                |
|--------------------|------------------|-----------------|------------------------------------------|
| `total_timesteps`  | 5M               | 10M             | 2x longer for thorough coverage          |
| `n_envs`           | 16               | 32              | More workers for better balance          |
| `batch_size`       | 2048             | 4096            | Larger batch (32 × 256 / 2)              |
| `eval_freq`        | 100k             | 200k            | Distributional eval is slower            |
| `save_freq`        | 50k              | 100k            | Less frequent checkpoints                |

**All other hyperparameters identical to Phase 4**:
- Learning rate: 0.0003
- n_steps: 256
- n_epochs: 10
- gamma: 0.995
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01 (head-wise: game=0.01, sanction=0.02)

### Environment Economics (same as Phase 4)

```yaml
env:
  permitted_color_index: 1  # Placeholder (sampled at runtime in Phase 5)
  alpha: 0.5
  beta: 0.5
  c: 0.2
  immunity_cooldown: 200
  startup_grey_grace: 25
  episode_timesteps: 2000
```

---

## Key Design Decisions

### 1. Independent vs Synchronized Sampling

**❌ Rejected: Synchronized sampling** (all workers use same community per episode)
- Causes catastrophic forgetting (ping-pong between communities)
- Creates schedule confounding (agent memorizes temporal pattern)
- Violates mixture objective optimization
- Non-stationary data distribution breaks on-policy learning

**✅ Adopted: Independent sampling** (each worker samples independently)
- Optimizes correct mixture objective: E_θ~μ[J(π|θ)]
- No schedule confounding (agent must read PERMITTED_COLOR)
- Stable on-policy gradients
- Law of Large Numbers ensures balance

### 2. One Seed Per Experiment

Training run has **one base seed**:
- Workers get derived seeds: `seed + rank`
- This provides environment randomness diversity
- Still counts as "one experiment" (deterministic given base seed)
- Same convention as Phase 4

### 3. Community Tagging (Non-Invasive)

Community tags added to schema without breaking Phase 4:
- `community_tag` and `community_idx` are Optional fields
- Phase 4 code ignores them (None values)
- Phase 5 code populates them
- Backward compatible

### 4. Distributional Evaluation Protocol

Phase 5 evaluation runs **60 episodes** (20 per community):
1. Generate 60 seeds from base seed (deterministic)
2. Split: 20 for RED, 20 for GREEN, 20 for BLUE
3. For each community:
   - Run baseline (resident-in-ego-slot)
   - Run treatment (ego with PERMITTED_COLOR)
   - Run control (ego without PERMITTED_COLOR)
4. Aggregate per-community metrics
5. Compute distributional summary

**Contrast with Phase 4**: Phase 4 runs 20 episodes for single community

### 5. Architectural Parity Maintained

Phase 5 uses **same FiLM two-head policy** as Phase 4:
- No architectural changes
- Control still uses null token (zeros)
- Treatment receives community-specific PERMITTED_COLOR per episode
- A/B difference remains institutional signal only

---

## Acceptance Tests

### D1 - Independent sampling
✅ Each worker samples community independently with different RNG seeds

### D2 - Community tagging
✅ Episodes correctly tagged with community_tag ('RED'/'GREEN'/'BLUE') and community_idx (1/2/3)

### D3 - Balance check
✅ Over 90 episodes, ~1:1:1 ratio across communities (Law of Large Numbers)

### D4 - Distributional metrics
✅ Per-color and distributional aggregation (avg/worst/best) works correctly

### D5 - Multi-community training
⏭️ Skipped by default (requires training run)
Run manually: `python agents/train/train_ppo.py treatment --multi-community --config agents/train/configs/smoke_test_multi.yaml`

### D6 - Distributional evaluation
✅ `run_distributional_evaluation()` works correctly across all communities

---

## W&B Logging

Phase 5 extends Phase 4's W&B logging with **per-color metrics**:

**Per-community logs** (if multi-community mode):
```
eval/red/value_gap_mean
eval/red/sanction_regret_mean
eval/red/correct_sanction_rate
eval/green/value_gap_mean
eval/green/sanction_regret_mean
eval/green/correct_sanction_rate
eval/blue/value_gap_mean
eval/blue/sanction_regret_mean
eval/blue/correct_sanction_rate
```

**Distributional summary logs**:
```
eval/dist/avg_value_gap          # E_θ~μ[ΔV]
eval/dist/worst_value_gap        # max_θ(ΔV)
eval/dist/best_value_gap         # min_θ(ΔV)
eval/dist/worst_community        # 'RED'/'GREEN'/'BLUE'
eval/dist/best_community         # 'RED'/'GREEN'/'BLUE'
eval/dist/balance_check          # max/min ratio
```

**Setup** (same as Phase 4):
```bash
export WANDB_API_KEY=your_key_here
```

**Visualizations**:
- Faceted plots: ΔV over time for RED/GREEN/BLUE
- Distributional summary: avg/worst/best ΔV trends
- Balance check: verify ~1:1:1 sampling ratio

---

## Troubleshooting

### Imbalanced Sampling

**Problem**: Balance check ratio > 1.5 (e.g., 200 RED, 100 GREEN, 180 BLUE)

**Diagnosis**:
- Law of Large Numbers needs more episodes
- Bad luck (statistical fluctuation)

**Solution**:
- Increase `n_envs` or training length
- Check RNG seeds (should be `seed + rank`)
- Verify no global synchronization

### Poor Performance on One Community

**Problem**: avg_value_gap = 2.0, but worst_value_gap = 5.0 (one community failing)

**Diagnosis**:
- Policy not generalizing to that community
- May need longer training
- Check if that community is undersampled

**Solution**:
- Train longer (balance improves with more data)
- Check balance_check_ratio
- Visualize per-color learning curves in W&B

### Catastrophic Forgetting Symptoms

**Problem**: Learning curves oscillate wildly, no convergence

**Diagnosis**:
- If using synchronized sampling (shouldn't be!)
- Or learning rate too high

**Solution**:
- Verify independent sampling (check `multi_community_mode=True`)
- Reduce learning rate
- Increase batch size

### Eval Results Don't Match Training

**Problem**: W&B shows good performance, but distributional eval is poor

**Diagnosis**:
- Training metrics may be biased (alpha bonus)
- Eval uses α=0 (stripped)

**Solution**:
- Compare training R_eval (stripped) vs eval ΔV
- Use `run_distributional_evaluation()` for accurate A/B test
- Check VecNormalize stats are loaded correctly

---

## Integration with Phases 1-4

Phase 5 builds on all previous phases:

**Phase 1** (physics):
- Identical substrate physics
- No changes to Lua environment

**Phase 2** (residents):
- Same 15 scripted residents
- Enforce across all three communities

**Phase 3** (metrics & eval):
- Extended eval harness: `run_distributional_evaluation()`
- Per-color metrics computed by grouping episodes

**Phase 4** (RL training):
- Same FiLM two-head policy
- Extended wrapper for multi-community mode
- Extended callbacks for per-color logging

---

## Future Work & TODOs

### Immediate TODOs

**1. Eval CLI for Phase 5** (`eval_cli_multi.py`):
- Create dedicated CLI for distributional evaluation
- Load checkpoint → run distributional eval → print summary
- Similar to `eval_cli.py` but calls `run_distributional_evaluation()`

**2. Curriculum Sampling**:
- Start with single community (like Phase 4)
- Gradually introduce diversity (e.g., 80% RED, 10% GREEN, 10% BLUE)
- Anneal to uniform (33/33/33) over training
- May improve learning stability

**3. Per-Community Checkpointing**:
- Save separate checkpoints for best performance on each community
- Useful for debugging which community is hardest

**4. Ablation: Fixed Community Order**:
- Compare independent sampling vs blocked randomization
- Expected: independent > blocked (no schedule confounding)

### Future Enhancements

**1. Adaptive Sampling**:
- Oversample worst-performing community
- Based on recent ΔV or SR metrics
- May improve worst-case performance

**2. Multi-Community Ablations**:
- **Info-only multi**: α=β=0, institutional signal only
- **Shaping-only multi**: Mask PERMITTED_COLOR, keep α/β/c
- Expected hierarchy: info-only < shaping-only < treatment

**3. Transfer Learning**:
- Train on 2 communities, test on 3rd (zero-shot)
- Measures true generalization vs memorization

**4. Explicit Worst-Case Optimization**:
- Minimize max_θ(ΔV) instead of E_θ[ΔV]
- Robust optimization objective
- May improve fairness across communities

---

## Expected Results

### Hypothesis H5: Distributional Competence

**If treatment > control (Phase 4)**, then Phase 5 predicts:

1. **avg_value_gap(treatment_multi) ≤ value_gap(treatment_single)**
   - Single policy performs at least as well as specialized policies

2. **worst_value_gap(treatment_multi) ≈ avg_value_gap(treatment_multi)**
   - No catastrophic failure on any community

3. **avg_value_gap(treatment_multi) < avg_value_gap(control_multi)**
   - Institutional signal helps across all communities

4. **balance_check_ratio ≈ 1.0**
   - Sampling is balanced (no bias)

### Success Criteria

**Strong success**:
- avg_value_gap < 1.0 (all communities)
- worst_value_gap < 1.5
- treatment_multi > control_multi (A/B difference holds)

**Acceptable success**:
- avg_value_gap < 2.0
- worst_value_gap < 3.0
- treatment_multi > control_multi

**Failure** (investigate):
- avg_value_gap > 3.0
- worst_value_gap >> avg_value_gap (one community failing)
- treatment_multi ≈ control_multi (A/B difference lost)

---

## Notes for Next Session

### What Works
✅ Independent sampling: Each worker samples community independently
✅ Community tagging: Episodes correctly tagged with RED/GREEN/BLUE
✅ Balance check: Law of Large Numbers produces ~1:1:1 ratio
✅ Distributional metrics: Per-color and summary aggregation
✅ Distributional evaluation: Harness works across all communities
✅ Multi-community training: `--multi-community` flag works
✅ Phase 5 callbacks: Per-color W&B logging implemented
✅ Acceptance tests D1-D6: All pass (D5 manual)

### What Needs Testing
⚠️ Long training runs (10M steps)
⚠️ A/B comparison: treatment_multi vs control_multi
⚠️ Worst-case performance: Is worst_community significantly worse?
⚠️ Transfer: Does policy generalize to unseen test episodes?
⚠️ W&B faceted plots: Do per-color learning curves look good?

### Known Limitations
- Eval CLI not yet implemented for Phase 5 (use Python snippet)
- No curriculum sampling (immediate uniform distribution)
- No adaptive sampling (oversample worst community)
- No explicit worst-case optimization

### Next Steps (Phase 6?)

1. **Run full training**:
   - treatment_multi: 10M steps, 32 workers
   - control_multi: 10M steps, 32 workers
   - Compare: avg_ΔV, worst_ΔV, per-color metrics

2. **Analyze results**:
   - If treatment_multi > control_multi: Validates distributional competence
   - If worst_community significantly worse: Investigate (undersampling? harder task?)
   - If balance_check_ratio > 1.5: Increase n_envs or training length

3. **Ablations**:
   - Fixed community order (blocked randomization)
   - Curriculum sampling (gradual diversity)
   - Transfer learning (train on 2, test on 3rd)

4. **Extensions**:
   - Eval CLI for Phase 5
   - Adaptive sampling
   - Explicit worst-case optimization
   - Multi-seed statistical analysis

---

## Summary

Phase 5 extends Phase 4's single-community RL training to **multi-community distributional competence**. The core innovation is **independent per-worker random sampling**, which ensures unbiased mixture gradient optimization without schedule confounding. The policy learns to generalize across all three communities (RED/GREEN/BLUE) by reading the PERMITTED_COLOR signal and adapting its enforcement accordingly.

**Total additions**: ~1,300 lines across 10 files
**Integration**: Phases 1 (physics), 2 (residents), 3 (eval harness), 4 (RL training)
**Ready for**: Long training runs and distributional A/B comparison

---

*Phase 5 implemented by RST, 2025*
