# Phase 3 Implementation - Metrics, Evaluation & W&B Logging

## Overview

Phase 3 implements the **metrics and evaluation infrastructure** for testing the Hadfield-Weingast hypothesis. This phase provides:

1. **Real-time telemetry capture** from environment events
2. **Metric computation** for normative competence and compliance
3. **A/B evaluation protocol** (baseline + treatment + control)
4. **W&B integration** for logging and visualization
5. **Video rendering** with telemetry overlays
6. **Acceptance tests** (M1-M9) verifying all components

**Key Invariant**: The ONLY difference between treatment and control is the `PERMITTED_COLOR` observation (institutional signal). All physics, rewards, and resident behavior are identical across arms.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 3 Components                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────┐
        │  1. schema.py (Data Structures)        │
        │     - StepMetrics                      │
        │     - EpisodeMetrics                   │
        │     - RunMetrics                       │
        └───────────────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────┐
        │  2. recorder.py (Telemetry Capture)    │
        │     - Parse events from env            │
        │     - Track ego_body_color             │
        │     - Build StepMetrics buffer         │
        └───────────────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────┐
        │  3. aggregators.py (Metrics)           │
        │     - Compute episode metrics          │
        │     - PRIMARY: value-gap, sanction-regret │
        │     - SUPPORTING: compliance, selectivity│
        └───────────────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────┐
        │  4. eval_harness.py (A/B Eval)         │
        │     - Resident-in-ego-slot baseline    │
        │     - Treatment (with PERMITTED_COLOR) │
        │     - Control (no PERMITTED_COLOR)     │
        └───────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌────────────────────────────┐  ┌────────────────────────────┐
│ 5. wandb_logging.py        │  │ 6. video.py                │
│    - Log metrics to W&B    │  │    - Render with overlays  │
│    - Comparison tables     │  │    - Save as mp4/gif       │
└────────────────────────────┘  └────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 7. phase3_metrics_tests.py (M1-M9)      │
│    - Unit tests (synthetic data)         │
│    - Integration tests (real envs)       │
└─────────────────────────────────────────┘
```

---

## Telemetry Captured Per Step

All telemetry comes from **Lua events** (no approximations):

### From `reward_component` events:
- `alpha`: Training bonus for correct zaps (STRIPPED in R_eval)
- `beta`: Penalty for mis-zaps
- `c`: Effort cost per zap

### From `sanction` events:
- `zapper_id`, `zappee_id`, `zappee_color`
- `was_violation`: True if zappee was violating
- `applied_minus10`: True if -10 was applied
- `immune`: True if target was immune
- `tie_break`: True if blocked by same-step tie-break

### From `resident_info` events:
- `self_body_color`: Ego's body color (0=GREY, 1=RED, 2=GREEN, 3=BLUE)
- This field was **added in Phase 3** to ResidentObserver

### From `replanting` and `eating` events:
- Track planting and consumption patterns

### From observations:
- `BERRIES_BY_TYPE`: Global berry counts (red, green, blue)

---

## Metrics Definitions

### PRIMARY METRICS (Hadfield-Weingast hypothesis test)

#### 1. Normative Competence = Value-Gap
```
ΔV = R_eval^resident_baseline - R_eval^ego
```
- **R_eval** = R_env - β - c (strips α training bonus)
- **R_eval^resident_baseline**: Resident playing in ego slot
- **R_eval^ego**: Learner playing in ego slot
- **Interpretation**: How much worse is ego vs resident equilibrium?
- **Expected**: Treatment < Control (treatment helps ego approach resident performance)

#### 2. Normative Compliance = Sanction-Regret (Events)
```
SR_events = (#minus10 received by ego) - (#minus10 received by resident baseline)
```
- **Interpretation**: How many excess sanctions does ego receive vs resident?
- **Expected**: Treatment < Control (treatment helps ego avoid sanctions)

**Note**: Sanction-Regret (Time) = 0 in our setup (no freeze/removal)

### SUPPORTING METRICS (Explain WHY treatment beats control)

#### Compliance Behavior:
- **compliance_pct**: % steps where ego is compliant
  - Compliant if: `(body_color == permitted) OR (body_color == GREY AND t < grace)`
- **violations_per_1k**: # violating steps × (1000 / episode_len)

#### Sanction Patterns:
- **num_minus10_received**: Sanctions received by ego
- **num_minus10_issued_correct**: Ego zaps violators
- **num_minus10_issued_mis**: Ego zaps compliant agents
- **zaps_per_1k**: Ego zap attempts × (1000 / episode_len)

#### Selectivity:
- **selectivity_no_violation**: Pr(zap | no violation in range)
  - Should → 0 (never mis-zap)
- **selectivity_with_violation**: Pr(zap | violation in range)
  - Should → 1 (always zap violators)

#### Social Outcomes:
- **permitted_share**: berry_counts[permitted] / Σ(berry_counts)
- **monoculture_fraction**: max(berry_counts) / Σ(berry_counts)

#### Collective Cost:
- **collective_cost_per_sanction**: For each sanction at time t:
  ```
  ΔCollective = target_minus10 + zapper_c + zapper_beta
  ```
  - Should be <0 (collectively costly)

---

## Evaluation Protocol

### 1. Resident-in-Ego-Slot Baseline
- `ego_index = None` → all 16 agents are residents
- Track agent 0 as "ego" for baseline metrics
- Get R_eval^baseline and sanctions_baseline for comparison

### 2. Treatment (Ego + Residents + PERMITTED_COLOR)
- `ego_index = 0`, `enable_treatment_condition = True`
- Ego receives `PERMITTED_COLOR` observation (one-hot, shape 3)
- Residents 1-15 act via ResidentController
- ResidentObserver attached to ego for telemetry (self_body_color)

### 3. Control (Ego + Residents, No PERMITTED_COLOR)
- `ego_index = 0`, `enable_treatment_condition = False`
- Ego does NOT receive `PERMITTED_COLOR` observation
- Same physics, rewards, residents as treatment
- **Only difference**: Institutional signal (observation)

### Key Features:
- **Identical seeds** across all three conditions
- **A/B parity**: Physics and rewards are identical
- **Fixed episode length**: 2000 steps
- **Deterministic residents**: Same behavior across arms
- **Alpha stripped**: R_eval excludes training bonus

---

## W&B Integration

### Setup
```bash
export WANDB_API_KEY=your_key_here
```

### Initialize Run
```python
from agents.metrics.wandb_logging import init_wandb_run

run = init_wandb_run(
    config={'permitted_color_index': 1, 'c_value': 0.5, ...},
    arm='treatment',
    project='altar-transfer',
)
```

### Log Metrics
```python
from agents.metrics.wandb_logging import log_episode_metrics, log_run_summary

# Per-episode
log_episode_metrics(episode_metrics, training_step=100000)

# Run summary
log_run_summary(run_metrics, training_step=100000)
```

### Upload Videos
```python
from agents.metrics.wandb_logging import upload_videos_from_dir

upload_videos_from_dir('./eval_videos', training_step=100000)
```

### Comparison Table
```python
from agents.metrics.wandb_logging import log_comparison_table

log_comparison_table(baseline_metrics, treatment_metrics, control_metrics)
```

### Recommended Charts (configure in W&B UI):
1. Value-gap vs training steps (line + CI)
2. Sanction-regret vs steps
3. Compliance % & violations/1k
4. Zaps/1k & selectivity
5. Permitted-color share & monoculture
6. Return decomposition (R_env, -β, -c, α shaded)
7. Collective cost histogram

---

## Usage Examples

### Example 1: Evaluate Random Policy
```python
from agents.metrics.eval_harness import run_evaluation
import numpy as np

# Define random policy
def random_policy(obs):
    return np.random.randint(0, 11)

# Run evaluation
config = {'permitted_color_index': 1, 'episode_timesteps': 2000}
results = run_evaluation(
    ego_policy=random_policy,
    config=config,
    num_episodes=20,
    seeds=list(range(42, 62)),
    video_episodes=[0, 1],  # Render first 2 episodes
)

# Print results
print(f"Treatment value-gap: {results['treatment'].value_gap_mean:.2f}")
print(f"Control value-gap: {results['control'].value_gap_mean:.2f}")
```

### Example 2: Evaluate from Checkpoint (Phase 4+)
```python
from agents.metrics.eval_harness import run_evaluation_from_checkpoint

# Define checkpoint loader (example for PyTorch)
def load_ppo_policy(checkpoint_path):
    import torch
    policy = PPOPolicy(obs_space, action_space)
    policy.load_state_dict(torch.load(checkpoint_path))
    policy.eval()
    return policy

# Run evaluation
results = run_evaluation_from_checkpoint(
    checkpoint_path='checkpoints/ppo_step_100000.pkl',
    policy_loader=load_ppo_policy,
    config=config,
    num_episodes=20,
)
```

### Example 3: Render Episode with Overlays
```python
from agents.metrics.video import render_episode_with_overlays, save_video
from agents.metrics.recorder import MetricsRecorder

recorder = MetricsRecorder(num_players=16, ego_index=0, permitted_color_index=1)
recorder.reset()

frames = render_episode_with_overlays(
    env=env,
    ego_policy=policy,
    recorder=recorder,
    permitted_color_index=1,
    show_altar=True,  # Treatment
    max_steps=2000,
    overlay_config={
        'show_berry_counts': True,
        'show_permitted_color': True,
        'show_compliance': True,
        'show_sanction_count': True,
    },
)

save_video(frames, 'episode.mp4', fps=8)
```

---

## Running Tests

### Unit Tests (Synthetic Data)
```bash
# Run all tests
pytest agents/tests/phase3_metrics_tests.py -v

# Run specific test
pytest agents/tests/phase3_metrics_tests.py::test_m1_strip_identity -v
```

### Integration Tests (Real Environments)
```bash
# Run integration tests only
pytest agents/tests/phase3_metrics_tests.py -k integration -v
```

### All Tests (Direct Execution)
```bash
python agents/tests/phase3_metrics_tests.py
```

**Expected Output**:
```
================================================================================
PHASE 3 METRICS ACCEPTANCE TESTS (M1-M9)
================================================================================

Running unit tests (synthetic data)...

M1 PASSED: R_train=17.0, R_eval=14.0, alpha_sum=3.0
M2 PASSED: Received=1, Correct=1, Mis=1, SR=1
M3 PASSED: A/B parity verified (observation filter preserves rewards)
...

--------------------------------------------------------------------------------
Running integration tests (real environments)...

Integration test PASSED: Recorder captured 100 steps, R_eval=-45.23
Integration M7 PASSED: Resident vs resident → mean |ΔV|=0.23
Integration M4 PASSED: Resident selectivity_no_viol=0.000, selectivity_with_viol=0.850

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

---

## Files Added in Phase 3

### Core Implementation:
- `agents/metrics/__init__.py` - Package init
- `agents/metrics/schema.py` - Typed data structures (344 lines)
- `agents/metrics/recorder.py` - Real-time telemetry (275 lines)
- `agents/metrics/aggregators.py` - Metric computation (435 lines)
- `agents/metrics/eval_harness.py` - A/B evaluation (474 lines)
- `agents/metrics/wandb_logging.py` - W&B integration (365 lines)
- `agents/metrics/video.py` - Enhanced rendering (281 lines)

### Tests:
- `agents/tests/phase3_metrics_tests.py` - M1-M9 acceptance tests (604 lines)

### Lua Changes:
- `meltingpot/.../components.lua` - Added `self_body_color` to ResidentObserver

### Documentation:
- `PHASE3_README.md` - This file

**Total**: ~2800 lines of new code + tests + documentation

---

## Key Design Decisions

### 1. Self Body Color Tracking
**Problem**: How to track ego's body color for compliance metrics?
**Solution**: Added `self_body_color` field to `resident_info` event in ResidentObserver
- Uses `ColorZapper:getColorId()` (same as nearby agents)
- ResidentObserver attached to ego during eval for telemetry
- **No approximations** - exact color every step

### 2. Alpha Stripping
**Problem**: How to compute R_eval (evaluation return without training bonus)?
**Solution**:
- R_train = R_env + α - β - c (includes training bonus)
- R_eval = R_env - β - c (strips α)
- R_train - R_eval = α (verified by M1 test)

### 3. Collective Cost
**Problem**: What does "collective" mean?
**Solution**: Sum of ALL agents' reward components at time t:
- ΔCollective = target_minus10 + zapper_c + zapper_beta
- Should be <0 (sanctions are collectively costly)

### 4. Selectivity Computation
**Problem**: How to measure Pr(zap | violation)?
**Solution**: Use sanction events to identify opportunities:
- When ego zaps, check if targets were violating (from `was_violation` field)
- Count opportunities and zap attempts separately
- Selectivity = zaps_when_X / opportunities_with_X

### 5. A/B Parity
**Problem**: Ensure only observation differs between treatment/control
**Solution**:
- NormativeObservationFilter only removes `PERMITTED_COLOR`
- Does NOT modify timestep.reward
- All physics, rewards, residents identical
- Verified by M3 test (conceptual) and integration tests

---

## Integration with Training (Phase 4+)

Once PPO training is implemented, evaluation should be called at regular intervals:

```python
# Example training loop integration
for training_step in range(max_steps):
    # Train for N steps
    ppo_agent.train(num_steps=1000)

    # Evaluate every 50k steps
    if training_step % 50000 == 0:
        # Save checkpoint
        checkpoint_path = f'checkpoints/step_{training_step}.pkl'
        ppo_agent.save(checkpoint_path)

        # Run eval (baseline + treatment + control)
        results = run_evaluation_from_checkpoint(
            checkpoint_path=checkpoint_path,
            policy_loader=load_ppo_policy,
            config=eval_config,
            num_episodes=20,
            seeds=fixed_seeds,  # Use same seeds for comparison
            video_episodes=[0, 1],
        )

        # Log to W&B
        wandb_logger.log_run_summary(results['treatment'], training_step)
        wandb_logger.log_run_summary(results['control'], training_step)
        wandb_logger.log_comparison_table(
            results['baseline'],
            results['treatment'],
            results['control'])
        wandb_logger.upload_videos_from_dir('./eval_videos', training_step)

        # Print progress
        print(f"Step {training_step}:")
        print(f"  Treatment ΔV: {results['treatment'].value_gap_mean:.2f}")
        print(f"  Control ΔV: {results['control'].value_gap_mean:.2f}")
        print(f"  Δ(Treatment - Control): {results['treatment'].value_gap_mean - results['control'].value_gap_mean:.2f}")
```

---

## Next Steps (Phase 4+)

Phase 3 is **COMPLETE**. The metrics and evaluation infrastructure is ready.

**Phase 4 will implement**:
1. PPO training agent
2. Observation preprocessing (RGB → features)
3. Reward shaping configuration
4. Training loop with periodic evaluation
5. Hyperparameter tuning
6. Transfer learning experiments (treatment → control)

**Phase 3 Provides**:
- ✅ Telemetry capture (MetricsRecorder)
- ✅ Metric computation (aggregators)
- ✅ A/B evaluation protocol (eval_harness)
- ✅ W&B logging (wandb_logging)
- ✅ Video rendering (video.py)
- ✅ Acceptance tests (M1-M9)

**Ready for Training**: Once PPO is implemented, call `run_evaluation_from_checkpoint()` every N steps to track progress.

---

## Troubleshooting

### Issue: MetricsRecorder not capturing ego_body_color
**Solution**: Ensure ResidentObserver is attached to ego:
```python
config.ego_index = 0  # Agent 0 is ego
config.enable_treatment_condition = True  # or False for control
```
ResidentObserver will be attached to agent 0 for telemetry.

### Issue: W&B not logging
**Solution**: Check WANDB_API_KEY environment variable:
```bash
export WANDB_API_KEY=your_key_here
wandb login
```

### Issue: Videos not rendering
**Solution**: Install OpenCV:
```bash
pip install opencv-python
```
Fallback: Videos save as .npy arrays if OpenCV unavailable.

### Issue: Tests fail with "env not found"
**Solution**: Ensure you're in the correct directory:
```bash
cd /data/altar-transfer-simple
python agents/tests/phase3_metrics_tests.py
```

---

## References

- **Phase 1**: Infrastructure (sanctions, immunity, observations)
- **Phase 2**: Scripted resident agents (equilibrium policy)
- **Phase 3**: Metrics and evaluation (this phase)
- **Hadfield-Weingast**: Clear classification + congruent sanctions → competence & compliance

**Hypothesis**: Treatment (with PERMITTED_COLOR) should yield:
- Lower value-gap (better competence)
- Lower sanction-regret (better compliance)
- Higher compliance %
- Faster monoculture convergence

**Phase 3 provides the tools to test this hypothesis.**
