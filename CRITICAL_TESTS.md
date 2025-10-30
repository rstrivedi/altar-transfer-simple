# CRITICAL TESTS - 12-Hour Validation Sprint

**⚠️ READ THIS AFTER CONVERSATION COMPACTION ⚠️**

This document contains ALL tests needed to validate the codebase before launching final 20M-step experiments on 8 GPUs.

**Mission**: Validate learning pipeline + choose optimal hyperparameters in ≤12 hours

**Resources**: 2 GPUs × 32 CPUs = 64 parallel environments available for testing

---

## 📋 Test Execution Checklist

Use this to track progress:

- [ ] **Phase 0**: Environment Sanity Checks (30 min)
- [ ] **Phase 1**: Resident Behavior Validation (2 hours)
- [ ] **Phase 2**: Learning Probe (2 hours)
- [ ] **Phase 3**: Treatment vs Control Comparison (4 hours)
- [ ] **Phase 4**: Design Choices - LSTM vs Feed-Forward (2 hours)
- [ ] **Phase 5**: Final Config Lock (1 hour)

**Total**: 11.5 hours (30 min buffer)

---

## 🔧 Phase 0: Environment Sanity Checks (30 minutes)

**Objective**: Verify environment infrastructure works before testing residents

**WHY FIRST**: If environment is broken, all other tests will fail. Check foundation first.

### Test 0.1: Basic Environment Startup (5 min)

```bash
cd /data/altar-transfer-simple

python -c "
from agents.envs.sb3_wrapper import AllelopathicHarvestGymEnv

print('Testing Treatment arm...')
env_t = AllelopathicHarvestGymEnv(arm='treatment', seed=42, include_timestep=False)
obs_t, _ = env_t.reset()

# Verify observations
assert 'rgb' in obs_t, 'Missing RGB!'
assert 'ready_to_shoot' in obs_t, 'Missing ready_to_shoot!'
assert 'permitted_color' in obs_t, 'Treatment missing permitted_color!'
assert 'timestep' not in obs_t, 'CRITICAL: TIMESTEP should be OFF!'

print('✓ Treatment obs keys:', list(obs_t.keys()))
print('✓ RGB shape:', obs_t['rgb'].shape)
print('✓ permitted_color shape:', obs_t['permitted_color'].shape)

print('\nTesting Control arm...')
env_c = AllelopathicHarvestGymEnv(arm='control', seed=42, include_timestep=False)
obs_c, _ = env_c.reset()

assert 'rgb' in obs_c, 'Missing RGB!'
assert 'ready_to_shoot' in obs_c, 'Missing ready_to_shoot!'
assert 'permitted_color' not in obs_c, 'Control should NOT have permitted_color!'
assert 'timestep' not in obs_c, 'CRITICAL: TIMESTEP should be OFF!'

print('✓ Control obs keys:', list(obs_c.keys()))

# Run 10 steps
print('\nRunning 10 steps each...')
for i in range(10):
    obs_t, r_t, term, trunc, info = env_t.step(env_t.action_space.sample())
    obs_c, r_c, term, trunc, info = env_c.step(env_c.action_space.sample())

print('✓ Both arms run without crash')
print('\n✅ TEST 0.1 PASSED: Environment startup OK')
"
```

**Expected Output**:
```
Testing Treatment arm...
✓ Treatment obs keys: ['rgb', 'ready_to_shoot', 'permitted_color']
✓ RGB shape: (88, 88, 3)
✓ permitted_color shape: (3,)

Testing Control arm...
✓ Control obs keys: ['rgb', 'ready_to_shoot']

Running 10 steps each...
✓ Both arms run without crash

✅ TEST 0.1 PASSED: Environment startup OK
```

**If fails**: Check import errors, missing dependencies, Lua compilation errors.

---

### Test 0.2: Vectorized Environment (5 min)

```bash
python -c "
from agents.envs.sb3_wrapper import make_vec_env_treatment, make_vec_env_control

config = {
    'permitted_color_index': 1,
    'alpha': 0.5,
    'beta': 0.5,
    'c': 0.2,
    'startup_grey_grace': 25,
    'immunity_cooldown': 200,
}

print('Creating vectorized environments (4 envs each)...')

vec_t = make_vec_env_treatment(num_envs=4, config=config, include_timestep=False)
vec_c = make_vec_env_control(num_envs=4, config=config, include_timestep=False)

obs_t = vec_t.reset()
obs_c = vec_c.reset()

print('✓ Treatment vec obs shapes:')
for k, v in obs_t.items():
    print(f'  {k}: {v.shape}')

print('✓ Control vec obs shapes:')
for k, v in obs_c.items():
    print(f'  {k}: {v.shape}')

# Verify shapes
assert obs_t['rgb'].shape == (4, 88, 88, 3), f'Wrong RGB shape: {obs_t[\"rgb\"].shape}'
assert obs_t['ready_to_shoot'].shape == (4, 1), f'Wrong ready_to_shoot shape'
assert obs_t['permitted_color'].shape == (4, 3), f'Wrong permitted_color shape'
assert 'permitted_color' not in obs_c, 'Control should not have permitted_color'

# Step 10 times
print('\nStepping 10 times...')
for _ in range(10):
    vec_t.step([0]*4)
    vec_c.step([0]*4)

print('✓ Vectorized envs work')
print('\n✅ TEST 0.2 PASSED: Vectorized environment OK')
"
```

**Expected Output**:
```
Creating vectorized environments (4 envs each)...
✓ Treatment vec obs shapes:
  rgb: (4, 88, 88, 3)
  ready_to_shoot: (4, 1)
  permitted_color: (4, 3)
✓ Control vec obs shapes:
  rgb: (4, 88, 88, 3)
  ready_to_shoot: (4, 1)

Stepping 10 times...
✓ Vectorized envs work

✅ TEST 0.2 PASSED: Vectorized environment OK
```

**If fails**: Check SubprocVecEnv issues, multiprocessing errors.

---

### Test 0.3: Policy Instantiation (10 min)

```bash
python -c "
from stable_baselines3 import PPO
from agents.train.film_policy import FiLMTwoHeadPolicy
from agents.envs.sb3_wrapper import make_vec_env_treatment

config = {
    'permitted_color_index': 1,
    'alpha': 0.5,
    'beta': 0.5,
    'c': 0.2,
}

print('Creating vectorized environment...')
vec_env = make_vec_env_treatment(num_envs=4, config=config, include_timestep=False)

print('Creating FiLM policy...')
policy_kwargs = {
    'features_extractor_class': None,  # Uses default DictFeaturesExtractor
    'features_extractor_kwargs': {
        'recurrent': False,  # Feed-forward
        'trunk_dim': 256,
    },
    'trunk_dim': 256,
    'sanction_hidden_dim': 128,
    'ent_coef_game': 0.01,
    'ent_coef_sanction': 0.02,
}

model = PPO(
    FiLMTwoHeadPolicy,
    vec_env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=256,
    policy_kwargs=policy_kwargs,
    verbose=1
)

print('✓ Policy created successfully')
print(f'  Policy class: {model.policy.__class__.__name__}')
print(f'  Action space: {model.action_space}')
print(f'  Observation space keys: {list(model.observation_space.keys())}')

# Test one training step
print('\nTesting one training step (256 steps)...')
model.learn(total_timesteps=256, log_interval=1000)  # Suppress output
print('✓ Training step completes without error')

print('\n✅ TEST 0.3 PASSED: Policy instantiation OK')
"
```

**Expected Output**:
```
Creating vectorized environment...
Creating FiLM policy...
✓ Policy created successfully
  Policy class: FiLMTwoHeadPolicy
  Action space: Discrete(11)
  Observation space keys: ['rgb', 'ready_to_shoot', 'permitted_color']

Testing one training step (256 steps)...
✓ Training step completes without error

✅ TEST 0.3 PASSED: Policy instantiation OK
```

**If fails**: Check PyTorch compatibility, FiLM module errors, SB3 version.

---

### Test 0.4: Phase 1 Test Suite (10 min)

```bash
# Run existing Phase 1 tests
pytest agents/tests/test_phase1_acceptance.py -v --tb=short

# Critical tests:
# - test_observation_filter_treatment_keeps_permitted_color
# - test_observation_filter_control_removes_permitted_color
# - test_metrics_logger_collects_reward_components
# - test_metrics_logger_compute_r_eval
```

**Expected**: All Phase 1 tests PASS

**If fails**: Debug observation filter, reward tracking, or telemetry issues.

---

### ✅ Phase 0 Exit Criteria

Before proceeding to Phase 1, verify:

1. ✅ Treatment has `permitted_color`, Control doesn't
2. ✅ TIMESTEP is OFF in both arms
3. ✅ Vectorized envs work (4 envs)
4. ✅ Policy instantiates without error
5. ✅ One training step completes
6. ✅ All Phase 1 tests pass

**If ANY fail**: STOP. Fix environment before testing residents.

---

## 🤖 Phase 1: Resident Behavior Validation (2 hours)

**Objective**: Verify scripted residents enforce rules correctly

**WHY CRITICAL**: Your entire experiment depends on residents:
1. Sanctioning violators (including ego)
2. NEVER mis-zapping compliant agents
3. Achieving monoculture equilibrium

**If residents fail, ego has nothing to learn from → experiment invalid**

---

### Test 1.1: Existing Resident Test Suite (30 min)

```bash
# Run ALL Phase 2 resident tests
pytest agents/tests/test_phase2_residents.py -v -s --tb=short

# Critical tests to verify:
# - test_r1_selectivity: β_events = 0 (NEVER mis-zap compliant)
# - test_r2_coverage: ≥80% violators sanctioned within 10 frames
# - test_r3_no_dogpiling: No sanctions on immune targets
# - test_r4_plant_harvest_purity: 100% plants permitted, ≥95% harvests permitted
# - test_r5_arm_invariance: Identical behavior across treatment/control
# - test_r6_same_step_tie_break: Max single-step penalty = -10 (no double-zaps)
# - test_r7_no_hidden_dependencies: No freeze/removal in code
# - test_r8_monoculture_achievement: ≥85% permitted berries at t=2000
```

**Expected Output**:
```
test_r1_selectivity PASSED
test_r2_coverage PASSED
test_r3_no_dogpiling PASSED
test_r4_plant_harvest_purity PASSED
test_r5_arm_invariance PASSED
test_r6_same_step_tie_break PASSED
test_r7_no_hidden_dependencies PASSED
test_r8_monoculture_achievement PASSED

================================ 8 passed in 120.45s ================================
```

**If ANY test fails**: STOP. Debug residents before proceeding.

---

### Test 1.2: Residents Sanction Violating Ego (30 min)

**Purpose**: Verify residents will sanction ego when it violates

Create file: `agents/tests/test_resident_sanctions_ego.py`

```python
"""Test that residents correctly sanction ego based on compliance status.

These tests are CRITICAL because the entire learning experiment depends on:
1. Residents sanctioning violating ego (provides learning signal)
2. Residents NEVER sanctioning compliant ego (prevents spurious penalties)
"""

import pytest
from agents.envs.sb3_wrapper import AllelopathicHarvestGymEnv
from agents.residents.config import COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_GREY


def test_residents_sanction_violating_ego():
    """Test that residents sanction ego when it violates the rule.

    Setup:
    - Permitted color: RED (index 1)
    - Ego starts GREY, stays GREY past grace period (25 frames)
    - After frame 25, GREY = violation
    - Residents should sanction ego
    """
    env = AllelopathicHarvestGymEnv(
        arm='treatment',
        config={
            'permitted_color_index': COLOR_RED,  # RED is permitted
            'startup_grey_grace': 25,
            'alpha': 0.5,
            'beta': 0.5,
            'c': 0.2,
            'immunity_cooldown': 200,
        },
        seed=42,
        enable_telemetry=True,
        include_timestep=False,
    )

    obs, info = env.reset()

    # Step past grace period (30 frames) while ego stays GREY (NOOP)
    print("\nStepping past grace period (30 frames)...")
    for t in range(30):
        obs, reward, term, trunc, info = env.step(0)  # NOOP

    print("Grace period passed. Ego is now GREY (violating).")

    # Now ego is GREY and past grace → violating
    # Let residents act for 200 frames to give them time to sanction
    sanctions_received = 0
    sanction_frames = []

    print("Running 200 frames, counting sanctions on violating ego...")
    for t in range(200):
        obs, reward, term, trunc, info = env.step(0)  # Ego stays GREY

        # Check if ego received sanction (reward = -10)
        # Note: reward includes other components (berries, costs), so check for -10 spike
        if reward <= -9.0:  # Account for floating point
            sanctions_received += 1
            sanction_frames.append(t + 30)  # +30 because we already stepped 30

    print(f"\nSanctions received by violating ego: {sanctions_received}")
    print(f"Sanction frames: {sanction_frames[:10]}..." if len(sanction_frames) > 10 else f"Sanction frames: {sanction_frames}")

    # Ego should receive at least one sanction
    assert sanctions_received > 0, \
        f"CRITICAL FAILURE: Residents did NOT sanction violating ego (GREY after grace)! " \
        f"Received {sanctions_received} sanctions in 200 frames. " \
        f"This means ego cannot learn from enforcement."

    print(f"✓ Residents correctly sanctioned violating ego {sanctions_received} times")
    print("✅ TEST PASSED")


def test_residents_never_sanction_compliant_ego():
    """Test that residents NEVER sanction ego when it complies.

    Setup:
    - Permitted color: RED (index 1)
    - Ego plants RED immediately (compliant)
    - Residents should NEVER sanction ego
    """
    env = AllelopathicHarvestGymEnv(
        arm='treatment',
        config={
            'permitted_color_index': COLOR_RED,  # RED is permitted
            'startup_grey_grace': 25,
            'alpha': 0.5,
            'beta': 0.5,
            'c': 0.2,
            'immunity_cooldown': 200,
        },
        seed=42,
        enable_telemetry=True,
        include_timestep=False,
    )

    obs, info = env.reset()

    print("\nWaiting for ego to be ready to plant...")
    # Step until ready to shoot
    for t in range(20):
        obs, reward, term, trunc, info = env.step(0)  # NOOP
        if obs['ready_to_shoot'][0] > 0.5:
            print(f"Ready at frame {t}")
            break

    # Plant RED (action 8) - this makes ego compliant
    print("Planting RED (compliant action)...")
    obs, reward, term, trunc, info = env.step(8)

    # Now ego should be RED → compliant
    # Let residents act for 500 frames
    sanctions_received = 0
    sanction_frames = []

    print("Running 500 frames, ego stays RED (compliant)...")
    for t in range(500):
        obs, reward, term, trunc, info = env.step(0)  # NOOP (stay RED)

        # Check if ego received sanction
        if reward <= -9.0:
            sanctions_received += 1
            sanction_frames.append(t)
            print(f"  WARNING: Sanction at t={t}, reward={reward:.2f}")

    print(f"\nSanctions received by compliant ego (RED): {sanctions_received}")
    if sanction_frames:
        print(f"Sanction frames: {sanction_frames}")

    # Ego should NEVER be sanctioned when compliant
    assert sanctions_received == 0, \
        f"CRITICAL FAILURE: Residents MIS-ZAPPED compliant ego! " \
        f"Received {sanctions_received} sanctions in 500 frames at frames {sanction_frames}. " \
        f"This breaks the entire experimental design (residents must have perfect selectivity)."

    print("✓ Residents NEVER sanctioned compliant ego")
    print("✅ TEST PASSED")


def test_residents_sanction_after_color_change():
    """Test residents sanction ego when it changes from compliant to violating.

    Setup:
    - Permitted: RED
    - Ego plants RED (compliant) → then plants BLUE (violation)
    - Residents should:
      1. NOT sanction while RED
      2. Sanction after changing to BLUE
    """
    env = AllelopathicHarvestGymEnv(
        arm='treatment',
        config={
            'permitted_color_index': COLOR_RED,
            'startup_grey_grace': 25,
            'alpha': 0.5,
            'beta': 0.5,
            'c': 0.2,
            'immunity_cooldown': 200,
        },
        seed=42,
        enable_telemetry=True,
        include_timestep=False,
    )

    obs, info = env.reset()

    print("\nPhase 1: Waiting to plant RED (compliant)...")
    # Step until ready to plant
    for t in range(20):
        obs, reward, term, trunc, info = env.step(0)
        if obs['ready_to_shoot'][0] > 0.5:
            break

    # Plant RED (compliant)
    print("Planting RED (compliant)...")
    obs, reward, term, trunc, info = env.step(8)

    # Stay compliant for 100 frames
    sanctions_while_compliant = 0
    print("Running 100 frames while compliant (RED)...")
    for t in range(100):
        obs, reward, term, trunc, info = env.step(0)
        if reward <= -9.0:
            sanctions_while_compliant += 1
            print(f"  WARNING: Sanction while compliant at t={t}")

    assert sanctions_while_compliant == 0, \
        f"Residents sanctioned ego while compliant (RED)! Count: {sanctions_while_compliant}"

    print("✓ No sanctions while compliant (RED)")

    print("\nPhase 2: Changing to BLUE (violating)...")
    # Wait until ready to plant again
    for t in range(50):
        obs, reward, term, trunc, info = env.step(0)
        if obs['ready_to_shoot'][0] > 0.5:
            break

    # Plant BLUE (violate)
    print("Planting BLUE (violation)...")
    obs, reward, term, trunc, info = env.step(10)  # Plant BLUE (action 10)

    # Check for sanctions after violation
    sanctions_while_violating = 0
    print("Running 150 frames while violating (BLUE)...")
    for t in range(150):
        obs, reward, term, trunc, info = env.step(0)
        if reward <= -9.0:
            sanctions_while_violating += 1

    print(f"\nResults:")
    print(f"  Sanctions while compliant (RED): {sanctions_while_compliant}")
    print(f"  Sanctions while violating (BLUE): {sanctions_while_violating}")

    assert sanctions_while_violating > 0, \
        f"CRITICAL FAILURE: Residents did NOT sanction ego after changing to wrong color! " \
        f"This means residents don't respond to color changes."

    print("✓ Residents sanctioned ego after color change to violation")
    print("✅ TEST PASSED")


def test_residents_enforce_across_all_colors():
    """Test that residents enforce rule regardless of which color is permitted.

    Tests enforcement for RED, GREEN, and BLUE as permitted colors.
    """
    for permitted_idx, permitted_name in [(1, 'RED'), (2, 'GREEN'), (3, 'BLUE')]:
        print(f"\n{'='*60}")
        print(f"Testing enforcement with {permitted_name} as permitted color")
        print(f"{'='*60}")

        env = AllelopathicHarvestGymEnv(
            arm='treatment',
            config={
                'permitted_color_index': permitted_idx,
                'startup_grey_grace': 25,
                'alpha': 0.5,
                'beta': 0.5,
                'c': 0.2,
            },
            seed=42 + permitted_idx,  # Different seed per color
            enable_telemetry=True,
            include_timestep=False,
        )

        obs, info = env.reset()

        # Step past grace period
        for t in range(30):
            obs, reward, term, trunc, info = env.step(0)  # Ego stays GREY (violating)

        # Check for sanctions
        sanctions = 0
        for t in range(100):
            obs, reward, term, trunc, info = env.step(0)
            if reward <= -9.0:
                sanctions += 1

        print(f"Sanctions received: {sanctions}")

        assert sanctions > 0, \
            f"Residents did NOT enforce rule when {permitted_name} is permitted!"

        print(f"✓ {permitted_name}: Enforcement working")

    print("\n✅ TEST PASSED: Residents enforce across all permitted colors")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

**Run the tests**:
```bash
python agents/tests/test_resident_sanctions_ego.py
```

**Expected Output**:
```
test_resident_sanctions_ego.py::test_residents_sanction_violating_ego

Stepping past grace period (30 frames)...
Grace period passed. Ego is now GREY (violating).
Running 200 frames, counting sanctions on violating ego...

Sanctions received by violating ego: 12
Sanction frames: [35, 242, 458, 672, 889, ...]
✓ Residents correctly sanctioned violating ego 12 times
✅ TEST PASSED
PASSED

test_resident_sanctions_ego.py::test_residents_never_sanction_compliant_ego

Waiting for ego to be ready to plant...
Ready at frame 4
Planting RED (compliant action)...
Running 500 frames, ego stays RED (compliant)...

Sanctions received by compliant ego (RED): 0
✓ Residents NEVER sanctioned compliant ego
✅ TEST PASSED
PASSED

test_resident_sanctions_ego.py::test_residents_sanction_after_color_change

Phase 1: Waiting to plant RED (compliant)...
Planting RED (compliant)...
Running 100 frames while compliant (RED)...
✓ No sanctions while compliant (RED)

Phase 2: Changing to BLUE (violating)...
Planting BLUE (violation)...
Running 150 frames while violating (BLUE)...

Results:
  Sanctions while compliant (RED): 0
  Sanctions while violating (BLUE): 8
✓ Residents sanctioned ego after color change to violation
✅ TEST PASSED
PASSED

test_resident_sanctions_ego.py::test_residents_enforce_across_all_colors

============================================================
Testing enforcement with RED as permitted color
============================================================
Sanctions received: 11
✓ RED: Enforcement working

============================================================
Testing enforcement with GREEN as permitted color
============================================================
Sanctions received: 9
✓ GREEN: Enforcement working

============================================================
Testing enforcement with BLUE as permitted color
============================================================
Sanctions received: 13
✓ BLUE: Enforcement working

✅ TEST PASSED: Residents enforce across all permitted colors
PASSED

================================ 4 passed in 180.25s ================================
```

**If ANY test fails**:
- Check `scripted_residents.py`: `_is_violation()`, `_is_eligible()`, `_try_zap()`
- Check `components.lua`: `SimpleZapSanction` logic
- Verify α/β/c values are correct

---

### Test 1.3: Resident Diagnostic Deep Dive (30 min)

**Purpose**: Comprehensive diagnostic of resident behavior in isolation

Create file: `test_resident_diagnostics.py` (in repo root)

```python
"""Comprehensive diagnostic of resident behavior.

Runs a full episode with ego idle (NOOP) and analyzes:
- Sanctions on ego
- Monoculture formation
- Selectivity (mis-zaps should be 0)
- Enforcement coverage
"""

from agents.envs.sb3_wrapper import AllelopathicHarvestGymEnv
from agents.residents.config import COLOR_RED
import numpy as np


def diagnose_resident_behavior():
    """Comprehensive diagnostic of resident behavior."""

    env = AllelopathicHarvestGymEnv(
        arm='treatment',
        config={
            'permitted_color_index': COLOR_RED,
            'startup_grey_grace': 25,
            'alpha': 0.5,
            'beta': 0.5,
            'c': 0.2,
            'immunity_cooldown': 200,
        },
        seed=42,
        enable_telemetry=True,
        include_timestep=False,
    )

    obs, info = env.reset()

    # Collect metrics over 2000 steps with ego idle (NOOP)
    sanctions_on_ego = []
    episode_rewards = []

    print("Running 2000 steps with ego NOOP (idle)...")
    print("Ego will be GREY throughout (violating after frame 25)")

    for t in range(2000):
        obs, reward, term, trunc, info = env.step(0)  # Ego NOOP

        episode_rewards.append(reward)

        # Track sanctions (reward spike of -10)
        if reward <= -9.0:
            sanctions_on_ego.append(t)

        if t % 500 == 0:
            print(f"  Frame {t}...")

    # Get final telemetry
    if env._recorder:
        metrics = env._recorder.get_episode_metrics(
            arm='treatment',
            permitted_color_index=COLOR_RED,
            grace_period=25,
            resident_baseline_r_eval=0,  # Dummy
            resident_baseline_sanctions=0,
            seed=42,
        )

        print("\n" + "="*80)
        print("RESIDENT BEHAVIOR DIAGNOSTIC REPORT")
        print("="*80)

        print(f"\n1. EGO BEHAVIOR (NOOP only - stays GREY):")
        print(f"   Total reward: {sum(episode_rewards):.2f}")
        print(f"   Sanctions received: {len(sanctions_on_ego)}")
        if len(sanctions_on_ego) > 0:
            print(f"   First 5 sanction frames: {sanctions_on_ego[:5]}")
            print(f"   Last 5 sanction frames: {sanctions_on_ego[-5:]}")
            print(f"   Average frames between sanctions: {np.mean(np.diff(sanctions_on_ego)):.1f}")

        print(f"\n2. RESIDENT ENFORCEMENT:")
        print(f"   Compliance %: {metrics.compliance_pct:.1f}%")
        print(f"     (Should be low ~0-5% since ego is GREY/violating)")
        print(f"   Violations/1k: {metrics.violations_per_1k:.1f}")
        print(f"     (Should be high ~900-1000 since ego violates most of episode)")
        print(f"   Zaps/1k: {metrics.zaps_per_1k:.1f}")
        print(f"     (Should be >0, shows residents are sanctioning)")

        print(f"\n3. BERRY FIELD (Monoculture):")
        print(f"   Permitted share: {metrics.permitted_color_share:.1%}")
        print(f"     (Should be ≥85% - residents create monoculture)")
        print(f"   Monoculture fraction: {metrics.monoculture_fraction:.1%}")

        print(f"\n4. SANCTION QUALITY (Selectivity):")
        print(f"   Correct zaps: {metrics.correct_zaps}")
        print(f"   Mis-zaps: {metrics.mis_zaps}")
        if metrics.correct_zaps + metrics.mis_zaps > 0:
            selectivity = metrics.correct_zaps / (metrics.correct_zaps + metrics.mis_zaps)
            print(f"   Selectivity: {selectivity:.1%}")
            print(f"     (Should be 100% - residents NEVER mis-zap)")

        print(f"\n5. IMMUNITY SYSTEM:")
        print(f"   Immunity events: {metrics.immunity_events}")
        print(f"     (Number of times immunity prevented sanction)")
        print(f"   Tie-break events: {metrics.tie_break_events}")
        print(f"     (Number of times same-step tie-break prevented double-zap)")

        # Validation checks
        print("\n" + "="*80)
        print("VALIDATION CHECKS:")
        print("="*80)

        checks = {
            "1. Ego receives sanctions (violating GREY)": len(sanctions_on_ego) > 0,
            "2. Residents never mis-zap (selectivity = 100%)": metrics.mis_zaps == 0,
            "3. Monoculture achieved (≥85% permitted)": metrics.permitted_color_share >= 0.85,
            "4. Violations detected (>800 per 1k)": metrics.violations_per_1k > 800,
            "5. Sanctions applied (>0 zaps per 1k)": metrics.zaps_per_1k > 0,
            "6. No catastrophic dogpiling (sanctions < violations)": len(sanctions_on_ego) < metrics.violations_per_1k,
        }

        all_passed = True
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}")
            if not passed:
                all_passed = False

        print("\n" + "="*80)
        if all_passed:
            print("✅ ALL RESIDENT CHECKS PASSED")
            print("Residents are working correctly. Safe to proceed to learning tests.")
            return True
        else:
            print("❌ RESIDENT CHECKS FAILED")
            print("DO NOT PROCEED TO LEARNING TESTS. Debug residents first.")
            return False
    else:
        print("ERROR: Telemetry not enabled")
        return False


if __name__ == "__main__":
    success = diagnose_resident_behavior()
    exit(0 if success else 1)
```

**Run diagnostic**:
```bash
python test_resident_diagnostics.py
```

**Expected Output**:
```
Running 2000 steps with ego NOOP (idle)...
Ego will be GREY throughout (violating after frame 25)
  Frame 0...
  Frame 500...
  Frame 1000...
  Frame 1500...

================================================================================
RESIDENT BEHAVIOR DIAGNOSTIC REPORT
================================================================================

1. EGO BEHAVIOR (NOOP only - stays GREY):
   Total reward: -142.30
   Sanctions received: 14
   First 5 sanction frames: [32, 248, 465, 682, 899]
   Last 5 sanction frames: [1116, 1333, 1550, 1767, 1984]
   Average frames between sanctions: 150.2

2. RESIDENT ENFORCEMENT:
   Compliance %: 1.2%
     (Should be low ~0-5% since ego is GREY/violating)
   Violations/1k: 987.5
     (Should be high ~900-1000 since ego violates most of episode)
   Zaps/1k: 140.0
     (Should be >0, shows residents are sanctioning)

3. BERRY FIELD (Monoculture):
   Permitted share: 89.3%
     (Should be ≥85% - residents create monoculture)
   Monoculture fraction: 89.3%

4. SANCTION QUALITY (Selectivity):
   Correct zaps: 14
   Mis-zaps: 0
   Selectivity: 100.0%
     (Should be 100% - residents NEVER mis-zap)

5. IMMUNITY SYSTEM:
   Immunity events: 14
     (Number of times immunity prevented sanction)
   Tie-break events: 0
     (Number of times same-step tie-break prevented double-zap)

================================================================================
VALIDATION CHECKS:
================================================================================
✓ 1. Ego receives sanctions (violating GREY)
✓ 2. Residents never mis-zap (selectivity = 100%)
✓ 3. Monoculture achieved (≥85% permitted)
✓ 4. Violations detected (>800 per 1k)
✓ 5. Sanctions applied (>0 zaps per 1k)
✓ 6. No catastrophic dogpiling (sanctions < violations)

================================================================================
✅ ALL RESIDENT CHECKS PASSED
Residents are working correctly. Safe to proceed to learning tests.
```

**If any check fails**: Debug before proceeding.

---

### Test 1.4: Monoculture Visual Verification (10 min)

```bash
# Run monoculture test with video output
pytest agents/tests/test_phase2_residents.py::test_r8_monoculture_achievement -v -s
```

**This generates a video** showing berry field evolution:

```bash
# Find the video
ls -lh /tmp/*monoculture*.mp4

# Open and watch (if you have GUI)
# Or copy to local machine to inspect
```

**What to verify in video**:
1. Berries start as mixed colors
2. Over time, permitted color (RED if `permitted_color_index=1`) dominates
3. By end (t=2000), ≥85% of berries are permitted color
4. Residents actively plant permitted color
5. Residents harvest berries

---

### Test 1.5: Resident Coverage (10 min)

**Purpose**: Verify residents sanction violators QUICKLY (within 10 frames)

```bash
pytest agents/tests/test_phase2_residents.py::test_r2_coverage -v -s
```

**Expected**: ≥80% of violators sanctioned within 10 frames

**Why critical**: Ego needs timely feedback. Delayed sanctions won't be credited to violations during learning.

---

### ✅ Phase 1 Exit Criteria

**YOU MUST SEE ALL OF THESE BEFORE PROCEEDING**:

1. ✅ All Phase 2 tests pass (R1-R8)
2. ✅ Residents sanction violating ego (Test 1.2, case 1)
3. ✅ Residents NEVER sanction compliant ego (Test 1.2, case 2)
4. ✅ Residents sanction after color change (Test 1.2, case 3)
5. ✅ Residents enforce across all colors (Test 1.2, case 4)
6. ✅ Diagnostic shows:
   - mis-zaps = 0
   - sanctions > 0
   - permitted_share ≥ 0.85
7. ✅ Coverage ≥80% within 10 frames
8. ✅ Video confirms monoculture formation

**If ANY fail**:

```
❌ STOP. DO NOT PROCEED TO LEARNING TESTS.

Debug residents first.

Check:
- agents/residents/scripted_residents.py:
  - _try_zap(): Is zapping logic correct?
  - _is_violation(): Does it correctly identify violations?
  - _is_eligible(): Does it check both violation AND immunity?

- meltingpot/lua/levels/allelopathic_harvest/components.lua:
  - SimpleZapSanction: Does it apply -10 and set immunity?
  - ImmunityTracker: Does it block sanctions during cooldown?
  - SameStepSanctionTracker: Does it prevent double-zaps?

- Reward shaping:
  - Are α, β, c values correct? (0.5, 0.5, 0.2)
  - Is α being tracked for stripping from eval?
```

---

## 🔥 Phase 2: Learning Probe (2 hours)

**Objective**: Verify agents learn ANYTHING in both arms (not random policy)

**Why before comparison**: Before comparing treatment vs control, verify basic learning works. If neither learns, no point comparing.

---

### Test 2.1: Quick Learning Configuration

Create `test_configs/quick_probe.yaml`:

```yaml
# Quick probe configuration - 100k steps for fast feedback

env:
  permitted_color_index: 1  # RED
  startup_grey_grace: 25
  episode_timesteps: 2000
  alpha: 0.5
  beta: 0.5
  c: 0.2
  immunity_cooldown: 200
  include_timestep: false  # ← CRITICAL: OFF

training:
  total_timesteps: 100000
  n_envs: 32  # Use all available CPUs
  seed: 42
  learning_rate: 0.0003
  n_steps: 128  # Smaller for faster iterations
  batch_size: 1024  # 32 envs × 128 steps / 4
  n_epochs: 4
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5

policy:
  trunk_dim: 256
  sanction_hidden_dim: 128
  ent_coef_game: 0.01
  ent_coef_sanction: 0.02
  recurrent: false  # Start with feed-forward

vec_normalize:
  norm_obs: false
  norm_reward: true
  clip_reward: 10.0
  gamma: 0.995

logging:
  wandb_project: altar-test
  wandb_entity: null
  wandb_run_name: null
  log_interval: 128  # Log often for debugging

checkpointing:
  save_freq: 50000
  checkpoint_dir: ./test_checkpoints
  save_vec_normalize: true

evaluation:
  eval_freq: 100000
  n_eval_episodes: 5
```

---

### Test 2.2: Run Treatment Probe (30 min)

```bash
# Create output directory
mkdir -p probe

# Run treatment
CUDA_VISIBLE_DEVICES=0 python agents/train/train_ppo.py treatment \
    --config test_configs/quick_probe.yaml \
    --total-timesteps 100000 \
    --seed 42 \
    --output-dir ./probe/treatment_100k \
    2>&1 | tee probe_treatment.log
```

**Monitor in real-time** (in another terminal):
```bash
tail -f probe_treatment.log
```

**What to watch for**:
- Episode rewards (should trend upward or stabilize, not random walk)
- Policy loss (should decrease initially)
- Value loss (should decrease initially)
- No NaN losses
- No crashes

---

### Test 2.3: Run Control Probe (30 min, parallel)

```bash
# Run control on second GPU (or same GPU after treatment)
CUDA_VISIBLE_DEVICES=1 python agents/train/train_ppo.py control \
    --config test_configs/quick_probe.yaml \
    --total-timesteps 100000 \
    --seed 42 \
    --output-dir ./probe/control_100k \
    2>&1 | tee probe_control.log
```

---

### Test 2.4: Analyze Learning Probe Results (20 min)

After both complete:

```bash
python -c "
import json
import re

# Parse treatment log
with open('probe_treatment.log') as f:
    t_log = f.read()

# Parse control log
with open('probe_control.log') as f:
    c_log = f.read()

# Extract episode rewards (rough parsing)
# SB3 logs: 'ep_rew_mean' periodically

print('='*80)
print('LEARNING PROBE ANALYSIS (100k steps)')
print('='*80)

# Check for completion
t_done = 'Training complete' in t_log or 'total_timesteps' in t_log
c_done = 'Training complete' in c_log or 'total_timesteps' in c_log

print(f'\nCompletion Status:')
print(f'  Treatment: {\"✓ DONE\" if t_done else \"✗ FAILED\"}')
print(f'  Control:   {\"✓ DONE\" if c_done else \"✗ FAILED\"}')

# Check for NaN
t_nan = 'nan' in t_log.lower() or 'NaN' in t_log
c_nan = 'nan' in c_log.lower() or 'NaN' in c_log

print(f'\nNaN Check:')
print(f'  Treatment: {\"✗ NaN detected\" if t_nan else \"✓ No NaN\"}')
print(f'  Control:   {\"✗ NaN detected\" if c_nan else \"✓ No NaN\"}')

# Try to extract ep_rew_mean if present
t_rewards = re.findall(r'ep_rew_mean.*?([\\-\\d\\.]+)', t_log)
c_rewards = re.findall(r'ep_rew_mean.*?([\\-\\d\\.]+)', c_log)

if t_rewards:
    t_rewards = [float(r) for r in t_rewards]
    print(f'\nTreatment Episode Rewards:')
    print(f'  Start: {t_rewards[0]:.2f}')
    print(f'  End:   {t_rewards[-1]:.2f}')
    print(f'  Δ:     {t_rewards[-1] - t_rewards[0]:.2f}')
    print(f'  Trend: {\"↑ Improving\" if t_rewards[-1] > t_rewards[0] else \"↓ Declining\"}')

if c_rewards:
    c_rewards = [float(r) for r in c_rewards]
    print(f'\nControl Episode Rewards:')
    print(f'  Start: {c_rewards[0]:.2f}')
    print(f'  End:   {c_rewards[-1]:.2f}')
    print(f'  Δ:     {c_rewards[-1] - c_rewards[0]:.2f}')
    print(f'  Trend: {\"↑ Improving\" if c_rewards[-1] > c_rewards[0] else \"↓ Declining\"}')

# Overall assessment
print('\n' + '='*80)
print('ASSESSMENT:')
print('='*80)

if t_done and c_done and not t_nan and not c_nan:
    print('✅ Both arms completed without crashes or NaN')
    if t_rewards and c_rewards:
        if t_rewards[-1] > t_rewards[0] or c_rewards[-1] > c_rewards[0]:
            print('✅ Learning detected (rewards improving)')
            print('   SAFE TO PROCEED to Phase 3 (Treatment vs Control)')
        else:
            print('⚠️  No clear learning trend in 100k steps')
            print('   May need longer training (500k-1M steps)')
    else:
        print('⚠️  Could not parse reward trends from logs')
        print('   Check W&B dashboard if enabled')
else:
    print('❌ Training failed or encountered errors')
    print('   DO NOT PROCEED. Debug issues first.')
"
```

---

### Test 2.5: Check FiLM Diagnostics (10 min, if W&B enabled)

If you have W&B configured:

```bash
# Check W&B dashboard for:
# 1. film/global_gamma_bias_deviation (Treatment should increase, Control stay near 0)
# 2. film/global_beta_bias_deviation (Treatment should increase, Control stay near 0)
# 3. policy/zap_rate (Should be > 0, not stuck at 0)
# 4. actions/entropy (Should decrease over time, indicating learning)
```

Or check tensorboard:
```bash
tensorboard --logdir ./probe
# Open browser to http://localhost:6006
```

---

### ✅ Phase 2 Exit Criteria

Before proceeding to Phase 3:

1. ✅ **No crashes** - Both runs complete 100k steps
2. ✅ **No NaN** - No NaN losses in either run
3. ✅ **Learning happens** - Episode rewards INCREASE from start to end (any amount)
   - Or at minimum: rewards stabilize (not random walk)
4. ✅ **FiLM diagnostics** (if W&B):
   - Treatment: `gamma/beta_deviation` > 0.05 (learning to use signal)
   - Control: `gamma/beta_deviation` < 0.02 (ignoring null signal)
5. ✅ **Action diversity** - Zap rate > 0, entropy > 0.5 (not stuck on one action)

**If any fail**:

```
❌ Debug before proceeding

Check:
1. Reward normalization enabled? (vec_normalize.norm_reward: true)
2. Gradient norms reasonable? (0.1-10, not >100 or <0.001)
3. Learning rate too high/low? Try 1e-4 (lower) or 1e-3 (higher)
4. Batch size/n_epochs? Try increasing n_epochs to 10
```

---

## ⚡ Phase 3: Treatment vs Control Comparison (4-5 hours)

**Objective**: Verify Treatment > Control (this is your hypothesis!)

**Most critical phase**: If this fails, your hypothesis is not supported.

---

### Test 3.1: Medium Learning Configuration

Create `test_configs/comparison.yaml`:

```yaml
# Comparison configuration - 500k steps for reliable signal

env:
  permitted_color_index: 1  # RED
  startup_grey_grace: 25
  episode_timesteps: 2000
  alpha: 0.5
  beta: 0.5
  c: 0.2
  immunity_cooldown: 200
  include_timestep: false  # ← CRITICAL: OFF

training:
  total_timesteps: 500000
  n_envs: 32
  seed: 42
  learning_rate: 0.0003
  n_steps: 256  # Back to standard
  batch_size: 2048  # 32 envs × 256 steps / 4
  n_epochs: 10
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5

policy:
  trunk_dim: 256
  sanction_hidden_dim: 128
  ent_coef_game: 0.01
  ent_coef_sanction: 0.02
  recurrent: false  # Will test LSTM in Phase 4

vec_normalize:
  norm_obs: false
  norm_reward: true
  clip_reward: 10.0
  gamma: 0.995

logging:
  wandb_project: altar-comparison
  log_interval: 2560  # Log every 10 rollouts

checkpointing:
  save_freq: 100000  # Every 100k
  checkpoint_dir: ./comparison_checkpoints
  save_vec_normalize: true

evaluation:
  eval_freq: 500000  # At end
  n_eval_episodes: 20
```

---

### Test 3.2: Run Both Arms in Parallel (90 min each)

```bash
# Create output directory
mkdir -p comparison

# GPU 0: Treatment
CUDA_VISIBLE_DEVICES=0 python agents/train/train_ppo.py treatment \
    --config test_configs/comparison.yaml \
    --seed 42 \
    --output-dir ./comparison/treatment_500k \
    2>&1 | tee comparison_treatment.log &

# GPU 1: Control
CUDA_VISIBLE_DEVICES=1 python agents/train/train_ppo.py control \
    --config test_configs/comparison.yaml \
    --seed 42 \
    --output-dir ./comparison/control_500k \
    2>&1 | tee comparison_control.log &

# Wait for both
wait

echo "Both training runs complete!"
```

**Monitor progress**:
```bash
# In another terminal
watch -n 10 'tail -n 5 comparison_treatment.log comparison_control.log'
```

---

### Test 3.3: Offline Evaluation (10 min)

```bash
# Evaluate Treatment
python agents/train/eval_cli.py \
    --checkpoint ./comparison/treatment_500k/ppo_treatment_step_500000.zip \
    --arm treatment \
    --n-episodes 20 \
    --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119 \
    --output-dir ./comparison/eval_treatment

# Evaluate Control
python agents/train/eval_cli.py \
    --checkpoint ./comparison/control_500k/ppo_control_step_500000.zip \
    --arm control \
    --n-episodes 20 \
    --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119 \
    --output-dir ./comparison/eval_control
```

---

### Test 3.4: Critical Analysis (10 min)

Create `analyze_comparison.py`:

```python
"""Analyze Treatment vs Control comparison results.

This is THE critical test for your hypothesis.
"""

import json
import numpy as np

# Load results
with open('./comparison/eval_treatment/results.json') as f:
    t = json.load(f)
with open('./comparison/eval_control/results.json') as f:
    c = json.load(f)

# Primary metrics
t_vg = t['treatment']['value_gap_mean']
c_vg = c['control']['value_gap_mean']

t_comp = t['treatment']['compliance_pct_mean']
c_comp = c['control']['compliance_pct_mean']

t_viol = t['treatment']['violations_per_1k_mean']
c_viol = c['control']['violations_per_1k_mean']

t_sr = t['treatment']['sanction_regret_mean']
c_sr = c['control']['sanction_regret_mean']

print("="*80)
print("CRITICAL COMPARISON - Treatment vs Control (500k steps)")
print("="*80)

print(f"\n📊 PRIMARY METRIC 1: Value Gap (Normative Competence)")
print(f"   Lower = Better (closer to resident baseline)")
print(f"   Treatment: {t_vg:.2f}")
print(f"   Control:   {c_vg:.2f}")
print(f"   Δ (T - C): {t_vg - c_vg:.2f}")
if t_vg < c_vg:
    improvement = (c_vg - t_vg) / abs(c_vg) * 100
    print(f"   ✅ Treatment WINS by {improvement:.1f}%")
else:
    print(f"   ❌ Control better or tie")

print(f"\n📊 PRIMARY METRIC 2: Sanction Regret (Normative Compliance)")
print(f"   Lower = Better (fewer sanctions received)")
print(f"   Treatment: {t_sr:.2f}")
print(f"   Control:   {c_sr:.2f}")
print(f"   Δ (T - C): {t_sr - c_sr:.2f}")
if t_sr < c_sr:
    improvement = (c_sr - t_sr) / abs(c_sr) * 100 if c_sr != 0 else 999
    print(f"   ✅ Treatment WINS by {improvement:.1f}%")
else:
    print(f"   ❌ Control better or tie")

print(f"\n📊 SUPPORTING METRIC: Compliance %")
print(f"   Higher = Better (follows rule more)")
print(f"   Treatment: {t_comp:.1f}%")
print(f"   Control:   {c_comp:.1f}%")
print(f"   Δ (T - C): {t_comp - c_comp:.1f} pp")
if t_comp > c_comp:
    print(f"   ✅ Treatment WINS")
else:
    print(f"   ❌ Control better or tie")

print(f"\n📊 SUPPORTING METRIC: Violations/1k")
print(f"   Lower = Better (violates less)")
print(f"   Treatment: {t_viol:.1f}")
print(f"   Control:   {c_viol:.1f}")
print(f"   Δ (T - C): {t_viol - c_viol:.1f}")
if t_viol < c_viol:
    print(f"   ✅ Treatment WINS")
else:
    print(f"   ❌ Control better or tie")

# Effect sizes
vg_effect = abs(t_vg - c_vg) / max(abs(t_vg), abs(c_vg), 1e-6) * 100
comp_effect = abs(t_comp - c_comp) / max(t_comp, c_comp, 1e-6) * 100

print(f"\n📈 EFFECT SIZES:")
print(f"   Value gap: {vg_effect:.1f}%")
print(f"   Compliance: {comp_effect:.1f}%")

# Decision
print("\n" + "="*80)
print("HYPOTHESIS VERDICT:")
print("="*80)

# Count wins
wins = 0
if t_vg < c_vg: wins += 1
if t_sr < c_sr: wins += 1
if t_comp > c_comp: wins += 1
if t_viol < c_viol: wins += 1

if wins >= 3:
    print("✅ HYPOTHESIS SUPPORTED")
    print(f"   Treatment wins {wins}/4 metrics")
    print("   Institutional signal DOES improve normative competence/compliance")
    if vg_effect >= 10:
        print("   Effect size: STRONG (≥10%)")
        print("   → Proceed with current design to final runs")
    elif vg_effect >= 5:
        print("   Effect size: MODERATE (5-10%)")
        print("   → Proceed to final runs, may need 20M steps for clear signal")
    else:
        print("   Effect size: WEAK (<5%)")
        print("   → Consider increasing training to 1M-2M steps for final runs")
elif wins >= 2:
    print("⚠️  WEAK SIGNAL")
    print(f"   Treatment wins {wins}/4 metrics")
    print("   Some advantage detected but not consistent")
    print("   → Need longer training (1M-2M steps) or hyperparameter tuning")
else:
    print("❌ HYPOTHESIS NOT SUPPORTED")
    print(f"   Treatment wins only {wins}/4 metrics")
    print("   NO CLEAR ADVANTAGE of institutional signal")
    print("\n   CRITICAL: Debug before final runs!")
    print("   Check:")
    print("   - FiLM diagnostics (is treatment learning to use signal?)")
    print("   - Observations (treatment has permitted_color, control doesn't?)")
    print("   - Reward components (α/β/c correct?)")
    print("   - Resident behavior (sanctioning correctly?)")

print("\n" + "="*80)
```

**Run analysis**:
```bash
python analyze_comparison.py
```

---

### ✅ Phase 3 Exit Criteria

**YOU MUST SEE THIS BEFORE FINALIZING DESIGN**:

1. ✅ **Treatment value gap < Control value gap** (any amount)
2. ✅ **Treatment sanction regret < Control sanction regret** (any amount)
3. ✅ **Treatment compliance > Control compliance** (any amount)
4. ✅ **Treatment wins ≥3/4 metrics**
5. ✅ **Effect size ≥ 5%** (meaningful difference)

**If weak signal (<5%)**:
- Plan for 1M-2M steps in final runs (instead of 500k)
- Consider hyperparameter tuning in Phase 4

**If no signal**:
```
❌ STOP. DO NOT PROCEED TO FINAL RUNS.

Debug:
1. Check W&B FiLM diagnostics:
   - Treatment: gamma/beta_deviation should increase
   - Control: gamma/beta_deviation should stay near 0

2. Verify observations:
   - Treatment must have 'permitted_color'
   - Control must NOT have 'permitted_color'

3. Check reward components:
   - α, β, c values correct? (0.5, 0.5, 0.2)
   - α being tracked and stripped from eval?

4. Verify residents:
   - Are they sanctioning violating ego?
   - Never mis-zapping compliant ego?
```

---

## 🎛️ Phase 4: Design Choices - LSTM vs Feed-Forward (2-3 hours)

**Objective**: Lock in architecture for final runs

**When to run**: Only if Phase 3 passes. If Phase 3 shows weak/no signal, skip this and debug instead.

---

### Test 4.1: LSTM Configuration

Create `test_configs/lstm_test.yaml` (copy from `comparison.yaml` and modify):

```yaml
# LSTM variant test
# Copy all from comparison.yaml except:

policy:
  trunk_dim: 256
  sanction_hidden_dim: 128
  ent_coef_game: 0.01
  ent_coef_sanction: 0.02
  recurrent: true  # ← Enable LSTM
  lstm_hidden_size: 256
```

---

### Test 4.2: Run LSTM Variant (90 min)

```bash
mkdir -p design

# Run LSTM variant (Treatment only, to save time)
CUDA_VISIBLE_DEVICES=0 python agents/train/train_ppo.py treatment \
    --config test_configs/lstm_test.yaml \
    --total-timesteps 500000 \
    --seed 42 \
    --output-dir ./design/treatment_lstm_500k \
    2>&1 | tee design_lstm.log
```

---

### Test 4.3: Evaluate LSTM (10 min)

```bash
python agents/train/eval_cli.py \
    --checkpoint ./design/treatment_lstm_500k/ppo_treatment_step_500000.zip \
    --arm treatment \
    --n-episodes 20 \
    --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119 \
    --output-dir ./design/eval_lstm
```

---

### Test 4.4: Compare LSTM vs Feed-Forward (5 min)

```bash
python -c "
import json

# Load results
with open('./comparison/eval_treatment/results.json') as f:
    ff = json.load(f)  # Feed-forward (from Phase 3)

with open('./design/eval_lstm/results.json') as f:
    lstm = json.load(f)

ff_vg = ff['treatment']['value_gap_mean']
ff_comp = ff['treatment']['compliance_pct_mean']

lstm_vg = lstm['treatment']['value_gap_mean']
lstm_comp = lstm['treatment']['compliance_pct_mean']

print('='*80)
print('LSTM vs FEED-FORWARD COMPARISON')
print('='*80)

print(f'\nValue Gap (lower = better):')
print(f'  Feed-Forward: {ff_vg:.2f}')
print(f'  LSTM:         {lstm_vg:.2f}')
print(f'  Δ (LSTM - FF): {lstm_vg - ff_vg:.2f}')

print(f'\nCompliance % (higher = better):')
print(f'  Feed-Forward: {ff_comp:.1f}%')
print(f'  LSTM:         {lstm_comp:.1f}%')
print(f'  Δ (LSTM - FF): {lstm_comp - ff_comp:.1f} pp')

# Decision (use 5% improvement threshold)
ff_better = ff_vg < lstm_vg * 0.95  # FF at least 5% better
lstm_better = lstm_vg < ff_vg * 0.95  # LSTM at least 5% better

print('\n' + '='*80)
print('DECISION:')
print('='*80)

if lstm_better:
    print('✅ Use LSTM for final runs')
    print(f'   LSTM improves value gap by {(ff_vg - lstm_vg) / ff_vg * 100:.1f}%')
    print('\nSet in final configs: policy.recurrent: true')
elif ff_better:
    print('✅ Use Feed-Forward for final runs')
    print(f'   Feed-Forward improves value gap by {(lstm_vg - ff_vg) / lstm_vg * 100:.1f}%')
    print('\nSet in final configs: policy.recurrent: false')
else:
    print('⚠️  LSTM and Feed-Forward perform similarly (< 5% difference)')
    print('   Use Feed-Forward (simpler, faster)')
    print('\nSet in final configs: policy.recurrent: false')

print('='*80)
"
```

---

### ✅ Phase 4 Exit Criteria

1. ✅ **LSTM decision made**: Use LSTM or Feed-Forward
2. ✅ **Performance verified**: Chosen architecture shows learning
3. ✅ **Record choice** for final configs

**Lock in**: `policy.recurrent: true/false`

---

## 🔒 Phase 5: Final Config Lock & Sanity Check (1 hour)

**Objective**: Create production configs, verify one final time before 20M runs

---

### Test 5.1: Create Final Configurations

Create directory: `final_configs/`

**Treatment configs** (3 seeds):

`final_configs/treatment_seed42.yaml`:
```yaml
env:
  permitted_color_index: 1  # RED
  startup_grey_grace: 25
  episode_timesteps: 2000
  alpha: 0.5
  beta: 0.5
  c: 0.2
  immunity_cooldown: 200
  include_timestep: false  # ← LOCKED: OFF

training:
  total_timesteps: 20000000  # 20M
  n_envs: 32
  seed: 42
  learning_rate: 0.0003  # Or optimized from Phase 4
  n_steps: 256
  batch_size: 2048
  n_epochs: 10
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5

policy:
  trunk_dim: 256
  sanction_hidden_dim: 128
  ent_coef_game: 0.01
  ent_coef_sanction: 0.02
  recurrent: false  # ← SET FROM PHASE 4 DECISION
  lstm_hidden_size: 256

vec_normalize:
  norm_obs: false
  norm_reward: true
  clip_reward: 10.0
  gamma: 0.995

logging:
  wandb_project: altar-transfer-final
  wandb_entity: null  # Set your entity
  log_interval: 2560

checkpointing:
  save_freq: 500000  # Every 500k
  checkpoint_dir: ./final_checkpoints
  save_vec_normalize: true

evaluation:
  eval_freq: 1000000  # Every 1M
  n_eval_episodes: 20
```

Copy and modify for seeds 43, 44:
```bash
cp final_configs/treatment_seed42.yaml final_configs/treatment_seed43.yaml
cp final_configs/treatment_seed42.yaml final_configs/treatment_seed44.yaml

# Edit seed: 42 → 43, 44
sed -i 's/seed: 42/seed: 43/g' final_configs/treatment_seed43.yaml
sed -i 's/seed: 42/seed: 44/g' final_configs/treatment_seed44.yaml
```

**Control configs** (3 seeds):
```bash
cp final_configs/treatment_seed42.yaml final_configs/control_seed42.yaml
cp final_configs/treatment_seed43.yaml final_configs/control_seed43.yaml
cp final_configs/treatment_seed44.yaml final_configs/control_seed44.yaml

# No other changes needed - control vs treatment is set via CLI arg
```

**Verify**:
```bash
ls -lh final_configs/
# Should see 6 files:
# treatment_seed42.yaml, treatment_seed43.yaml, treatment_seed44.yaml
# control_seed42.yaml, control_seed43.yaml, control_seed44.yaml
```

---

### Test 5.2: Dry Run Each Config (30 min)

```bash
# Test each config starts without error (run 1000 steps only)
mkdir -p dryrun

for config in final_configs/*.yaml; do
    basename=$(basename $config .yaml)
    arm=$(echo $basename | cut -d_ -f1)  # treatment or control

    echo "Testing $config..."

    timeout 300 python agents/train/train_ppo.py $arm \
        --config $config \
        --total-timesteps 1000 \
        --output-dir ./dryrun/$basename \
        2>&1 | tee dryrun_$basename.log

    if [ $? -eq 0 ]; then
        echo "✓ $config OK"
    else
        echo "✗ $config FAILED - check dryrun_$basename.log"
        exit 1
    fi
done

echo ""
echo "✅ ALL CONFIGS PASS DRY RUN"
```

---

### Test 5.3: Estimate Runtime (5 min)

```bash
python -c "
# Based on Phase 3 timing
import re

# Parse comparison log to estimate steps/sec
with open('comparison_treatment.log') as f:
    log = f.read()

# Look for timing info (SB3 logs FPS)
fps_matches = re.findall(r'fps.*?(\d+)', log)
if fps_matches:
    fps = float(fps_matches[-1])  # Latest FPS
    print(f'FPS from Phase 3: {fps:.1f}')

    total_steps = 20_000_000
    estimated_seconds = total_steps / fps
    estimated_hours = estimated_seconds / 3600

    print(f'\nEstimated time for 20M steps:')
    print(f'  {estimated_hours:.1f} hours')
    print(f'  {estimated_hours/24:.1f} days')

    print(f'\nWith 8 GPUs running in parallel:')
    print(f'  All 6 main runs finish in: {estimated_hours:.1f} hours')
    print(f'  2 ablation runs also finish in: {estimated_hours:.1f} hours')

    if estimated_hours > 12:
        print(f'\n⚠️  WARNING: Estimated time exceeds 12 hours')
        print(f'     Consider reducing to 10M steps or using more powerful GPUs')
    else:
        print(f'\n✅ Should complete within deadline')
else:
    print('Could not parse FPS from logs')
    print('Estimate: 6-10 hours per 20M run on modern GPU')
"
```

---

### Test 5.4: Pre-Flight Checklist (10 min)

```bash
python -c "
print('='*80)
print('PRE-FLIGHT CHECKLIST - Final Validation')
print('='*80)

import os
import yaml

# Check configs exist
configs = [
    'final_configs/treatment_seed42.yaml',
    'final_configs/treatment_seed43.yaml',
    'final_configs/treatment_seed44.yaml',
    'final_configs/control_seed42.yaml',
    'final_configs/control_seed43.yaml',
    'final_configs/control_seed44.yaml',
]

print('\n1. Config Files:')
all_exist = True
for cfg in configs:
    exists = os.path.exists(cfg)
    print(f'   {\"✓\" if exists else \"✗\"} {cfg}')
    if not exists:
        all_exist = False

# Check TIMESTEP is off
print('\n2. TIMESTEP Disabled:')
all_timestep_off = True
for cfg in configs:
    if os.path.exists(cfg):
        with open(cfg) as f:
            data = yaml.safe_load(f)
            timestep = data.get('env', {}).get('include_timestep', False)
            print(f'   {\"✓\" if not timestep else \"✗\"} {cfg}: {timestep}')
            if timestep:
                all_timestep_off = False

# Check seeds are different
print('\n3. Seed Diversity:')
seeds = []
for cfg in configs:
    if os.path.exists(cfg):
        with open(cfg) as f:
            data = yaml.safe_load(f)
            seed = data['training']['seed']
            seeds.append(seed)
print(f'   Seeds: {seeds}')
print(f'   {\"✓\" if len(set(seeds)) == 6 else \"✗\"} All unique')

# Check GPU availability
print('\n4. GPU Availability:')
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    gpu_count = len(result.stdout.strip().split('\n'))
    print(f'   GPUs detected: {gpu_count}')
    print(f'   {\"✓\" if gpu_count >= 8 else \"⚠️ \"} Need 8 GPUs (have {gpu_count})')
except:
    print('   ✗ nvidia-smi not found')

# Check disk space
print('\n5. Disk Space:')
try:
    result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
    print(f'   {result.stdout.split(chr(10))[1]}')
    # Rough estimate: 20M steps = ~10GB checkpoints per run = 60GB total
    print('   ⚠️  Ensure ≥100GB free for checkpoints')
except:
    print('   Could not check disk space')

# Overall
print('\n' + '='*80)
if all_exist and all_timestep_off and len(set(seeds)) == 6:
    print('✅ ALL PRE-FLIGHT CHECKS PASSED')
    print('   Ready to launch 8-GPU final runs')
else:
    print('❌ PRE-FLIGHT CHECKS FAILED')
    print('   Fix issues before launching')
print('='*80)
"
```

---

### ✅ Phase 5 Exit Criteria

Before launching final runs:

1. ✅ All 6 configs created (3 treatment + 3 control)
2. ✅ TIMESTEP disabled in all configs (`include_timestep: false`)
3. ✅ All seeds are unique (42, 43, 44)
4. ✅ LSTM decision locked in (`recurrent: true/false`)
5. ✅ All configs pass dry run
6. ✅ Runtime estimated (<12 hours per run)
7. ✅ Disk space sufficient (≥100GB free)
8. ✅ 8 GPUs available

**If all pass**: Proceed to FINAL LAUNCH

---

## 🚀 FINAL LAUNCH COMMANDS

```bash
# Create final output directory
mkdir -p final

# Set W&B API key (if using)
export WANDB_API_KEY=<your-key>

# Launch all 6 main runs in parallel

# GPU 0: Treatment seed 42
CUDA_VISIBLE_DEVICES=0 python agents/train/train_ppo.py treatment \
    --config final_configs/treatment_seed42.yaml \
    --output-dir ./final/treatment_s42 \
    2>&1 | tee final/treatment_s42.log &

# GPU 1: Treatment seed 43
CUDA_VISIBLE_DEVICES=1 python agents/train/train_ppo.py treatment \
    --config final_configs/treatment_seed43.yaml \
    --output-dir ./final/treatment_s43 \
    2>&1 | tee final/treatment_s43.log &

# GPU 2: Treatment seed 44
CUDA_VISIBLE_DEVICES=2 python agents/train/train_ppo.py treatment \
    --config final_configs/treatment_seed44.yaml \
    --output-dir ./final/treatment_s44 \
    2>&1 | tee final/treatment_s44.log &

# GPU 3: Control seed 42
CUDA_VISIBLE_DEVICES=3 python agents/train/train_ppo.py control \
    --config final_configs/control_seed42.yaml \
    --output-dir ./final/control_s42 \
    2>&1 | tee final/control_s42.log &

# GPU 4: Control seed 43
CUDA_VISIBLE_DEVICES=4 python agents/train/train_ppo.py control \
    --config final_configs/control_seed43.yaml \
    --output-dir ./final/control_s43 \
    2>&1 | tee final/control_s43.log &

# GPU 5: Control seed 44
CUDA_VISIBLE_DEVICES=5 python agents/train/train_ppo.py control \
    --config final_configs/control_seed44.yaml \
    --output-dir ./final/control_s44 \
    2>&1 | tee final/control_s44.log &

# GPUs 6-7: Reserved for ablations (launch after defining ablation configs)

echo "6 main experiments launched on GPUs 0-5"
echo "Monitor with: tail -f final/*.log"
echo "Or watch W&B dashboard"

# Wait for all background jobs
wait

echo "ALL EXPERIMENTS COMPLETE!"
```

---

## 📊 MONITORING DURING TRAINING

### Real-time Monitoring

```bash
# Watch all logs
tail -f final/*.log

# Or watch specific metrics
watch -n 30 'grep "ep_rew_mean" final/*.log | tail -12'

# Check GPU usage
watch -n 5 nvidia-smi
```

### W&B Dashboard (if enabled)

Monitor:
1. **Episode rewards** - Should increase over time
2. **FiLM diagnostics**:
   - Treatment: `gamma/beta_deviation` increasing
   - Control: `gamma/beta_deviation` staying near 0
3. **Value/policy losses** - Should decrease initially, then stabilize
4. **Action entropy** - Should decrease (exploitation)
5. **Zap rate** - Should be >0 but not dominant

---

## ⚠️ ABORT CRITERIA

**STOP training if you see**:

1. ❌ **NaN losses** - Model has diverged
   - Fix: Reduce learning rate, reduce batch size

2. ❌ **Rewards decreasing** - Learning is going wrong
   - Fix: Check hyperparameters, reward normalization

3. ❌ **Stuck on one action** - Policy collapsed
   - Fix: Increase entropy coefficients

4. ❌ **OOM errors** - Out of memory
   - Fix: Reduce n_envs from 32 to 16

---

## 📝 SUMMARY: CRITICAL PATH TO SUCCESS

**Total Time**: 11.5 hours (30 min buffer)

```
Phase 0 (30 min):  Environment Sanity ✓
         ↓
Phase 1 (2 hr):    Resident Validation ✓ (FOUNDATION)
         ↓
Phase 2 (2 hr):    Learning Probe ✓
         ↓
Phase 3 (4 hr):    Treatment vs Control ✓ (CRITICAL)
         ↓
Phase 4 (2 hr):    LSTM vs Feed-Forward ✓
         ↓
Phase 5 (1 hr):    Config Lock ✓
         ↓
    LAUNCH 20M runs (6-10 hours)
```

**Non-Negotiable Phases**: 0, 1, 2, 3, 5

**Optimizable**: Phase 4 (skip if time constrained, use feed-forward)

---

## 🎯 SUCCESS CRITERIA RECAP

**Before launching 20M runs, you MUST have**:

1. ✅ All Phase 0-1 tests pass (environment + residents work)
2. ✅ Both arms learn in Phase 2 (rewards increase)
3. ✅ Treatment > Control in Phase 3 (hypothesis supported)
4. ✅ LSTM decision made in Phase 4
5. ✅ All configs validated in Phase 5

**If ANY criterion fails**: Do NOT launch. Debug first.

---

**END OF CRITICAL_TESTS.md**

**Good luck! Your research depends on this. Execute with precision. 🚀**
