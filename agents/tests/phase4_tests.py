# Added by RST: Phase 4 acceptance tests
"""Acceptance tests for Phase 4 RL integration (T1-T6).

Tests verify:
T1 - SB3 env check (check_env passes for both arms)
T2 - Space parity (treatment has permitted_color, control doesn't)
T3 - Reward identity (r_train = r_env + alpha - beta - c, strip test holds)
T4 - Residents integration (residents enforce as in Phase 2)
T5 - Tie-break & immunity (at most one -10 per step, no second -10 within K steps)
T6 - Learning smoke test (treatment improves faster than control)

Usage:
    pytest agents/tests/phase4_tests.py -v
    pytest agents/tests/phase4_tests.py::test_t1_sb3_env_check -v
"""

import pytest
import numpy as np
from stable_baselines3.common.env_checker import check_env

from agents.envs.sb3_wrapper import AllelopathicHarvestGymEnv


def get_test_config():
    """Get standard test configuration."""
    return {
        'permitted_color_index': 1,
        'startup_grey_grace': 25,
        'episode_timesteps': 2000,
        'altar_coords': (5, 15),
        'alpha': 0.5,
        'beta': 0.5,
        'c': 0.2,
        'immunity_cooldown': 200,
    }


# T1 - SB3 env check

def test_t1_sb3_env_check_treatment():
    """T1: SB3 check_env passes for treatment arm."""
    env = AllelopathicHarvestGymEnv(
        arm='treatment',
        config=get_test_config(),
        seed=42,
        enable_telemetry=True,
    )

    # Run SB3's check_env (will raise if issues found)
    check_env(env, warn=True)

    env.close()
    print("✓ T1 (treatment): check_env passed")


def test_t1_sb3_env_check_control():
    """T1: SB3 check_env passes for control arm."""
    env = AllelopathicHarvestGymEnv(
        arm='control',
        config=get_test_config(),
        seed=42,
        enable_telemetry=True,
    )

    # Run SB3's check_env (will raise if issues found)
    check_env(env, warn=True)

    env.close()
    print("✓ T1 (control): check_env passed")


# T2 - Space parity

def test_t2_space_parity():
    """T2: Treatment has permitted_color, control doesn't; action spaces identical."""
    config = get_test_config()

    # Treatment env
    env_treatment = AllelopathicHarvestGymEnv(arm='treatment', config=config, seed=42)

    # Control env
    env_control = AllelopathicHarvestGymEnv(arm='control', config=config, seed=42)

    # Check observation spaces
    obs_treatment = env_treatment.observation_space.spaces
    obs_control = env_control.observation_space.spaces

    # Treatment should have permitted_color
    assert 'permitted_color' in obs_treatment, "Treatment obs should include permitted_color"
    assert obs_treatment['permitted_color'].shape == (3,), "permitted_color should be (3,) one-hot"

    # Control should NOT have permitted_color
    assert 'permitted_color' not in obs_control, "Control obs should NOT include permitted_color"

    # Both should have rgb, ready_to_shoot, timestep
    for key in ['rgb', 'ready_to_shoot', 'timestep']:
        assert key in obs_treatment, f"Treatment obs should include {key}"
        assert key in obs_control, f"Control obs should include {key}"

    # Action spaces should be identical
    assert env_treatment.action_space == env_control.action_space, "Action spaces should be identical"
    assert env_treatment.action_space.n == 11, "Action space should be Discrete(11)"

    env_treatment.close()
    env_control.close()
    print("✓ T2: Space parity verified")


# T3 - Reward identity

def test_t3_reward_identity():
    """T3: r_train = r_env + alpha - beta - c, and strip test holds."""
    config = get_test_config()
    env = AllelopathicHarvestGymEnv(arm='treatment', config=config, seed=42, enable_telemetry=True)

    obs, info = env.reset()

    # Run for 100 steps with random actions
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Get telemetry from recorder
    recorder = env.get_recorder()
    assert recorder is not None, "Telemetry should be enabled"

    step_metrics = recorder.get_step_metrics()
    assert len(step_metrics) > 0, "Should have recorded step metrics"

    # Compute cumulative sums
    r_env_sum = sum(s.r_env for s in step_metrics)
    alpha_sum = sum(s.alpha for s in step_metrics)
    beta_sum = sum(s.beta for s in step_metrics)
    c_sum = sum(s.c for s in step_metrics)

    # Compute r_train (from step rewards)
    cumulative_sums = recorder.get_cumulative_sums()
    r_total_sum = cumulative_sums['r_total_sum'][env.ego_index]

    # r_train = r_total (already includes alpha, beta, c)
    r_train = r_total_sum

    # Identity 1: r_train = r_env + alpha - beta - c
    expected_r_train = r_env_sum + alpha_sum - beta_sum - c_sum
    assert np.isclose(r_train, expected_r_train, atol=1e-6), \
        f"r_train ({r_train}) should equal r_env + alpha - beta - c ({expected_r_train})"

    # Identity 2: R_eval = R_train - alpha (strip test)
    r_eval = recorder.get_r_eval()
    expected_r_eval = r_train - alpha_sum
    assert np.isclose(r_eval, expected_r_eval, atol=1e-6), \
        f"R_eval ({r_eval}) should equal R_train - alpha ({expected_r_eval})"

    env.close()
    print("✓ T3: Reward identity verified")


# T4 - Residents integration

def test_t4_residents_integration():
    """T4: With ego idle (NOOP), residents enforce as in Phase 2."""
    config = get_test_config()
    env = AllelopathicHarvestGymEnv(arm='treatment', config=config, seed=42, enable_telemetry=True)

    obs, info = env.reset()

    # Run with ego NOOP (action 0)
    sanction_count = 0
    for _ in range(200):
        action = 0  # NOOP
        obs, reward, terminated, truncated, info = env.step(action)

        # Count sanctions from events
        if 'events' in info:
            for event in info['events']:
                if event.get('name') == 'sanction' and event.get('applied_minus10'):
                    sanction_count += 1

        if terminated or truncated:
            break

    # Verify that sanctions occurred (residents are enforcing)
    assert sanction_count > 0, "Residents should apply sanctions even when ego is idle"

    env.close()
    print(f"✓ T4: Residents integration verified ({sanction_count} sanctions observed)")


# T5 - Tie-break & immunity

def test_t5_tie_break_and_immunity():
    """T5: At most one -10 per step; no second -10 within K steps on same target."""
    config = get_test_config()
    config['immunity_cooldown'] = 50  # Shorter for testing

    env = AllelopathicHarvestGymEnv(arm='treatment', config=config, seed=42, enable_telemetry=True)

    obs, info = env.reset()

    # Track sanctions per target per timestep
    sanctions_by_timestep = {}  # timestep -> list of (zappee_id, applied_minus10)
    last_sanction_timestep = {}  # zappee_id -> timestep of last -10

    for t in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Parse sanction events
        if 'events' in info:
            timestep_sanctions = []
            for event in info['events']:
                if event.get('name') == 'sanction':
                    zappee_id = event.get('zappee_id')
                    applied_minus10 = event.get('applied_minus10', False)

                    if zappee_id is not None:
                        timestep_sanctions.append((zappee_id, applied_minus10))

                        # Check immunity constraint
                        if applied_minus10:
                            if zappee_id in last_sanction_timestep:
                                steps_since_last = t - last_sanction_timestep[zappee_id]
                                assert steps_since_last >= config['immunity_cooldown'], \
                                    f"Second -10 on target {zappee_id} after only {steps_since_last} steps (K={config['immunity_cooldown']})"

                            last_sanction_timestep[zappee_id] = t

            sanctions_by_timestep[t] = timestep_sanctions

        if terminated or truncated:
            break

    # Check tie-break: at most one -10 per target per timestep
    for t, sanctions in sanctions_by_timestep.items():
        minus10_by_target = {}
        for zappee_id, applied_minus10 in sanctions:
            if applied_minus10:
                minus10_by_target[zappee_id] = minus10_by_target.get(zappee_id, 0) + 1

        for zappee_id, count in minus10_by_target.items():
            assert count <= 1, f"Target {zappee_id} received {count} -10s at timestep {t} (tie-break violation)"

    env.close()
    print("✓ T5: Tie-break & immunity verified")


# T6 - Learning smoke test

@pytest.mark.slow
def test_t6_learning_smoke():
    """T6: Treatment shows faster compliance rise than control (directional check).

    NOTE: This is a smoke test and may be flaky. It verifies that learning is happening,
    not that treatment definitively outperforms control (requires longer training).
    """
    pytest.skip("T6 requires long training (1M steps), run manually with smoke_test.yaml")

    # This test would require:
    # 1. Training treatment for 1M steps
    # 2. Training control for 1M steps
    # 3. Comparing final compliance% and SR_events
    # Too slow for unit tests; should be run as integration test

    # Pseudo-code:
    # train_treatment(total_timesteps=1_000_000)
    # train_control(total_timesteps=1_000_000)
    # eval_treatment = evaluate(checkpoint_treatment)
    # eval_control = evaluate(checkpoint_control)
    # assert eval_treatment.compliance_pct > eval_control.compliance_pct
    # assert eval_treatment.sanction_regret < eval_control.sanction_regret

    print("✓ T6: Learning smoke test (skipped, run manually)")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
