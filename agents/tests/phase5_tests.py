# Added by RST: Phase 5 acceptance tests
"""Acceptance tests for Phase 5 multi-community distributional competence (D1-D6).

Tests verify:
D1 - Independent sampling: Each worker samples community independently
D2 - Community tagging: Episodes correctly tagged with community_tag/community_idx
D3 - Balance check: Over many episodes, ~1:1:1 ratio across communities
D4 - Distributional metrics: Per-color and distributional aggregation works
D5 - Multi-community training: train_ppo.py works with --multi-community flag
D6 - Distributional evaluation: run_distributional_evaluation() works correctly

Usage:
    pytest agents/tests/phase5_tests.py -v
    pytest agents/tests/phase5_tests.py::test_d1_independent_sampling -v
"""

import pytest
import numpy as np
from collections import Counter
from typing import List

from agents.envs.sb3_wrapper import AllelopathicHarvestGymEnv, make_vec_env_multi_community
from agents.metrics.schema import (
    StepMetrics,
    EpisodeMetrics,
    DistributionalRunMetrics,
    RunMetrics,
)
from agents.metrics.aggregators import (
    compute_episode_metrics,
    aggregate_distributional_metrics,
)
from agents.metrics.eval_harness import run_distributional_evaluation


def get_test_config():
    """Get standard test configuration."""
    return {
        'permitted_color_index': 1,  # Placeholder (will be sampled)
        'startup_grey_grace': 25,
        'episode_timesteps': 2000,
        'altar_coords': (5, 15),
        'alpha': 0.5,
        'beta': 0.5,
        'c': 0.2,
        'immunity_cooldown': 200,
    }


# D1 - Independent sampling

def test_d1_independent_sampling():
    """D1: Each worker samples community independently with different RNG seeds."""
    config = get_test_config()

    # Create multiple envs with different seeds (simulating workers)
    seed = 42
    n_workers = 10

    envs = []
    for rank in range(n_workers):
        env = AllelopathicHarvestGymEnv(
            arm='treatment',
            config=config.copy(),
            seed=seed + rank,  # Different seed per worker
            enable_telemetry=True,
            multi_community_mode=True,
        )
        envs.append(env)

    # Reset all envs and check they sampled (potentially) different communities
    sampled_communities = []
    for env in envs:
        obs, info = env.reset()
        # Get community from env
        community_idx = env.config['permitted_color_index']
        sampled_communities.append(community_idx)

        # Verify community is valid
        assert community_idx in [1, 2, 3], f"Invalid community_idx: {community_idx}"

    # Check that we got some diversity (not all same community)
    # With 10 workers and 3 communities, very unlikely all are same (p < 0.001)
    unique_communities = set(sampled_communities)
    assert len(unique_communities) >= 2, \
        f"Independent sampling should produce diverse communities, got: {sampled_communities}"

    # Clean up
    for env in envs:
        env.close()

    print(f"✓ D1: Independent sampling verified (sampled communities: {sampled_communities})")


# D2 - Community tagging

def test_d2_community_tagging():
    """D2: Episodes are correctly tagged with community_tag and community_idx."""
    config = get_test_config()

    # Test all three communities
    for community_idx, community_name in [(1, 'RED'), (2, 'GREEN'), (3, 'BLUE')]:
        env = AllelopathicHarvestGymEnv(
            arm='treatment',
            config=config.copy(),
            seed=42,
            enable_telemetry=True,
            multi_community_mode=True,
        )

        # Force specific community for testing
        env._current_community_idx = community_idx
        env._current_community_name = community_name
        env.config['permitted_color_index'] = community_idx
        env.env_config.permitted_color_index = community_idx

        obs, info = env.reset()

        # Run for a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Get telemetry and check community tagging
        recorder = env.get_recorder()
        assert recorder is not None, "Telemetry should be enabled"

        step_metrics = recorder.get_step_metrics()
        assert len(step_metrics) > 0, "Should have recorded step metrics"

        # Verify all steps have correct community tag
        for step in step_metrics:
            assert step.community_tag == community_name, \
                f"Step should have community_tag='{community_name}', got '{step.community_tag}'"
            assert step.community_idx == community_idx, \
                f"Step should have community_idx={community_idx}, got {step.community_idx}"

        env.close()

    print("✓ D2: Community tagging verified for RED/GREEN/BLUE")


# D3 - Balance check

def test_d3_balance_check():
    """D3: Over many episodes, ~1:1:1 ratio across communities (Law of Large Numbers)."""
    config = get_test_config()
    config['episode_timesteps'] = 100  # Shorter episodes for testing

    seed = 42
    n_episodes = 90  # Should get ~30 per community

    sampled_communities = []

    for episode_idx in range(n_episodes):
        env = AllelopathicHarvestGymEnv(
            arm='treatment',
            config=config.copy(),
            seed=seed + episode_idx,
            enable_telemetry=True,
            multi_community_mode=True,
        )

        obs, info = env.reset()

        # Record sampled community
        community_idx = env.config['permitted_color_index']
        sampled_communities.append(community_idx)

        env.close()

    # Count communities
    community_counts = Counter(sampled_communities)

    # Verify we got all three communities
    assert set(community_counts.keys()) == {1, 2, 3}, \
        "Should sample all three communities"

    # Verify balance (each should be ~30 ± tolerance)
    expected_per_community = n_episodes / 3  # 30
    tolerance = 0.3  # ±30% tolerance (21-39 per community)

    for community_idx in [1, 2, 3]:
        count = community_counts[community_idx]
        min_expected = expected_per_community * (1 - tolerance)
        max_expected = expected_per_community * (1 + tolerance)

        assert min_expected <= count <= max_expected, \
            f"Community {community_idx} count {count} outside expected range [{min_expected:.0f}, {max_expected:.0f}]"

    # Compute balance ratio (max/min should be close to 1.0)
    counts = list(community_counts.values())
    balance_ratio = max(counts) / min(counts)

    assert balance_ratio < 2.0, \
        f"Balance ratio {balance_ratio:.2f} too high (max/min counts)"

    print(f"✓ D3: Balance check verified (counts: RED={community_counts[1]}, GREEN={community_counts[2]}, BLUE={community_counts[3]}, ratio={balance_ratio:.2f})")


# D4 - Distributional metrics

def test_d4_distributional_metrics():
    """D4: Per-color and distributional aggregation works correctly."""
    config = get_test_config()

    # Create mock episodes for each community
    def create_mock_episode(community_tag: str, community_idx: int, value_gap: float) -> EpisodeMetrics:
        """Create a mock episode with known metrics."""
        return EpisodeMetrics(
            arm='treatment',
            permitted_color_index=community_idx,
            episode_length=100,
            r_eval=10.0,
            r_train=12.0,
            ego_body_color=community_idx,
            compliance_pct=0.8,
            violations_per_1k=50.0,
            permitted_share=0.6,
            monoculture_fraction=0.3,
            zaps_attempted=10,
            zaps_correct=8,
            zaps_incorrect=2,
            zaps_received=5,
            sanctions_applied=7,
            sanctions_blocked=1,
            value_gap=value_gap,
            sanction_regret=1.5,
            collective_cost_per_sanction=-0.5,
            community_tag=community_tag,
            community_idx=community_idx,
        )

    # Create episodes with different value gaps
    red_episodes = [create_mock_episode('RED', 1, 2.0), create_mock_episode('RED', 1, 2.5)]
    green_episodes = [create_mock_episode('GREEN', 2, 3.0), create_mock_episode('GREEN', 2, 3.5)]
    blue_episodes = [create_mock_episode('BLUE', 3, 1.5), create_mock_episode('BLUE', 3, 2.0)]

    all_episodes = red_episodes + green_episodes + blue_episodes

    # Aggregate distributional metrics
    dist_metrics = aggregate_distributional_metrics(
        episodes=all_episodes,
        arm='treatment',
        config=config,
    )

    # Verify per-color metrics exist
    assert dist_metrics.red_metrics is not None, "Should have RED metrics"
    assert dist_metrics.green_metrics is not None, "Should have GREEN metrics"
    assert dist_metrics.blue_metrics is not None, "Should have BLUE metrics"

    # Verify per-color value gaps
    assert np.isclose(dist_metrics.red_metrics.value_gap_mean, 2.25, atol=0.01), \
        f"RED value_gap should be 2.25, got {dist_metrics.red_metrics.value_gap_mean}"
    assert np.isclose(dist_metrics.green_metrics.value_gap_mean, 3.25, atol=0.01), \
        f"GREEN value_gap should be 3.25, got {dist_metrics.green_metrics.value_gap_mean}"
    assert np.isclose(dist_metrics.blue_metrics.value_gap_mean, 1.75, atol=0.01), \
        f"BLUE value_gap should be 1.75, got {dist_metrics.blue_metrics.value_gap_mean}"

    # Verify distributional summary
    expected_avg = (2.25 + 3.25 + 1.75) / 3  # 2.4167
    assert np.isclose(dist_metrics.avg_value_gap, expected_avg, atol=0.01), \
        f"Average value_gap should be {expected_avg:.3f}, got {dist_metrics.avg_value_gap}"

    assert dist_metrics.worst_value_gap == 3.25, "Worst should be GREEN (3.25)"
    assert dist_metrics.worst_community == "GREEN", "Worst community should be GREEN"

    assert dist_metrics.best_value_gap == 1.75, "Best should be BLUE (1.75)"
    assert dist_metrics.best_community == "BLUE", "Best community should be BLUE"

    # Verify balance check (2 episodes per community)
    assert dist_metrics.balance_check_ratio == 1.0, "Balance should be perfect (2:2:2)"

    print("✓ D4: Distributional metrics verified (avg=2.42, worst=GREEN:3.25, best=BLUE:1.75)")


# D5 - Multi-community training

@pytest.mark.slow
def test_d5_multi_community_training():
    """D5: train_ppo.py works with --multi-community flag (smoke test)."""
    pytest.skip("D5 requires training run, test manually with: python agents/train/train_ppo.py treatment --multi-community --config agents/train/configs/smoke_test_multi.yaml")

    # This test would require:
    # 1. Running: python agents/train/train_ppo.py treatment --multi-community --config agents/train/configs/smoke_test_multi.yaml
    # 2. Verifying it completes without errors
    # 3. Checking that community sampling happened (from logs)
    # Too slow for unit tests; should be run as integration test

    print("✓ D5: Multi-community training (skipped, run manually)")


# D6 - Distributional evaluation

def test_d6_distributional_evaluation():
    """D6: run_distributional_evaluation() works correctly across all communities."""
    config = get_test_config()
    config['episode_timesteps'] = 200  # Shorter for testing

    # Create a simple policy (random)
    def random_policy(obs):
        """Random policy for testing."""
        return np.random.randint(0, 11)

    # Run distributional evaluation
    results = run_distributional_evaluation(
        ego_policy=random_policy,
        config=config,
        num_episodes_per_community=3,  # Small number for testing
        seed=42,
    )

    # Verify results structure
    assert 'baseline' in results, "Should have baseline results"
    assert 'treatment' in results, "Should have treatment results"
    assert 'control' in results, "Should have control results"

    # Verify all are DistributionalRunMetrics
    for key in ['baseline', 'treatment', 'control']:
        dist_metrics = results[key]
        assert isinstance(dist_metrics, DistributionalRunMetrics), \
            f"{key} should be DistributionalRunMetrics"

        # Verify per-community metrics exist
        assert dist_metrics.red_metrics is not None, f"{key} should have RED metrics"
        assert dist_metrics.green_metrics is not None, f"{key} should have GREEN metrics"
        assert dist_metrics.blue_metrics is not None, f"{key} should have BLUE metrics"

        # Verify distributional summary exists
        assert dist_metrics.avg_value_gap >= 0, f"{key} avg_value_gap should be non-negative"
        assert dist_metrics.worst_community in ['RED', 'GREEN', 'BLUE'], \
            f"{key} worst_community should be RED/GREEN/BLUE"
        assert dist_metrics.best_community in ['RED', 'GREEN', 'BLUE'], \
            f"{key} best_community should be RED/GREEN/BLUE"

        # Verify balance check (should be 1.0 with equal episodes)
        assert np.isclose(dist_metrics.balance_check_ratio, 1.0, atol=0.01), \
            f"{key} balance_check_ratio should be ~1.0, got {dist_metrics.balance_check_ratio}"

        # Verify episode counts
        assert dist_metrics.num_episodes == 9, f"{key} should have 9 total episodes (3 per community)"

    print(f"✓ D6: Distributional evaluation verified (baseline avg_ΔV={results['baseline'].avg_value_gap:.3f}, treatment avg_ΔV={results['treatment'].avg_value_gap:.3f})")


# Integration test: Multi-community env creation

def test_make_vec_env_multi_community():
    """Test make_vec_env_multi_community helper function."""
    config = get_test_config()

    # Create vectorized multi-community env
    vec_env = make_vec_env_multi_community(
        arm='treatment',
        num_envs=4,
        config=config,
        seed=42,
        enable_telemetry=False,
    )

    # Reset and verify it works
    obs = vec_env.reset()

    # Verify observation shape
    assert 'rgb' in obs, "Should have rgb observation"
    assert obs['rgb'].shape[0] == 4, "Should have 4 environments"

    # Step and verify it works
    actions = [vec_env.action_space.sample() for _ in range(4)]
    obs, rewards, dones, infos = vec_env.step(actions)

    assert len(rewards) == 4, "Should have 4 rewards"
    assert len(dones) == 4, "Should have 4 done flags"

    vec_env.close()

    print("✓ Integration: make_vec_env_multi_community works")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
