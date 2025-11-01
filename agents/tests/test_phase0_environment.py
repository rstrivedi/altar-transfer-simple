"""Phase 0: Environment Sanity Checks

Test 0.1: Basic Environment Startup
Test 0.2: Vectorized Environment Creation
Test 0.3: Policy Instantiation
"""

import sys
import os
# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import yaml
import numpy as np
from agents.envs.sb3_wrapper import AllelopathicHarvestGymEnv


def test_0_1_basic_environment_startup():
    """Test 0.1: Basic environment creation and reset."""
    print("\n" + "="*80)
    print("Test 0.1: Basic Environment Startup")
    print("="*80)

    print('\nLoading treatment config...')
    with open('agents/train/configs/treatment.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print('Creating environment...')
    env = AllelopathicHarvestGymEnv(arm='treatment', config=config['env'])

    print('Resetting environment...')
    obs, info = env.reset()

    print(f'\n✓ SUCCESS: Environment created and reset')
    print(f'✓ Observation keys: {list(obs.keys())}')
    print(f'✓ RGB shape: {obs["rgb"].shape}')
    print(f'✓ ready_to_shoot shape: {obs["ready_to_shoot"].shape}')
    print(f'✓ permitted_color shape: {obs["permitted_color"].shape}')
    print(f'✓ timestep in obs: {"timestep" in obs}')

    # Verify expectations
    assert obs["rgb"].shape == (88, 88, 3), f"Expected RGB shape (88, 88, 3), got {obs['rgb'].shape}"
    assert obs["ready_to_shoot"].shape == (1,), f"Expected ready_to_shoot shape (1,), got {obs['ready_to_shoot'].shape}"
    assert obs["permitted_color"].shape == (3,), f"Expected permitted_color shape (3,), got {obs['permitted_color'].shape}"
    assert "timestep" not in obs, "timestep should NOT be in observations (include_timestep=False by default)"

    env.close()
    print('\n✓ Environment closed successfully')
    print('\n' + "="*80)
    print("TEST 0.1 PASSED")
    print("="*80 + '\n')


if __name__ == '__main__':
    test_0_1_basic_environment_startup()
