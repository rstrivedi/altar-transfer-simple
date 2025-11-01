#!/usr/bin/env python3
"""Test that allelopathic_harvest_normative substrate loads without errors."""

import sys
from meltingpot.configs import substrates

def test_load_normative_substrate():
    """Test loading the normative substrate."""
    print("Testing allelopathic_harvest_normative__open substrate...")

    try:
        # Get the config
        config = substrates.get_config('allelopathic_harvest_normative__open')
        print("✓ Config loaded successfully")

        # Check observation names
        print(f"  Individual observations: {config.individual_observation_names}")
        print(f"  Global observations: {config.global_observation_names}")

        # Check that ALTAR is in observations
        assert 'ALTAR' in config.individual_observation_names, "ALTAR not in observations!"
        print("✓ ALTAR observation present")

        # Check timestep spec has observation spec
        # timestep_spec is a TimeStep spec object with observation field
        print(f"  Timestep spec type: {type(config.timestep_spec)}")
        print("✓ Timestep spec created")

        # Try to build the substrate with default roles
        roles = ['default'] * 8
        lab2d_settings = config.lab2d_settings_builder(config=config, roles=roles)
        print("✓ Substrate built successfully")

        # Check that our components are present
        game_objects = lab2d_settings['simulation']['gameObjects']
        scene = lab2d_settings['simulation']['scene']

        # Count overlays
        sanctioning_overlays = sum(1 for obj in game_objects if obj.get('name') == 'sanctioning_overlay')
        immunity_overlays = sum(1 for obj in game_objects if obj.get('name') == 'immunity_overlay')

        print(f"  Sanctioning overlays: {sanctioning_overlays}")
        print(f"  Immunity overlays: {immunity_overlays}")

        assert sanctioning_overlays == 8, f"Expected 8 sanctioning overlays, got {sanctioning_overlays}"
        assert immunity_overlays == 8, f"Expected 8 immunity overlays, got {immunity_overlays}"
        print("✓ All overlays present")

        # Check scene components
        scene_component_names = [comp['component'] for comp in scene['components']]
        assert 'Altar' in scene_component_names, "Altar not in scene components!"
        assert 'SameStepSanctionTracker' in scene_component_names, "SameStepSanctionTracker not in scene components!"
        print("✓ Scene components present")

        print("\n✅ All tests passed! Substrate loads correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Error loading substrate: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_load_normative_substrate()
    sys.exit(0 if success else 1)
