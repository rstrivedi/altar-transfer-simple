# Added by RST: Phase 3 metrics package
"""Metrics and evaluation infrastructure for normative allelopathic harvest.

This package provides:
- schema.py: Typed data structures for telemetry
- recorder.py: Real-time event and state capture
- aggregators.py: Pure functions for metric computation
- eval_harness.py: A/B evaluation protocol with resident baseline
- wandb_logging.py: Weights & Biases integration
- video.py: Enhanced video rendering with overlays
"""

from agents.metrics.schema import (
    StepMetrics,
    EpisodeMetrics,
    RunMetrics,
)

__all__ = [
    'StepMetrics',
    'EpisodeMetrics',
    'RunMetrics',
]
