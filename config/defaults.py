"""
config/defaults.py
===================
Default configuration values, ported from loadDefaultConfig.m.

Unlike the MATLAB version (a single nested struct), the JAX port spreads
settings across purpose-specific dataclasses that are threaded explicitly
through the pipeline (PriorConfig in core/prior.py, OptimizerConfig in
inference/optimizer.py). This module holds the one piece that didn't
already have a home - preprocessing settings - plus a convenience
default_full_config() that mirrors loadDefaultConfig.m's role of
returning "the" default configuration for the whole pipeline.

MATLAB defaults (loadDefaultConfig.m) and where each now lives:

    config.preproc.DSmode / DSvalue        -> PreprocConfig (this file)
    config.prior.mu / sig                  -> PriorConfig.mu_*, sig_*
    config.optimizer.*                     -> OptimizerConfig
    config.model.T_h / LFL / LFL_sigma     -> PriorConfig.T_h / LFL / LFL_sigma
    config.inference.*                     -> InferenceConfig (top level)
    config.diagnostics.*                   -> not ported (plotting hooks)
"""

from dataclasses import dataclass, field

from core.prior import PriorConfig
from inference.optimizer import OptimizerConfig


# ---------------------------------------------------------------------------
# Preprocessing settings  (config.preproc.* in loadDefaultConfig.m)
# ---------------------------------------------------------------------------

@dataclass
class PreprocConfig:
    """
    Signal downsampling configuration, passed to prepare_signals.

    Attributes
    ----------
    DSmode  : str
        'factor' | 'frequency' | 'rate'. Default 'factor'.
    DSvalue : float
        Value interpreted according to DSmode. Default 1, which under
        'factor' mode means no downsampling - matches loadDefaultConfig.m.
    """
    DSmode  : str   = 'factor'
    DSvalue : float = 1


# ---------------------------------------------------------------------------
# Full default configuration  (mirrors loadDefaultConfig.m as a whole)
# ---------------------------------------------------------------------------

@dataclass
class FullConfig:
    """
    Bundle of all sub-configurations, for convenience / introspection.

    This mirrors the structure of loadDefaultConfig.m's returned struct.
    The actual pipeline functions take PriorConfig / OptimizerConfig /
    PreprocConfig independently rather than this bundle, since that keeps
    each stage's dependencies explicit; use this when you want "the
    MATLAB defaults" as a single object to inspect or partially override.
    """
    preproc   : PreprocConfig  = field(default_factory=PreprocConfig)
    prior     : PriorConfig    = field(default_factory=PriorConfig)
    optimizer : OptimizerConfig = field(default_factory=OptimizerConfig)


def default_full_config() -> FullConfig:
    """Return the default configuration, matching loadDefaultConfig.m."""
    return FullConfig()
