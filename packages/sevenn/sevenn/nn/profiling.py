"""
SevenNet Profiling Utilities

[PROFILING] This file is NEWLY ADDED for mlip-profiling project.
[PROFILING] Not part of the original SevenNet codebase.

Utilities for detailed operation profiling using torch.profiler.
Profiling can be enabled/disabled via global settings to minimize overhead.

Usage:
    from sevenn.nn.profiling import record_function_if_enabled, set_profiling_enabled
    
    # Enable profiling
    set_profiling_enabled(True)
    
    # Use in code
    with record_function_if_enabled("SevenNet::convolution"):
        # ... operations ...
"""

from contextlib import contextmanager

from torch.profiler import record_function

# Global profiling settings
_PROFILING_ENABLED = False


def set_profiling_enabled(enabled: bool) -> None:
    """Enable or disable profiling."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = enabled


def is_profiling_enabled() -> bool:
    """Return whether profiling is enabled."""
    return _PROFILING_ENABLED


@contextmanager
def record_function_if_enabled(name: str):
    """
    Execute record_function only when profiling is enabled.
    Minimizes overhead when disabled.
    """
    if _PROFILING_ENABLED:
        with record_function(name):
            yield
    else:
        yield
