"""Utility helpers for FlowMag.

Online utilities depend on optional packages (e.g., torch, omegaconf); we guard
those imports so lightweight consumers (like downloaders) still work without
pulling heavy deps.
"""

import importlib

from neurovc.contrib.flowmag_util.models import (
    download_all_flowmag_models,
    download_flowmag_model,
    flowmag_model_downloader,
)

__all__ = [
    "download_all_flowmag_models",
    "download_flowmag_model",
    "flowmag_model_downloader",
]

_ONLINE_EXPORTS = {
    "FlowMagOnline",
    "FlowMagTTA",
    "OnlineMagnifier",
    "TTAStep",
    "default_alpha_policy",
    "load_flowmag_model",
    "unwrap",
}

__all__.extend(sorted(_ONLINE_EXPORTS))


def __getattr__(name):
    if name not in _ONLINE_EXPORTS:
        raise AttributeError(name)

    try:
        module = importlib.import_module("neurovc.contrib.flowmag_util.online")
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "FlowMag online utilities require optional dependencies "
            "(install via `pip install neurovc[contrib]`)."
        ) from exc

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pragma: no cover - cosmetic
    return sorted(set(__all__) | set(globals()))
