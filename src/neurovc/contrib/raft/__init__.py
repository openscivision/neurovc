try:
    from neurovc.contrib.raft.raft_helpers import RAFTOpticalFlow as _RAFTOpticalFlow

    HAS_TORCH = True
except Exception:  # noqa: BLE001
    HAS_TORCH = False

    class RAFTOpticalFlow:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise ImportError(
                "torch and related dependencies are required for RAFTOpticalFlow. "
                "Install the torch extra: 'pip install neurovc[torch]'"
            )
else:
    RAFTOpticalFlow = _RAFTOpticalFlow  # type: ignore[assignment]
    del _RAFTOpticalFlow


__all__ = [
    "HAS_TORCH",
    "RAFTOpticalFlow",
]
