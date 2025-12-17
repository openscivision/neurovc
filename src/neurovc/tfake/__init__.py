try:
    from tfan import ThermalLandmarks as ThermalLandmarks  # type: ignore
except ImportError as exc:
    raise ImportError(
        "The ThermalLandmarks class has moved to the 'tfan' package "
        "(install via 'pip install thermal-facial-alignment'). "
        f"Import error: {exc}"
    ) from exc
