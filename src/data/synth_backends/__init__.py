"""Synth backend package."""

__all__ = [
    "DexedParameterHelper",
    "DexedDawRenderer",
    "SurgeParameterHelper",
    "SurgePedalboardRenderer",
]


def __getattr__(name: str):
    if name == "DexedParameterHelper":
        from src.data.synth_backends.dexed.dexed_bridge import DexedParameterHelper as _DexedParameterHelper

        return _DexedParameterHelper
    if name == "DexedDawRenderer":
        from src.data.synth_backends.dexed.dexed_renderer import DexedDawRenderer as _DexedDawRenderer

        return _DexedDawRenderer
    if name == "SurgeParameterHelper":
        from src.data.synth_backends.surge.surge_bridge import SurgeParameterHelper as _SurgeParameterHelper

        return _SurgeParameterHelper
    if name == "SurgePedalboardRenderer":
        from src.data.synth_backends.surge.surge_renderer import (
            SurgePedalboardRenderer as _SurgePedalboardRenderer,
        )

        return _SurgePedalboardRenderer
    raise AttributeError(f"module 'src.data.synth_backends' has no attribute {name!r}")
