__all__ = [
    "MidiConfig",
    "SummaryParameterSpace",
    "SurgeParamMeta",
    "SurgeParameterHelper",
    "SurgePedalboardRenderer",
]


def __getattr__(name: str):
    if name in {
        "MidiConfig",
        "SummaryParameterSpace",
        "SurgeParamMeta",
        "SurgeParameterHelper",
    }:
        from src.data.synth_backends.surge.surge_bridge import (
            MidiConfig as _MidiConfig,
            SummaryParameterSpace as _SummaryParameterSpace,
            SurgeParamMeta as _SurgeParamMeta,
            SurgeParameterHelper as _SurgeParameterHelper,
        )

        mapping = {
            "MidiConfig": _MidiConfig,
            "SummaryParameterSpace": _SummaryParameterSpace,
            "SurgeParamMeta": _SurgeParamMeta,
            "SurgeParameterHelper": _SurgeParameterHelper,
        }
        return mapping[name]

    if name == "SurgePedalboardRenderer":
        from src.data.synth_backends.surge.surge_renderer import (
            SurgePedalboardRenderer as _SurgePedalboardRenderer,
        )

        return _SurgePedalboardRenderer

    raise AttributeError(f"module 'src.data.synth_backends.surge' has no attribute {name!r}")
