__all__ = [
    "DexedParamMeta",
    "SummaryParameterSpace",
    "DexedParameterHelper",
    "DexedDawRenderer",
]


def __getattr__(name: str):
    if name in {"DexedParamMeta", "SummaryParameterSpace", "DexedParameterHelper"}:
        from src.data.synth_backends.dexed.dexed_bridge import (
            DexedParamMeta as _DexedParamMeta,
            DexedParameterHelper as _DexedParameterHelper,
            SummaryParameterSpace as _SummaryParameterSpace,
        )

        mapping = {
            "DexedParamMeta": _DexedParamMeta,
            "SummaryParameterSpace": _SummaryParameterSpace,
            "DexedParameterHelper": _DexedParameterHelper,
        }
        return mapping[name]

    if name == "DexedDawRenderer":
        from src.data.synth_backends.dexed.dexed_renderer import DexedDawRenderer as _DexedDawRenderer

        return _DexedDawRenderer

    raise AttributeError(f"module 'src.data.synth_backends.dexed' has no attribute {name!r}")
