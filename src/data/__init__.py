"""Data package."""

__all__ = [
    "H5SynthDataset",
    "H5SynthDataModule",
]


def __getattr__(name: str):
    if name == "H5SynthDataset":
        from src.data.h5_synth_dataset import H5SynthDataset as _H5SynthDataset

        return _H5SynthDataset
    if name == "H5SynthDataModule":
        from src.data.h5_synth_datamodule import H5SynthDataModule as _H5SynthDataModule

        return _H5SynthDataModule
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
