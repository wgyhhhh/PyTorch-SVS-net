"""PyTorch reimplementation of the SVS-Net training pipeline."""

__all__ = ["SVSNet"]


def __getattr__(name: str):
    if name == "SVSNet":
        from .model import SVSNet

        return SVSNet
    raise AttributeError(name)
