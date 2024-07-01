import torch
import torch.nn as nn


def set_model_attr(model: nn.Module, attr: str, value: str):
    """Set the attribute of a model to a string.

    Args:
        model (nn.Module): PyTorch model.
        attr (str): The attribute to set.
        value (str): The value to set the attribute to.
    """
    ords = list(map(ord, value))
    model.__setattr__(attr, torch.tensor(ords, dtype=torch.uint8))


def get_model_attr(model: nn.Module, attr: str) -> str | None:
    """Get the attribute of a model as a string.

    Args:
        model (nn.Module): PyTorch model.
        attr (str): The attribute to get.

    Returns:
        str: The attribute as a string.
    """
    if not hasattr(model, attr):
        return None
    return ''.join(map(chr, model.__getattribute__(attr)))
