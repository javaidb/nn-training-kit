from abc import ABC
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


class Hyperparameter(BaseModel, ABC):
    """
    Base class for defining hyperparameters.

    Parameters
    ----------
    name : str
        The name of the hyperparameter.
    """
    name: str


class IntegerHyperparameter(Hyperparameter):
    """
    Hyperparameter for integer types.

    Parameters
    ----------
    low : int
        The lowest value in the tuning range.
    high : int
        The highest value in the tuning range.
    log : bool, optional
        Whether to sample from a log scale, by default False.
    """
    low: int
    high: int
    log: Optional[bool] = False


class FloatHyperparameter(Hyperparameter):
    """
    Hyperparameter for float types.
    
    This can be used for parameters like learning rate, accuracy_tolerance, etc.

    Parameters
    ----------
    low : float
        The lowest value in the tuning range.
    high : float
        The highest value in the tuning range.
    log : bool, optional
        Whether to sample from a log scale, by default False.
    """
    low: float
    high: float
    log: Optional[bool] = False


class CategoricalHyperparameter(Hyperparameter):
    """
    Hyperparameter for categorical types.

    Parameters
    ----------
    choices : list
        A list of choices, which can include bool, int, float, or str.
    """
    choices: list[Union[bool, int, float, str]]


class _HyperparameterTypes(str, Enum):
    """Enumeration for supported hyperparameter types."""

    integer: str = "integer"
    float: str = "float"
    categorical: str = "categorical"


def get_hyperparameter(type: str) -> Hyperparameter:
    """
    Retrieve the corresponding hyperparameter class based on type.

    Parameters
    ----------
    type : str
        The type of hyperparameter (e.g., 'integer', 'float', 'categorical').

    Returns
    -------
    Hyperparameter
        The corresponding hyperparameter class.

    Raises
    ------
    ValueError
        If the provided hyperparameter type is invalid.
    NotImplementedError
        If the hyperparameter type is not yet implemented.
    """
    # Normalize the type to lowercase for comparison
    type = type.lower()

    if type == "integer":
        return IntegerHyperparameter
    elif type == "float":
        return FloatHyperparameter
    elif type == "categorical":
        return CategoricalHyperparameter
    else:
        # Get supported types from Enum
        _supported_choices = [option.value for option in _HyperparameterTypes]
        if type not in _supported_choices:
            raise ValueError(
                f"Hyperparameter type '{type}' is not supported. Choose from: {_supported_choices}"
            )
        # Raise error if not implemented by the developer
        raise NotImplementedError(
            f"Hyperparameter type '{type}' is not implemented by the developer."
        )