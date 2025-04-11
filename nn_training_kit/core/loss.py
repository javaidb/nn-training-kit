from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor, nn


class LossFunction(nn.Module, ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class RMSE(LossFunction):
    """Root mean square error loss function."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Check for NaN values
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            print("\nWARNING: NaN values detected in RMSE loss inputs")
            # Replace NaN values with zeros
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
            y_true = torch.nan_to_num(y_true, nan=0.0)
        
        # Add epsilon to prevent sqrt of zero
        mse_loss_fn = nn.MSELoss()
        mse = mse_loss_fn(y_pred, y_true)
        rmse = torch.sqrt(mse + self.eps)
        return rmse


class MAE(LossFunction):
    """Mean absolute error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Check for NaN values
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            print("\nWARNING: NaN values detected in MAE loss inputs")
            # Replace NaN values with zeros
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
            y_true = torch.nan_to_num(y_true, nan=0.0)
        
        return torch.mean(torch.abs(y_pred - y_true))


class MSE(LossFunction):
    """Mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Check for NaN values
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            print("\nWARNING: NaN values detected in MSE loss inputs")
            # Replace NaN values with zeros
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
            y_true = torch.nan_to_num(y_true, nan=0.0)
        
        # Clip values to prevent overflow
        y_pred = torch.clamp(y_pred, min=-1e6, max=1e6)
        y_true = torch.clamp(y_true, min=-1e6, max=1e6)
        
        mse_loss_fn = nn.MSELoss()
        return mse_loss_fn(y_pred, y_true)


class _LossFunctionChoices(str, Enum):
    """Supported loss function options used for data validation."""

    rmse: str = "rmse"
    mae: str = "mae"
    mse: str = "mse"
    weighted_rmse: str = "weighted_rmse"


def get_loss_function(name: str) -> LossFunction:
    """
    Get loss fcuntion class.

    Parameters
    ----------
    name : str
        Name of loss function

    Returns
    -------
    LossFunction
        Loss function

    Raises
    ------
    ValueError
        If the name of loss function is not a valid option.
    NotImplementedError
        If the loss function is not implemented. This is a development error.
    """
    name = name.lower()
    
    if name == "rmse":
        return RMSE()
    elif name == "mae":
        return MAE()
    elif name == "mse":
        return MSE()
    else:
        _supported_choices = [option.value for option in _LossFunctionChoices]
        if name not in _supported_choices:
            raise ValueError(
                f"{name} loss function not supported. Please select from {_supported_choices}"
            )

        raise NotImplementedError(
            f"{name} loss function not implemented by developer."
        )