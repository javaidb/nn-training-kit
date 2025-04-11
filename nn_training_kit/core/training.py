from typing import Any, Dict, Optional

import lightning as L
import torch
from pydantic import BaseModel
from torch import Tensor, nn, optim


class StepOutput(BaseModel, protected_namespaces=(), arbitrary_types_allowed=True):
    """
    Represents the output of a training, validation, or test step.

    Attributes
    ----------
    loss : Tensor
        The loss value for the step (e.g., training loss, validation accuracy).
    true_output : Tensor
        The true output (labels) for the step.
    model_output : Tensor
        The model's predicted output.
    """

    loss: Tensor
    true_output: Tensor
    model_output: Tensor


class TrainingModule(L.LightningModule):
    """
    A PyTorch Lightning module that handles the training, validation, and testing steps.
    This module supports the integration of custom loss functions and optimizers.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    loss_function : nn.Module
        The loss function to compute the discrepancy between true and predicted outputs.
    optimizer : optim.Optimizer
        The optimizer used for training the model.
    accuracy_tolerance : float, optional
        The tolerance value for calculating accuracy. Default is 0.01.
    device_name : str, optional
        The device to use for training. Default is "auto".
    eps : float, optional
        Small constant for numerical stability. Default is 1e-8.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: optim.Optimizer,
        accuracy_tolerance: Optional[float] = 0.01,
        device_name: str = "auto",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.accuracy_tolerance = accuracy_tolerance
        self.eps = eps  # Small constant for numerical stability
        
        # Determine device
        if device_name == "auto":
            self._device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device_name = device_name
        
        # Move model and loss function to device
        self.model = self.model.to(self._device_name)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self._device_name)

    @property
    def device_name(self) -> str:
        """Get the device name."""
        return self._device_name

    def calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy with numerical stability measures."""
        # Ensure tensors are on the same device
        outputs = outputs.to(self._device_name)
        targets = targets.to(self._device_name)
        
        # Handle NaN values
        outputs = torch.nan_to_num(outputs, nan=0.0)
        targets = torch.nan_to_num(targets, nan=0.0)
        
        # For regression, calculate accuracy as percentage of predictions within acceptable error
        error = torch.abs(outputs - targets)
        acceptable_error = torch.std(targets) * 0.1 + self.eps  # 10% of std dev plus eps
        accuracy = torch.mean((error <= acceptable_error).float())
        
        return accuracy

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        inputs, targets = batch
        
        # Handle NaN and Inf values in inputs
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("\nWARNING: NaN/Inf values detected in training inputs")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("\nWARNING: NaN/Inf values detected in training targets")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Handle NaN/Inf in outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("\nWARNING: NaN/Inf values detected in model outputs during training")
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Calculate loss
        loss = self.loss_fn(outputs, targets)
        
        # Check for invalid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("\nWARNING: Invalid loss detected in training")
            return torch.tensor(float('inf'), device=self._device_name)
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a single validation step."""
        inputs, targets = batch
        
        # Handle NaN and Inf values in inputs
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("\nWARNING: NaN/Inf values detected in validation inputs")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("\nWARNING: NaN/Inf values detected in validation targets")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Handle NaN/Inf in outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("\nWARNING: NaN/Inf values detected in model outputs during validation")
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Calculate loss
        loss = self.loss_fn(outputs, targets)
        
        # Check for invalid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("\nWARNING: Invalid loss detected in validation")
            self.log('val_loss', float('inf'), on_step=True, on_epoch=True, prog_bar=True)
            self.log('val_accuracy', 0.0, on_step=True, on_epoch=True, prog_bar=True)
            return
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(outputs, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a single test step."""
        inputs, targets = batch
        
        # Handle NaN and Inf values in inputs
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("\nWARNING: NaN/Inf values detected in test inputs")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("\nWARNING: NaN/Inf values detected in test targets")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Handle NaN/Inf in outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("\nWARNING: NaN/Inf values detected in model outputs during testing")
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Calculate loss
        loss = self.loss_fn(outputs, targets)
        
        # Check for invalid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("\nWARNING: Invalid loss detected in testing")
            self.log('test_loss', float('inf'), on_step=True, on_epoch=True, prog_bar=True)
            self.log('test_accuracy', 0.0, on_step=True, on_epoch=True, prog_bar=True)
            return
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(outputs, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict:
        """Configure optimizers for training."""
        return {"optimizer": self.optimizer}
