from typing import Any, Dict, Optional

import pytorch_lightning as pl
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


class TrainingModule(pl.LightningModule):
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
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: optim.Optimizer,
        accuracy_tolerance: Optional[float] = 0.01,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.accuracy_tolerance = accuracy_tolerance

    def calculate_accuracy(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate percentage accuracy within a certain voltage tolerance.
        
        Parameters
        ----------
        y_true : Tensor
            The true values (ground truth).
        y_pred : Tensor
            The predicted values from the model.
            
        Returns
        -------
        Tensor
            The calculated accuracy as a percentage.
        """
        # Calculate absolute error in actual voltage space
        abs_error = torch.abs(y_true - y_pred)
        # Calculate percentage of predictions within tolerance
        accuracy = torch.mean((abs_error <= self.accuracy_tolerance).float()) * 100
        return accuracy

    def _step(self, batch: Any, batch_idx: int, mode: str) -> Dict[str, Any]:
        """
        A common method for handling training, validation, and test steps.

        Parameters
        ----------
        batch : Any
            The input batch containing features and labels.
        batch_idx : int
            The index of the current batch.
        mode : str
            The mode of the step, can be 'training', 'validation', or 'test'.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the loss value and step output.
        """
        X, Y = batch
        Y_pred = self.model(X)

        step_loss = self.loss_fn(Y, Y_pred)
        step_accuracy = self.calculate_accuracy(Y, Y_pred)

        self.log(
            f"{mode}_loss",
            step_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            f"{mode}_accuracy",
            step_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        
        step_output = StepOutput(
            loss=step_loss,
            true_output=Y,
            model_output=Y_pred
        )

        return {
            "loss": step_loss,
            "step_output": step_output,
        }

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Handles a single training step."""
        return self._step(batch, batch_idx, mode="training")

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Handles a single validation step."""
        return self._step(batch, batch_idx, mode="validation")

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Handles a single test step."""
        return self._step(batch, batch_idx, mode="test")

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer for the training process."""
        return self.optimizer
