import pytest
import torch
from torch import nn

from nn_training_kit.core.training import TrainingModule
from nn_training_kit.core.loss import MSE


def test_training_module():
    # Create a simple model
    model = nn.Linear(10, 1)
    loss_fn = MSE()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create training module with default accuracy tolerance
    training_module = TrainingModule(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer
    )
    
    # Create dummy batch
    batch = (torch.randn(32, 10), torch.randn(32, 1))
    
    # Test training step
    output = training_module.training_step(batch, 0)
    assert "loss" in output
    assert "step_output" in output
    assert isinstance(output["loss"], torch.Tensor)


def test_training_module_with_custom_tolerance():
    # Create a simple model
    model = nn.Linear(10, 1)
    loss_fn = MSE()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create training module with custom accuracy tolerance
    custom_tolerance = 0.05
    training_module = TrainingModule(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        accuracy_tolerance=custom_tolerance
    )
    
    # Create dummy batch
    batch = (torch.randn(32, 10), torch.randn(32, 1))
    
    # Test training step
    output = training_module.training_step(batch, 0)
    assert "loss" in output
    assert "step_output" in output
    assert isinstance(output["loss"], torch.Tensor)
    
    # Test that the custom tolerance is used
    assert training_module.accuracy_tolerance == custom_tolerance 