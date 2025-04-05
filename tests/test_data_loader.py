"""
Tests for the data loader module.
"""

import os
import pytest
import torch
from pathlib import Path

from tests.data.data_loader import load_tensor_data, load_csv_data, get_data_paths
from tests.data.generate_sample_data import main as generate_data


@pytest.fixture(scope="module")
def sample_data():
    """Generate sample data for testing."""
    # Get the path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    # Generate sample data if it doesn't exist
    if not os.path.exists(os.path.join(data_dir, "train", "linear_X.pt")):
        generate_data()
    
    return data_dir


def test_get_data_paths():
    """Test getting data paths."""
    train_path, val_path, test_path = get_data_paths()
    
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)


def test_load_tensor_data(sample_data):
    """Test loading tensor data."""
    train_path = os.path.join(sample_data, "train")
    
    # Load linear data
    X_linear, y_linear = load_tensor_data(train_path, "linear")
    assert isinstance(X_linear, torch.Tensor)
    assert isinstance(y_linear, torch.Tensor)
    assert X_linear.shape[1] == 10  # 10 features
    assert y_linear.shape[1] == 1   # 1 target
    
    # Load nonlinear data
    X_nonlinear, y_nonlinear = load_tensor_data(train_path, "nonlinear")
    assert isinstance(X_nonlinear, torch.Tensor)
    assert isinstance(y_nonlinear, torch.Tensor)
    assert X_nonlinear.shape[1] == 10  # 10 features
    assert y_nonlinear.shape[1] == 1   # 1 target


def test_load_csv_data(sample_data):
    """Test loading CSV data."""
    train_path = os.path.join(sample_data, "train")
    
    # Load linear data
    X_linear, y_linear = load_csv_data(train_path, "linear")
    assert isinstance(X_linear, torch.Tensor)
    assert isinstance(y_linear, torch.Tensor)
    assert X_linear.shape[1] == 10  # 10 features
    assert y_linear.shape[1] == 1   # 1 target
    
    # Load nonlinear data
    X_nonlinear, y_nonlinear = load_csv_data(train_path, "nonlinear")
    assert isinstance(X_nonlinear, torch.Tensor)
    assert isinstance(y_nonlinear, torch.Tensor)
    assert X_nonlinear.shape[1] == 10  # 10 features
    assert y_nonlinear.shape[1] == 1   # 1 target 