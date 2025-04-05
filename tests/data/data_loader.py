"""
Data loader for testing the nn-training-kit library.

This module provides functions to load sample data for testing.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_tensor_data(data_dir: str, prefix: str = "linear") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load data from PyTorch tensor files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data files.
    prefix : str, optional
        Prefix of the data files, by default "linear".
        
    Returns
    -------
    X : torch.Tensor
        Input features.
    y : torch.Tensor
        Target values.
    """
    X_path = os.path.join(data_dir, f"{prefix}_X.pt")
    y_path = os.path.join(data_dir, f"{prefix}_y.pt")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    X = torch.load(X_path)
    y = torch.load(y_path)
    
    return X, y


def load_csv_data(data_dir: str, prefix: str = "linear") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load data from CSV files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data files.
    prefix : str, optional
        Prefix of the data files, by default "linear".
        
    Returns
    -------
    X : torch.Tensor
        Input features.
    y : torch.Tensor
        Target values.
    """
    X_path = os.path.join(data_dir, f"{prefix}_X.csv")
    y_path = os.path.join(data_dir, f"{prefix}_y.csv")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)
    
    X = torch.tensor(X_df.values, dtype=torch.float32)
    y = torch.tensor(y_df.values, dtype=torch.float32)
    
    return X, y


def get_data_paths(data_type: str = "linear") -> Tuple[str, str, str]:
    """
    Get paths to train, validation, and test data.
    
    Parameters
    ----------
    data_type : str, optional
        Type of data to load, by default "linear".
        
    Returns
    -------
    train_path : str
        Path to training data directory.
    val_path : str
        Path to validation data directory.
    test_path : str
        Path to test data directory.
    """
    base_dir = Path(__file__).parent
    
    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "val")
    test_path = os.path.join(base_dir, "test")
    
    return train_path, val_path, test_path 