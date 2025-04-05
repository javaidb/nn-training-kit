"""
Generate sample data for testing the nn-training-kit library.

This script creates synthetic data that can be used for testing the library.
The data is saved in the tests/data directory.
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def generate_linear_data(num_samples=1000, input_dim=10, noise_level=0.1, seed=42):
    """
    Generate synthetic data for linear regression.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    input_dim : int
        Dimension of the input features.
    noise_level : float
        Level of noise to add to the target.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    """
    np.random.seed(seed)
    
    # Generate random weights
    weights = np.random.randn(input_dim)
    
    # Generate input features
    X = np.random.randn(num_samples, input_dim)
    
    # Generate target values with some noise
    y = np.dot(X, weights) + noise_level * np.random.randn(num_samples)
    
    return X, y


def generate_nonlinear_data(num_samples=1000, input_dim=10, noise_level=0.1, seed=42):
    """
    Generate synthetic data for nonlinear regression.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    input_dim : int
        Dimension of the input features.
    noise_level : float
        Level of noise to add to the target.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    """
    np.random.seed(seed)
    
    # Generate input features
    X = np.random.randn(num_samples, input_dim)
    
    # Generate target values with nonlinear transformation and noise
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]**2 + noise_level * np.random.randn(num_samples)
    
    return X, y


def save_data(X, y, output_dir, prefix):
    """
    Save data to CSV files.
    
    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    output_dir : str
        Directory to save the data.
    prefix : str
        Prefix for the output files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create feature columns
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create dataframes
    X_df = pd.DataFrame(X, columns=feature_cols)
    y_df = pd.DataFrame(y, columns=["target"])
    
    # Save to CSV
    X_df.to_csv(os.path.join(output_dir, f"{prefix}_X.csv"), index=False)
    y_df.to_csv(os.path.join(output_dir, f"{prefix}_y.csv"), index=False)
    
    # Save as PyTorch tensors
    torch.save(torch.tensor(X, dtype=torch.float32), os.path.join(output_dir, f"{prefix}_X.pt"))
    torch.save(torch.tensor(y, dtype=torch.float32).reshape(-1, 1), os.path.join(output_dir, f"{prefix}_y.pt"))


def main():
    """Generate and save sample data."""
    # Set paths
    base_dir = Path(__file__).parent
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"
    
    # Generate linear data
    X_train_linear, y_train_linear = generate_linear_data(num_samples=800, seed=42)
    X_val_linear, y_val_linear = generate_linear_data(num_samples=100, seed=43)
    X_test_linear, y_test_linear = generate_linear_data(num_samples=100, seed=44)
    
    # Generate nonlinear data
    X_train_nonlinear, y_train_nonlinear = generate_nonlinear_data(num_samples=800, seed=42)
    X_val_nonlinear, y_val_nonlinear = generate_nonlinear_data(num_samples=100, seed=43)
    X_test_nonlinear, y_test_nonlinear = generate_nonlinear_data(num_samples=100, seed=44)
    
    # Save linear data
    save_data(X_train_linear, y_train_linear, train_dir, "linear")
    save_data(X_val_linear, y_val_linear, val_dir, "linear")
    save_data(X_test_linear, y_test_linear, test_dir, "linear")
    
    # Save nonlinear data
    save_data(X_train_nonlinear, y_train_nonlinear, train_dir, "nonlinear")
    save_data(X_val_nonlinear, y_val_nonlinear, val_dir, "nonlinear")
    save_data(X_test_nonlinear, y_test_nonlinear, test_dir, "nonlinear")
    
    print(f"Sample data generated and saved to {base_dir}")


if __name__ == "__main__":
    main() 