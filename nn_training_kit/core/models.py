import torch
from torch import nn

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout_rate: float = 0.0, clip_value: float = 1e6):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output dimensions
            dropout_rate: Dropout rate between layers
            clip_value: Maximum absolute value for outputs
        """
        super().__init__()
        
        self.clip_value = clip_value
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),  # Add BatchNorm for stability
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with numerical stability measures."""
        # Ensure input is on the same device as model
        x = x.to(self.model[0].weight.device)
        
        # Check for NaN inputs
        if torch.isnan(x).any():
            print("\nWARNING: NaN values detected in model input")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Forward pass
        out = self.model(x)
        
        # Check for and handle NaN outputs
        if torch.isnan(out).any():
            print("\nWARNING: NaN values detected in model output")
            out = torch.nan_to_num(out, nan=0.0)
        
        # Clip values to prevent extremes
        out = torch.clamp(out, min=-self.clip_value, max=self.clip_value)
        
        return out 