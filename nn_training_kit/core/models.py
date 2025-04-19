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

class CNN(nn.Module):
    """Convolutional Neural Network for sequence or image data."""
    
    def __init__(self, input_channels: int, conv_channels: list, kernel_sizes: list, 
                 hidden_dims: list, output_dim: int, dropout_rate: float = 0.0):
        """
        Initialize the CNN.
        
        Args:
            input_channels: Number of input channels
            conv_channels: List of convolutional layer channel counts
            kernel_sizes: List of kernel sizes for each conv layer
            hidden_dims: List of fully connected layer dimensions
            output_dim: Number of output dimensions
            dropout_rate: Dropout rate between layers
        """
        super().__init__()
        
        # Build convolutional layers
        conv_layers = []
        prev_channels = input_channels
        
        for channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(prev_channels, channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout_rate)
            ])
            prev_channels = channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = conv_channels[-1]  # Last conv layer's output channels
        
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Ensure input is on the same device as model
        x = x.to(self.conv_layers[0].weight.device)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x

class RNN(nn.Module):
    """Recurrent Neural Network for sequential data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout_rate: float = 0.0, rnn_type: str = 'lstm'):
        """
        Initialize the RNN.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            num_layers: Number of RNN layers
            output_dim: Number of output dimensions
            dropout_rate: Dropout rate between layers
            rnn_type: Type of RNN ('lstm' or 'gru')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # RNN layer
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout_rate)
        else:  # Default to GRU
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Ensure input is on the same device as model
        x = x.to(self.input_layer.weight.device)
        
        # Input layer
        x = self.input_layer(x)
        x = self.dropout(x)
        
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        else:
            hidden = h0
        
        # RNN layer
        out, _ = self.rnn(x, hidden)
        
        # Output layer (use last time step)
        out = self.output_layer(out[:, -1, :])
        
        return out

def get_model_by_name(model_name: str):
    """
    Get model class by name (case-insensitive).
    
    Args:
        model_name: Name of the model class (e.g., 'mlp' for MLP)
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model_name doesn't match any available model
    """
    models = {
        'mlp': MLP,
        'cnn': CNN,
        'rnn': RNN,
        # Add more models here as they are implemented
    }
    
    # Try to match the name case-insensitively
    model_name_lower = model_name.lower() if isinstance(model_name, str) else None
    
    if model_name_lower in models:
        return models[model_name_lower]
    
    # If the model_name is already a class (not a string), return it directly
    if not isinstance(model_name, str):
        return model_name
        
    available_models = ", ".join(models.keys())
    raise ValueError(f"Unknown model name: {model_name}. Available models: {available_models}") 