# Neural Network Training Kit

A PyTorch Lightning-based toolkit for streamlined neural network training with hyperparameter tuning capabilities.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [License](#license)

## Introduction

The Neural Network Training Kit provides a structured approach to training neural networks using PyTorch Lightning. It simplifies the training process by handling common tasks like loss function selection, optimizer configuration, and hyperparameter tuning.

## Features

- **Training Module**: A PyTorch Lightning module that handles training, validation, and testing steps
- **Configuration System**: YAML-based configuration for model, data, trainer, and optimizer settings
- **Hyperparameter Tuning**: Support for hyperparameter optimization with various search strategies
- **Accuracy Tolerance**: Configurable accuracy tolerance for model evaluation
- **Logging**: Integrated logging for tracking training metrics and artifacts
- **Sample Data**: Synthetic data generation for testing and demonstration

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/nn-training-kit.git
cd nn-training-kit
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
import torch
import torch.nn as nn
from nn_training_kit.core.training import TrainingModule

# Create a simple model
model = nn.Linear(10, 1)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create training module
training_module = TrainingModule(
    model=model,
    loss_function=loss_fn,
    optimizer=optimizer,
    accuracy_tolerance=0.01
)

# Use with PyTorch Lightning Trainer
from pytorch_lightning import Trainer
trainer = Trainer(max_epochs=10)
trainer.fit(training_module, train_dataloader, val_dataloader)
```

### Using Configuration Files

```python
from nn_training_kit.core.config_loader import load_config

# Load configuration from YAML
config = load_config("config.yaml")

# Create model, loss function, and optimizer from config
model = create_model_from_config(config.model)
loss_fn = config.trainer.loss_function
optimizer = create_optimizer_from_config(config.optimizer, model.parameters())

# Create training module with config
training_module = TrainingModule(
    model=model,
    loss_function=loss_fn,
    optimizer=optimizer,
    accuracy_tolerance=config.trainer.accuracy_tolerance
)
```

## Configuration

The toolkit supports YAML-based configuration files with the following structure:

```yaml
experiment: "my_experiment"
num_trials: 3
max_epochs: 10

model:
  model:
    type: "mlp"
    input_size: 10
    hidden_sizes: [64, 32]
    output_size: 1

data_module:
  data_module:
    batch_size: 32
    num_workers: 0
    train_data_path: "./data/train"
    val_data_path: "./data/val"
    test_data_path: "./data/test"

trainer:
  loss_function: "mse"
  accuracy_tolerance: 0.01

optimizer:
  optimizer_algorithm: "adam"
  lr: 0.001
```

## Testing

The repository includes a comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_training.py
```

Sample data for testing can be generated using:

```bash
python tests/data/generate_sample_data.py
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
