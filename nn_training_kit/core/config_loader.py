import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from nn_training_kit.core.training_config import process_user_config, TrainingConfig


def load_config(config_path: str) -> TrainingConfig:
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
        
    Returns
    -------
    TrainingConfig
        The processed configuration object.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return process_user_config(config_dict)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary.
    """
    return {
        "experiment": {
            "experiment_name": "default_experiment",
            "task": "default_task",
            "dataset": "default_dataset"
        },
        "run_name": None,
        "artifact_path": "./artifacts",
        "max_time": 999_999,
        "include_date_in_run_name": False,
        "model": {
            "type": "linear",
            "input_size": 10,
            "output_size": 1
        },
        "data_module": {
            "batch_size": 32,
            "num_workers": 0,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        },
        "trainer": {
            "max_epochs": 10,
            "loss_function": "mse",
            "accuracy_tolerance": 0.01,
            "hyperparameter_tuning": {
                "enabled": True,
                "n_trials": 10
            }
        },
        "optimizer": {
            "optimizer_algorithm": "adam",
            "lr": 0.001
        }
    }


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save.
    output_path : str
        Path where to save the YAML file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False) 