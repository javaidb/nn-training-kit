import os
import yaml
import lightning as L
from typing import Dict, Any, Optional, Union
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from nn_training_kit.core.training_config import process_user_config, TrainingConfig
from nn_training_kit.core.data_module import DataModule
from nn_training_kit.core.models import MLP, get_model_by_name
from nn_training_kit.core.training import TrainingModule
from nn_training_kit.core.loss import MSE
from nn_training_kit.core.hyperparameter_tuning import run_hyperparameter_tuning
from torch import optim
import optuna


def train_from_config(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    model_class: Optional[Any] = None,
    data_module: Optional[DataModule] = None,
    verbose: bool = True
) -> optuna.Study:
    """
    Train a model using configuration from a YAML file or a dictionary.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. If provided, this will be used instead of the config parameter.
    config : Dict[str, Any], optional
        Configuration dictionary. If config_path is provided, this will be ignored.
    model_class : Any, optional
        Model class to use. If not provided, will use the model specified in the config.
    data_module : DataModule, optional
        Pre-configured data module. If not provided, will create one from the config.
    verbose : bool, default=True
        Whether to print verbose output.
        
    Returns
    -------
    optuna.Study
        The hyperparameter tuning study object.
    """
    # Load config from file if provided
    if config_path:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")
    
    if config is None:
        raise ValueError("Either config_path or config must be provided")
    
    # Extract experiment details
    experiment_config = config.get("experiment", {})
    experiment_name = experiment_config.get("experiment_name", "default_experiment")
    experiment_tags = {
        "task": experiment_config.get("task", "unknown"),
        "dataset": experiment_config.get("dataset", "unknown")
    }
    
    # Update config with experiment details in the format expected by process_user_config
    config["experiment"] = experiment_name
    config["experiment_tags"] = experiment_tags
    
    # Create data module if not provided
    if data_module is None:
        try:
            data_module = DataModule(
                csv_path=config["data_module"]["csv_path"],
                target_column=config["data_module"]["target_column"],
                batch_size=config["data_module"]["batch_size"],
                num_workers=config["data_module"]["num_workers"],
                train_ratio=config["data_module"].get("train_ratio", 0.7),
                val_ratio=config["data_module"].get("val_ratio", 0.15),
                test_ratio=config["data_module"].get("test_ratio", 0.15)
            )
            
            # Setup data module to get input dimensions
            data_module.setup()
            if verbose:
                print(f"Data loaded successfully. Input dimensions: {data_module.input_dim}")
                
                # Debug validation/test datasets
                print(f"Training dataset size: {len(data_module.train_dataset) if hasattr(data_module, 'train_dataset') else 'Unknown'}")
                print(f"Validation dataset size: {len(data_module.val_dataset) if hasattr(data_module, 'val_dataset') else 'Unknown'}")
                print(f"Test dataset size: {len(data_module.test_dataset) if hasattr(data_module, 'test_dataset') else 'Unknown'}")
        except Exception as e:
            raise ValueError(f"Error setting up data module: {e}")
    
    # Configure model
    try:
        # If model_class is provided, use it
        if model_class is not None:
            config["model"]["model_name"] = model_class
            if verbose:
                print(f"Using provided model class: {model_class.__name__}")
        # Otherwise, use the model from config or default to MLP
        elif "model_name" not in config["model"]:
            config["model"]["model_name"] = MLP
            if verbose:
                print("Using default MLP model class")
        
        # Add input_dim to the model configuration
        config["model"]["input_dim"] = data_module.input_dim
        if verbose:
            print("Model configuration set successfully")
    except Exception as e:
        raise ValueError(f"Error setting model configuration: {e}")
    
    # Update config with data module
    config["data_module"]["data_module"] = data_module
    
    # Process config
    try:
        training_config = process_user_config(config)
        if verbose:
            print("Config processed successfully")
    except Exception as e:
        raise ValueError(f"Error processing config: {e}")
    
    # Run hyperparameter tuning
    study = run_hyperparameter_tuning(training_config=training_config)
    
    if verbose:
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    
    return study


def train_model(
    model_class: Any,
    data_module: DataModule,
    experiment_name: str = "default_experiment",
    experiment_tags: Optional[Dict[str, str]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    trainer_params: Optional[Dict[str, Any]] = None,
    optimizer_params: Optional[Dict[str, Any]] = None,
    hyperparameter_tuning: bool = True,
    n_trials: int = 10,
    verbose: bool = True
) -> optuna.Study:
    """
    Train a model with a simplified interface.
    
    Parameters
    ----------
    model_class : Any
        Model class to use.
    data_module : DataModule
        Pre-configured data module.
    experiment_name : str, default="default_experiment"
        Name of the experiment.
    experiment_tags : Dict[str, str], optional
        Tags for the experiment.
    model_params : Dict[str, Any], optional
        Parameters for the model.
    trainer_params : Dict[str, Any], optional
        Parameters for the trainer.
    optimizer_params : Dict[str, Any], optional
        Parameters for the optimizer.
    hyperparameter_tuning : bool, default=True
        Whether to perform hyperparameter tuning.
    n_trials : int, default=10
        Number of trials for hyperparameter tuning.
    verbose : bool, default=True
        Whether to print verbose output.
        
    Returns
    -------
    optuna.Study
        The hyperparameter tuning study object.
    """
    # Setup data module if not already set up
    if not hasattr(data_module, 'input_dim'):
        data_module.setup()
        if verbose:
            print(f"Data loaded successfully. Input dimensions: {data_module.input_dim}")
    
    # Create config
    config = {
        "experiment": experiment_name,
        "experiment_tags": experiment_tags or {},
        "model": {
            "model_name": model_class,
            "input_dim": data_module.input_dim,
            **(model_params or {})
        },
        "data_module": {
            "data_module": data_module
        },
        "trainer": {
            "loss_function": "mse",
            "accuracy_tolerance": 0.01,
            **(trainer_params or {})
        },
        "optimizer": {
            "optimizer_algorithm": "adam",
            "lr": 0.001,
            **(optimizer_params or {})
        }
    }
    
    # Add hyperparameter tuning if enabled
    if hyperparameter_tuning:
        config["trainer"]["hyperparameter_tuning"] = {
            "enabled": True,
            "n_trials": n_trials
        }
    
    # Process config
    try:
        training_config = process_user_config(config)
        if verbose:
            print("Config processed successfully")
    except Exception as e:
        raise ValueError(f"Error processing config: {e}")
    
    # Run hyperparameter tuning
    study = run_hyperparameter_tuning(training_config=training_config)
    
    if verbose:
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    
    return study 