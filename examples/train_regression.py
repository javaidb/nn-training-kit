import os
import yaml
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from nn_training_kit.core.training_config import process_user_config
from nn_training_kit.core.data_module import DataModule
from nn_training_kit.core.models import MLP
from nn_training_kit.core.training import TrainingModule
from nn_training_kit.core.loss import MSE
from nn_training_kit.core.hyperparameter_tuning import run_hyperparameter_tuning
from torch import optim

def main():
    # Load config
    try:
        with open("examples/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Create data module
    try:
        data_module = DataModule(
            csv_path=config["data_module"]["csv_path"],
            target_column=config["data_module"]["target_column"],
            batch_size=config["data_module"]["batch_size"],
            num_workers=config["data_module"]["num_workers"]
        )
        
        # Setup data module to get input dimensions
        data_module.setup()
        print(f"Data loaded successfully. Input dimensions: {data_module.input_dim}")
    except Exception as e:
        print(f"Error setting up data module: {e}")
        return
    
    # Create model
    try:
        # Don't create an instance, just pass the class
        config["model"]["model"] = MLP
        # Add input_dim to the model configuration
        config["model"]["input_dim"] = data_module.input_dim
        print("Model class set successfully")
    except Exception as e:
        print(f"Error setting model class: {e}")
        return
    
    # Update config with data module
    config["data_module"]["data_module"] = data_module
    
    # Process config
    try:
        training_config = process_user_config(config)
        print("Config processed successfully")
    except Exception as e:
        print(f"Error processing config: {e}")
        return
    
    
    study = run_hyperparameter_tuning(training_config=training_config)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 