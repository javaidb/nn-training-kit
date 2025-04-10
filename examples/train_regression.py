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
        model = MLP(
            input_dim=data_module.input_dim,
            hidden_dims=config["model"]["hidden_dims"],
            output_dim=config["model"]["output_dim"],
            dropout_rate=config["model"]["dropout_rate"]
        )
        print("Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Update config with model and data module
    config["model"]["model"] = model
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

    # # Get loss function and optimizer
    # try:
    #     # Create MSE loss directly
    #     loss_function = MSE()
        
    #     # Create Adam optimizer directly
    #     optimizer = optim.Adam(
    #         model.parameters(),
    #         lr=training_config.optimizer.lr
    #     )
    #     print("Using MSE loss and Adam optimizer")
    # except Exception as e:
    #     print(f"Error setting up loss and optimizer: {e}")
    #     return
    
    # # Create training module
    # training_module = TrainingModule(
    #     model=model,
    #     loss_function=loss_function,
    #     optimizer=optimizer,
    #     accuracy_tolerance=training_config.trainer.accuracy_tolerance
    # )
    
    # # Create callbacks directly in the code
    # callbacks = [
    #     EarlyStopping(
    #         monitor='validation_loss',
    #         patience=10,
    #         mode='min'
    #     ),
    #     ModelCheckpoint(
    #         dirpath='checkpoints',
    #         filename='best_model',
    #         monitor='validation_loss',
    #         mode='min'
    #     )
    # ]
    
    # # Create logger
    # try:
    #     logger = MLFlowLogger(
    #         experiment_name=training_config.experiment,
    #         tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    #     )
    #     print("Logger created successfully")
    # except Exception as e:
    #     print(f"Error setting up logger: {e}")
    #     return
    
    # # Create trainer
    # trainer = L.Trainer(
    #     max_epochs=training_config.max_epochs,
    #     logger=logger,
    #     callbacks=callbacks,
    #     enable_progress_bar=True,
    #     enable_model_summary=True
    # )
    
    # # Train model
    # try:
    #     print("Starting training...")
    #     trainer.fit(training_module, data_module)
    #     print("Training completed successfully!")
        
    #     # Test model
    #     print("Running test set evaluation...")
    #     test_results = trainer.test(training_module, data_module)
    #     print(f"Test results: {test_results}")
        
    # except Exception as e:
    #     print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 