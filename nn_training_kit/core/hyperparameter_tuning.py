from typing import Any, Callable

import lightning as L
import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import Callback
from optuna.trial import Trial
from torch import Tensor
import mlflow
import math

from nn_training_kit.core.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from nn_training_kit.core.logging import Logger
from nn_training_kit.core.training import TrainingModule
from nn_training_kit.core.training_config import TrainingConfig


def define_module(trial: Trial, module: Any, module_init_params: dict) -> Any:
    """
    Instantiate module with user defined parameters. Handles hyperparameters per
    Optuna's documentations.

    Parameters
    ----------
    trial : Trial
        Optuna's trial object
    module : Any
        Module class
    module_init_params : dict
        Parameters used for module instantiation

    Returns
    -------
    Any
        Instantiated module
    """
    # Check if module is a class or an instance
    if hasattr(module, '__name__'):
        print(f"\nDefining module: {module.__name__}")
    else:
        print(f"\nDefining module: {module.__class__.__name__}")
        # If module is already an instance, return it
        return module

    def is_hyperparameter(input_type: Any) -> bool:
        """Returns whether input is a hyperparameter."""
        return input_type in suggest_functions.keys()

    suggest_functions = {
        IntegerHyperparameter: trial.suggest_int,
        FloatHyperparameter: trial.suggest_float,
        CategoricalHyperparameter: trial.suggest_categorical,
    }
    processed_init_params = {}

    for module_arg, value in module_init_params.items():
        if value == module:
            continue

        value_type = type(value)
        if is_hyperparameter(value_type):
            suggest_fn = suggest_functions.get(value_type)
            processed_init_params[module_arg] = suggest_fn(**dict(value))
            print(f"Hyperparameter {module_arg}: {processed_init_params[module_arg]}")
        else:
            processed_init_params[module_arg] = value
            print(f"Fixed parameter {module_arg}: {value}")

    return module(**processed_init_params)


def initialize_trial(
    training_config: TrainingConfig, trial: Trial, callbacks: list[Callback] = []
) -> tuple[pl.Trainer, TrainingModule, L.LightningDataModule]:
    """
    Initialize model, data module, training module, and trainer for a trial.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    trial : Trial
        Optuna trial
    callbacks : list[Callback], optional
        List of Lightning callbacks, by default []

    Returns
    -------
    tuple[pl.Trainer, TrainingModule, L.LightningDataModule]
        Initialized trainer, training module, and data module
    """
    print("\n=== Initializing Trial Components ===")

    # Init model
    print("Initializing model...")
    model_config = training_config.model
    model_class = model_config.model
    model_init_params = dict(model_config)
    model = define_module(
        trial=trial, module=model_class, module_init_params=model_init_params
    )
    print(f"Model initialized: {model.__class__.__name__}")

    # Init data module
    print("\nInitializing data module...")
    data_module_config = training_config.data_module
    data_module_class = data_module_config.data_module
    data_module_init_params = dict(data_module_config)
    data_module = define_module(
        trial=trial,
        module=data_module_class,
        module_init_params=data_module_init_params,
    )
    data_module.setup()  # Set up the data module
    print(f"Data module initialized: {data_module.__class__.__name__}")

    # Init optimizer
    print("\nInitializing optimizer...")
    optimizer_config = training_config.optimizer
    
    # If optimizer_config is None, create a default optimizer
    if optimizer_config is None:
        from torch import optim
        print("No optimizer config found, creating default optimizer")
        
        # Default to Adam optimizer
        optimizer_class = optim.Adam
        lr = 0.001
        
        # Try to get parameters from deprecated trainer config (backward compatibility)
        if hasattr(training_config.trainer, 'optimizer') and training_config.trainer.optimizer:
            print("WARNING: Using deprecated 'optimizer' from trainer config")
            optimizer_name = training_config.trainer.optimizer.lower()
            if optimizer_name == 'adam':
                optimizer_class = optim.Adam
            elif optimizer_name == 'sgd':
                optimizer_class = optim.SGD
            elif optimizer_name == 'rmsprop':
                optimizer_class = optim.RMSprop
            else:
                print(f"Unknown optimizer: {optimizer_name}, defaulting to Adam")
        
        if hasattr(training_config.trainer, 'learning_rate') and training_config.trainer.learning_rate:
            print("WARNING: Using deprecated 'learning_rate' from trainer config")
            lr = training_config.trainer.learning_rate
            
        optimizer = optimizer_class(model.parameters(), lr=lr)
        print(f"Created default optimizer: {optimizer_class.__name__} with learning rate: {lr}")
    else:
        # Use optimizer config as before
        optimizer_class = optimizer_config.optimizer_algorithm
        optimizer_init_params = dict(optimizer_config)
        optimizer_init_params.update({"params": model.parameters()})
        optimizer = define_module(
            trial=trial,
            module=optimizer_class,
            module_init_params=optimizer_init_params,
        )
    
    print(f"Optimizer initialized: {optimizer.__class__.__name__}")

    # Init training module
    print("\nInitializing training module...")
    training_module = TrainingModule(
        model=model,
        loss_function=training_config.trainer.loss_function,
        optimizer=optimizer,
    )
    print("Training module initialized successfully")

    # Init trainer
    print("\nInitializing PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=training_config.trainer.max_epochs,
        max_time={"minutes": training_config.max_time},
        logger=True,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    print("Trainer initialized successfully")

    return trainer, training_module, data_module


def get_objective_function(training_config: TrainingConfig, logger: Logger) -> Callable:
    """
    Get objective function for Optuna study.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    logger : Logger
        Hyperparameter tuning logger

    Returns
    -------
    Callable
        Objective function
    """
    print("\n=== Creating Objective Function ===")

    def objective(trial: Trial) -> Tensor:
        """Objective function for Optuna study."""
        print(f"\n=== Starting Trial {trial.number} ===")

        with logger.start_trial_logs(trial=trial):
            print("Initializing trial components...")
            trainer, training_module, data_module = initialize_trial(
                training_config=training_config, trial=trial, callbacks=[logger]
            )
            print("Trial components initialized successfully")

            print("\n=== Starting Training ===")
            trainer.fit(training_module, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
            
            # Debug logging for metrics
            print("\n=== Debug: Available Metrics ===")
            for key, value in trainer.logged_metrics.items():
                print(f"{key}: {value}")
                # Store metrics in trial's user_attrs
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    trial.set_user_attr(f"metric:{key}", float(value))
            
            # First try to get validation accuracy
            validation_accuracy = trainer.logged_metrics.get("val_accuracy_epoch")
            if validation_accuracy is not None and not math.isnan(float(validation_accuracy)):
                print(f"\nTraining completed. Using validation accuracy: {validation_accuracy}")
                return validation_accuracy
            
            # Fall back to validation loss (negated so higher is better)
            validation_loss = trainer.logged_metrics.get("val_loss_epoch")
            if validation_loss is not None and not math.isnan(float(validation_loss)):
                print(f"WARNING: Validation accuracy not available. Using negative validation loss: {-validation_loss}")
                return -float(validation_loss)  # Negate so higher is better
            
            # Fall back to training accuracy
            train_accuracy = trainer.logged_metrics.get("train_accuracy_epoch")
            if train_accuracy is not None and not math.isnan(float(train_accuracy)):
                print(f"WARNING: Validation metrics not available. Using training accuracy: {train_accuracy}")
                return train_accuracy
            
            # Last resort: training loss (negated so higher is better)
            train_loss = trainer.logged_metrics.get("train_loss_epoch")
            if train_loss is not None and not math.isnan(float(train_loss)):
                print(f"WARNING: No validation metrics and no training accuracy. Using negative training loss: {-train_loss}")
                return -float(train_loss)  # Negate so higher is better
            
            # If we still can't find any useful metric, return a small improvement over -inf
            print("ERROR: No usable metrics found. Returning -1000 as fallback.")
            return -1000.0

    return objective


def test_best_trial(
    training_config: TrainingConfig, logger: Logger, best_trial: Trial
) -> None:
    """
    Fit and test best trial from hyperparameter tuning study.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    logger : Logger
        Hyperparameter tuning logger
    best_trial : Trial
        Optuna's best trial
    """

    with logger.start_best_trial_logs(trial=best_trial):
        trainer, training_module, data_module = initialize_trial(
            training_config=training_config, trial=best_trial, callbacks=[logger]
        )
        trainer.fit(training_module, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        
        # Log all available metrics for the best trial
        print("\n=== Best Trial Metrics ===")
        for key, value in trainer.logged_metrics.items():
            print(f"{key}: {value}")
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                mlflow.log_metric(key, float(value))
        
        logger.log_model(model=trainer.model)

        test_results = trainer.test(training_module, dataloaders=data_module.test_dataloader())
        
        # Log test metrics
        if test_results and len(test_results) > 0:
            for key, value in test_results[0].items():
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    mlflow.log_metric(key, float(value))

    return


def run_hyperparameter_tuning(training_config: TrainingConfig) -> None:
    """
    Start hyperparameter tuning using Optuna framework and log training using MLflow
    framework.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    """
    # Check if hyperparameter tuning is enabled
    hyperparameter_tuning_config = None
    
    # Check if hyperparameter tuning config exists in trainer
    if hasattr(training_config.trainer, 'hyperparameter_tuning') and training_config.trainer.hyperparameter_tuning:
        hyperparameter_tuning_config = training_config.trainer.hyperparameter_tuning
    # Check if it exists at root level (for backward compatibility)
    elif hasattr(training_config, 'hyperparameter_tuning') and training_config.hyperparameter_tuning:
        hyperparameter_tuning_config = training_config.hyperparameter_tuning
    
    # If config exists and tuning is disabled, return early
    if hyperparameter_tuning_config and not hyperparameter_tuning_config.enabled:
        print("\n=== Hyperparameter Tuning Disabled ===")
        return
    
    # Determine number of trials
    if hyperparameter_tuning_config and hasattr(hyperparameter_tuning_config, 'n_trials'):
        n_trials = hyperparameter_tuning_config.n_trials
        print(f"Using {n_trials} trials from hyperparameter_tuning configuration")
    else:
        # Fall back to num_trials in trainer config or default to 1
        n_trials = getattr(training_config.trainer, 'num_trials', 1)
        print(f"Using {n_trials} trials from trainer.num_trials (fallback)")
    
    print("\n=== Starting Hyperparameter Tuning ===")
    print(f"Experiment name: {training_config.experiment}")
    print(f"Number of trials: {n_trials}")
    print(f"Max epochs per trial: {training_config.trainer.max_epochs}")
    print(f"Max time per trial: {training_config.max_time} minutes")
    print(f"Include date in run name: {training_config.include_date_in_run_name}")

    logger = Logger(
        experiment=training_config.experiment,
        run_name=training_config.run_name,
        experiment_tags=training_config.experiment_tags,
        include_date_in_run_name=training_config.include_date_in_run_name,
    )
    print(f"Created MLflow logger with experiment ID: {logger.experiment_id}")
    print(f"Run name: {logger.run_name}")

    print("\n=== Setting up Objective Function ===")
    objective = get_objective_function(training_config=training_config, logger=logger)
    print("Objective function created successfully")

    print("\n=== Starting Optuna Study ===")
    with logger.start_hyperparameter_tuning_logs():
        study = optuna.create_study(direction="maximize")  # Changed to maximize for accuracy
        print("Created Optuna study with 'maximize' direction")
        
        print("\n=== Running Optimization ===")
        study.optimize(objective, n_trials=n_trials)
        print(f"Optimization completed. Best trial value: {study.best_value}")
        
        # Log the best trial's parameters and value
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_trial_value", study.best_value)
        
        # Log all metrics from trials 
        for trial in study.trials:
            if trial.state.is_finished() and hasattr(trial, 'user_attrs'):
                for key, value in trial.user_attrs.items():
                    if key.startswith('metric:'):
                        metric_name = key.split(':', 1)[1]
                        mlflow.log_metric(f"trial_{trial.number}_{metric_name}", value)
        
        print("\n=== Testing Best Trial ===")
        best_trial_metrics = {}
        
        # Create a function to collect metrics from the best trial
        def collect_best_trial_metrics(trainer, data_module):
            nonlocal best_trial_metrics
            # Collect all metrics for each epoch and step
            for epoch in range(training_config.trainer.max_epochs):
                # Run training and collect metrics
                for batch_idx, batch in enumerate(data_module.train_dataloader()):
                    trainer.train_loop.run_training_batch(batch, batch_idx)
                    step = epoch * len(data_module.train_dataloader()) + batch_idx
                    
                    # Log step metrics to the parent run
                    train_loss_step = trainer.logged_metrics.get("train_loss_step")
                    train_accuracy_step = trainer.logged_metrics.get("train_accuracy_step")
                    val_loss_step = trainer.logged_metrics.get("val_loss_step")
                    val_accuracy_step = trainer.logged_metrics.get("val_accuracy_step")
                    
                    if train_loss_step is not None and not math.isnan(float(train_loss_step)):
                        mlflow.log_metric("train_loss_step", float(train_loss_step), step=step)
                    if train_accuracy_step is not None and not math.isnan(float(train_accuracy_step)):
                        mlflow.log_metric("train_accuracy_step", float(train_accuracy_step), step=step)
                    if val_loss_step is not None and not math.isnan(float(val_loss_step)):
                        mlflow.log_metric("val_loss_step", float(val_loss_step), step=step)
                    if val_accuracy_step is not None and not math.isnan(float(val_accuracy_step)):
                        mlflow.log_metric("val_accuracy_step", float(val_accuracy_step), step=step)
                    mlflow.log_metric("step", step, step=step)
                
                # Log epoch metrics
                train_loss_epoch = trainer.logged_metrics.get("train_loss_epoch")
                train_accuracy_epoch = trainer.logged_metrics.get("train_accuracy_epoch")
                val_loss_epoch = trainer.logged_metrics.get("val_loss_epoch")
                val_accuracy_epoch = trainer.logged_metrics.get("val_accuracy_epoch")
                
                if train_loss_epoch is not None and not math.isnan(float(train_loss_epoch)):
                    mlflow.log_metric("train_loss_epoch", float(train_loss_epoch), step=epoch)
                if train_accuracy_epoch is not None and not math.isnan(float(train_accuracy_epoch)):
                    mlflow.log_metric("train_accuracy_epoch", float(train_accuracy_epoch), step=epoch)
                if val_loss_epoch is not None and not math.isnan(float(val_loss_epoch)):
                    mlflow.log_metric("val_loss_epoch", float(val_loss_epoch), step=epoch)
                if val_accuracy_epoch is not None and not math.isnan(float(val_accuracy_epoch)):
                    mlflow.log_metric("val_accuracy_epoch", float(val_accuracy_epoch), step=epoch)
                mlflow.log_metric("epoch", epoch, step=epoch)
        
        # Run the best trial and log metrics directly to the parent run
        test_best_trial(
            training_config=training_config, 
            logger=logger, 
            best_trial=study.best_trial
        )
        
        # Copy metrics from the best trial to the parent run
        metric_client = mlflow.tracking.MlflowClient()
        best_trial_run_id = None
        
        # Find the best_trial run ID
        runs = metric_client.search_runs(experiment_ids=[logger.experiment_id], filter_string="tags.mlflow.runName = 'best_trial'")
        if runs:
            best_trial_run_id = runs[0].info.run_id
            
            if best_trial_run_id:
                # Get all metrics from the best trial run
                run = metric_client.get_run(best_trial_run_id)
                
                # First collect all step metrics to determine max steps
                step_metrics = {}
                for key, value in run.data.metrics.items():
                    if key.endswith('_step'):
                        if key not in step_metrics:
                            step_metrics[key] = []
                        step_metrics[key].append((key, float(value)))
                
                # Then log all metrics with proper steps
                for key, value in run.data.metrics.items():
                    if '_step' in key:
                        # For step metrics, we need to log with the step value
                        step_value = int(key.split('_step_')[1]) if '_step_' in key else 0
                        mlflow.log_metric(key, float(value), step=step_value)
                    elif '_epoch' in key:
                        # For epoch metrics, log with the epoch number
                        epoch_value = int(key.split('_epoch_')[1]) if '_epoch_' in key else 0
                        mlflow.log_metric(key, float(value), step=epoch_value)
                    else:
                        # For other metrics, just log the value
                        mlflow.log_metric(key, float(value))
                    
                    print(f"Logged metric from best trial: {key} = {value}")
                
                # Additionally, fetch history for step-level metrics
                for metric_key in ["train_loss_step", "train_accuracy_step", "val_loss_step", "val_accuracy_step", 
                                 "test_loss_step", "test_accuracy_step", "step"]:
                    try:
                        print(f"Logging history metric: {metric_key}")
                        # Get metric history
                        metric_history = metric_client.get_metric_history(best_trial_run_id, metric_key)
                        
                        # Log each point in the history
                        for metric in metric_history:
                            # Log to parent run with the proper step
                            mlflow.log_metric(
                                key=metric_key,
                                value=metric.value,
                                step=metric.step,
                                timestamp=metric.timestamp
                            )
                        print(f"Reference log (step {metric.step}): {metric_key} = {metric.value}")
                    except Exception as e:
                        print(f"Error getting history for {metric_key}: {e}")
        
        print("Best trial testing completed")

    print("\n=== Hyperparameter Tuning Completed ===")
    return study