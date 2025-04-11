from copy import copy
from typing import Any, Optional, Union, Dict

import pytorch_lightning as L
from pydantic import AfterValidator, BaseModel
from torch import nn
from typing_extensions import Annotated

from nn_training_kit.core.hyperparameter import Hyperparameter, get_hyperparameter
from nn_training_kit.core.loss import _LossFunctionChoices, get_loss_function
from nn_training_kit.core.optimizer import _OptimizerChoices, get_optimizer


class ModelConfig(BaseModel, extra="allow", arbitrary_types_allowed=True):
    """
    Model configuration base model for data validation.

    Parameters:
    -----------
    model : Union[nn.Module, Any]
        PyTorch model
    """

    model: Union[nn.Module, Any]


class DataModuleConfig(BaseModel, extra="allow", arbitrary_types_allowed=True):
    """
    Data module configuration base model for data validation.

    Parameters:
    -----------
    data_module : Union[L.LightningDataModule, Any]
        Lightning data module
    """

    data_module: Union[L.LightningDataModule, Any]


class TrainerConfig(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """
    Trainer configuration base model for data validation.

    Parameters:
    -----------
    loss_function : str
        Name of loss function
    accuracy_tolerance : Union[float, Hyperparameter], optional
        The tolerance value for calculating accuracy. Default is 0.01.
    """

    loss_function: Annotated[_LossFunctionChoices, AfterValidator(get_loss_function)]
    accuracy_tolerance: Optional[Union[float, Hyperparameter]] = 0.01


class OptimizerConfig(BaseModel, extra="allow"):
    """
    Optimizer configuration base model for data validation.

    Parameters:
    -----------
    optimizer_algorithm : str
        Optimizer algorithm
    lr : Union[float, Hyperparameter]
        Learning rate
    """

    optimizer_algorithm: Annotated[_OptimizerChoices, AfterValidator(get_optimizer)]
    lr: Union[float, Hyperparameter]


class TrainingConfig(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """
    Training run configuration base model for data validation.

    Parameters:
    -----------
    experiment : str
        Name of experiment
    num_trials: int
        Number of hyperparameter tuning trials
    experiment_tags : Dict[str, str], Optional
        Tags for experiment, by default {}
    run_name : str, Optional
        Name of run, by default None
    artifact_path : str, Optional
        Path of artifact, by default None
    max_epochs : int, Optional
        Max number of epoch per training, by default None
    max_time : float, Optional
        Max time for each training in minutes, by default None
    include_date_in_run_name : bool, Optional
        Whether to include date in run name, by default False
    model : ModelConfig
        Settings for model
    data_module : DataModuleConfig
        Settings for data module
    trainer : TrainerConfig
        Settings for trainer
    optimizer : OptimizerConfig
        Settings for optimizer
    """

    experiment: str
    num_trials: int
    max_epochs: Optional[int]
    experiment_tags: Optional[Dict[str, str]] = {}
    run_name: Optional[str] = None
    artifact_path: Optional[str] = None
    max_time: Optional[float] = 999_999  # [minutes]
    include_date_in_run_name: Optional[bool] = False

    model: ModelConfig
    data_module: DataModuleConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig


def process_user_config(user_config: dict) -> TrainingConfig:
    """
    Parses user input and prepare config for model training.

    Parameters
    ----------
    user_config : dict
        User settings for training

    Returns
    -------
    TrainingConfig
        Config for model training
    """

    def is_hyperparameter(variable_input: dict) -> bool:
        """Determines whether a variable should be treated as a hyperparameter."""

        return variable_input.get(hyperparam_type_key)

    hyperparam_type_key = "hyperparameter_type"
    config_groups = ["model", "optimizer", "trainer", "data_module"]

    processed_config = copy(user_config)

    for group in config_groups:
        for arg_name, arg_value in user_config.get(group).items():
            if isinstance(arg_value, dict) and is_hyperparameter(arg_value):
                hyperparam_type = get_hyperparameter(arg_value.get(hyperparam_type_key))
                arg_value.pop(hyperparam_type_key)

                processed_config[group][arg_name] = hyperparam_type(**arg_value)

    training_config = TrainingConfig(**processed_config)

    return training_config