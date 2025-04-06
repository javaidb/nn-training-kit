import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import lightning as L
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

class ProcessedDataset(Dataset):
    """Dataset class that handles CSV data loading and preprocessing.
    
    This class supports preprocessing steps including:
    - Loading data from CSV files
    - Handling missing values
    - Normalizing/standardizing features
    - Converting data to PyTorch tensors
    """

    def __init__(
        self, 
        data_directory: str,
        features_file: Optional[str] = None,
        targets_file: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the dataset with preprocessing options.
        
        Args:
            data_directory: Directory containing the CSV data files
            features_file: Name of the CSV file containing features (X)
            targets_file: Name of the CSV file containing targets (y)
            preprocessing_config: Dictionary with preprocessing options:
                - 'normalization': 'standard', 'minmax', 'robust', or None
                - 'missing_values': 'mean', 'median', 'most_frequent', or None
                - 'feature_columns': List of column names for features
                - 'target_columns': List of column names for targets
        """
        self.data_directory = Path(data_directory)
        self.features_file = features_file
        self.targets_file = targets_file
        self.preprocessing_config = preprocessing_config or {}
        
        # Initialize preprocessing objects
        self.feature_scaler = None
        self.target_scaler = None
        self.imputer = None
        
        # Load and preprocess data
        self.X, self.y = self._load_data()
        self._preprocess_data()
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from CSV files."""
        # If specific files are not provided, try to find them
        if not self.features_file or not self.targets_file:
            files = list(self.data_directory.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {self.data_directory}")
            
            # Try to find feature and target files
            feature_files = [f for f in files if "X" in f.name or "features" in f.name]
            target_files = [f for f in files if "y" in f.name or "targets" in f.name]
            
            if feature_files and target_files:
                self.features_file = feature_files[0].name
                self.targets_file = target_files[0].name
            else:
                # If we can't find specific files, try to load a single file
                if len(files) == 1:
                    return self._load_single_file(files[0])
                else:
                    raise ValueError("Could not determine feature and target files")
        
        # Load features
        features_path = self.data_directory / self.features_file
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        # Load targets
        targets_path = self.data_directory / self.targets_file
        if not targets_path.exists():
            raise FileNotFoundError(f"Targets file not found: {targets_path}")
        
        # Load CSV files
        X = self._load_csv(features_path)
        y = self._load_csv(targets_path)
        
        return X, y
    
    def _load_single_file(self, file_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single CSV file that contains both features and targets."""
        df = pd.read_csv(file_path)
        
        # Try to determine feature and target columns
        feature_cols = self.preprocessing_config.get('feature_columns', None)
        target_cols = self.preprocessing_config.get('target_columns', None)
        
        if not feature_cols or not target_cols:
            # Assume last column is target if not specified
            if not target_cols:
                target_cols = [df.columns[-1]]
            if not feature_cols:
                feature_cols = [col for col in df.columns if col not in target_cols]
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def _load_csv(self, file_path: Path) -> torch.Tensor:
        """Load data from a CSV file."""
        df = pd.read_csv(file_path)
        
        # If specific columns are specified, use them
        columns = self.preprocessing_config.get('feature_columns', None)
        if columns:
            df = df[columns]
        
        return torch.tensor(df.values, dtype=torch.float32)
    
    def _preprocess_data(self) -> None:
        """Apply preprocessing steps to the data."""
        # Handle missing values
        if self.preprocessing_config.get('missing_values'):
            strategy = self.preprocessing_config['missing_values']
            self.imputer = SimpleImputer(strategy=strategy)
            
            # Convert to numpy for imputation
            X_np = self.X.numpy() if isinstance(self.X, torch.Tensor) else self.X
            
            # Impute missing values
            X_imputed = self.imputer.fit_transform(X_np)
            self.X = torch.tensor(X_imputed, dtype=torch.float32)
        
        # Normalize/standardize features
        normalization = self.preprocessing_config.get('normalization')
        if normalization:
            if normalization == 'standard':
                self.feature_scaler = StandardScaler()
            elif normalization == 'minmax':
                self.feature_scaler = MinMaxScaler()
            elif normalization == 'robust':
                self.feature_scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported normalization method: {normalization}")
            
            # Convert to numpy for scaling
            X_np = self.X.numpy() if isinstance(self.X, torch.Tensor) else self.X
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X_np)
            self.X = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Normalize targets if specified
        target_normalization = self.preprocessing_config.get('target_normalization')
        if target_normalization:
            if target_normalization == 'standard':
                self.target_scaler = StandardScaler()
            elif target_normalization == 'minmax':
                self.target_scaler = MinMaxScaler()
            elif target_normalization == 'robust':
                self.target_scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported target normalization method: {target_normalization}")
            
            # Convert to numpy for scaling
            y_np = self.y.numpy() if isinstance(self.y, torch.Tensor) else self.y
            
            # Scale targets
            y_scaled = self.target_scaler.fit_transform(y_np)
            self.y = torch.tensor(y_scaled, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset."""
        return self.X[idx], self.y[idx]
    
    def inverse_transform_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Transform scaled targets back to original scale."""
        if self.target_scaler is None:
            return y
        
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        y_original = self.target_scaler.inverse_transform(y_np)
        return torch.tensor(y_original, dtype=torch.float32)


class DataModule(L.LightningDataModule):
    """Data module that splits dataset into train, validation and test."""

    def __init__(
        self,
        data_directory: str,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        features_file: Optional[str] = None,
        targets_file: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the data module.
        
        Args:
            data_directory: Directory containing the CSV data files
            train_split: Proportion of data to use for training
            val_split: Proportion of data to use for validation
            test_split: Proportion of data to use for testing
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            features_file: Name of the CSV file containing features (X)
            targets_file: Name of the CSV file containing targets (y)
            preprocessing_config: Dictionary with preprocessing options
        """
        super().__init__()

        if round(sum([train_split, val_split, test_split]), 6) != 1:
            raise ValueError("All of train/val/test splits must sum up to 1.")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features_file = features_file
        self.targets_file = targets_file
        self.preprocessing_config = preprocessing_config or {}

        self.dataset = ProcessedDataset(
            data_directory=data_directory,
            features_file=features_file,
            targets_file=targets_file,
            preprocessing_config=preprocessing_config
        )

    def setup(self, stage: str) -> None:
        """Set up the dataset splits."""
        num_dataset = len(self.dataset)
        num_training_set = round(self.train_split * num_dataset)
        num_validation_set = round(self.val_split * num_dataset)
        num_test_set = num_dataset - num_training_set - num_validation_set  # Ensure we use all data

        self.training_set, self.validation_set, self.test_set = random_split(
            self.dataset, [num_training_set, num_validation_set, num_test_set]
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        return DataLoader(
            self.training_set, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""
        return DataLoader(
            self.validation_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test data loader."""
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def inverse_transform_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Transform scaled targets back to original scale."""
        return self.dataset.inverse_transform_targets(y)