import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Optional, Tuple, List, Union, Any
import lightning as L
from sklearn.preprocessing import StandardScaler

class CSVDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        target_column: str,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.scaler = StandardScaler()
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
    def setup(self, stage: Optional[str] = None):
        # Read data
        df = pd.read_csv(self.csv_path)
        
        # Split features and target
        X = df.drop(columns=[self.target_column]).values
        y = df[self.target_column].values.reshape(-1, 1)
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_normalized)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset
        dataset = CSVDataset(X_tensor, y_tensor)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Store input dimension for model configuration
        self.input_dim = X.shape[1]
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        ) 