# app/utils/data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
from typing import Tuple, List, Dict
import requests
from tqdm import tqdm


class QuickDrawDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 categories: List[str],
                 transform=None,
                 train: bool = True,
                 download: bool = False):
        """
        Args:
            root_dir: Root directory for the dataset
            categories: List of categories to load
            transform: Optional transforms to apply
            train: If True, creates training dataset, else test dataset
            download: If True, downloads data if not present
        """
        self.root_dir = Path(root_dir)
        self.categories = categories
        self.transform = transform
        self.train = train

        # Create directory if it doesn't exist
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if download:
            self._download_data()

        self.data, self.targets = self._load_data()

    def _download_data(self):
        """Download QuickDraw dataset if not present"""
        base_url = "https://storage.cloud.google.com/quickdraw_dataset/full/simplified/The%20Eiffel%20Tower.ndjson"

        for category in tqdm(self.categories, desc="Downloading categories"):
            filename = f"{category}.npy"
            filepath = self.root_dir / filename

            if not filepath.exists():
                url = f"{base_url}/{filename}"
                response = requests.get(url)
                with open(filepath, 'wb') as f:
                    f.write(response.content)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data"""
        data_list = []
        target_list = []

        for idx, category in enumerate(self.categories):
            filepath = self.root_dir / f"{category}.npy"
            if not filepath.exists():
                raise FileNotFoundError(f"No data file found for {category}")

            # Load the category data
            category_data = np.load(filepath)

            # Split into train/test
            split_idx = int(len(category_data) * 0.9)
            if self.train:
                category_data = category_data[:split_idx]
            else:
                category_data = category_data[split_idx:]

            # Reshape to 28x28 and normalize
            category_data = category_data.reshape(-1,
                                                  28, 28).astype(np.float32) / 255.0

            data_list.append(category_data)
            target_list.append(np.full(len(category_data), idx))

        return (np.concatenate(data_list),
                np.concatenate(target_list))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.data[idx]
        target = self.targets[idx]

        # Add channel dimension and convert to tensor
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, target


def get_dataloaders(root_dir: str,
                    categories: List[str],
                    batch_size: int = 32,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing DataLoaders.

    Args:
        root_dir: Directory containing the dataset
        categories: List of categories to use
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = QuickDrawDataset(
        root_dir=root_dir,
        categories=categories,
        train=True,
        download=True
    )

    test_dataset = QuickDrawDataset(
        root_dir=root_dir,
        categories=categories,
        train=False,
        download=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
