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
import ssl
import certifi
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress only the single warning from urllib3 needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class QuickDrawDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 categories: List[str],
                 transform=None,
                 train: bool = True,
                 download: bool = False,
                 sample_size: int = 10000):  # Limit samples per category for testing
        """
        Args:
            root_dir: Root directory for the dataset
            categories: List of categories to load
            transform: Optional transforms to apply
            train: If True, creates training dataset, else test dataset
            download: If True, downloads data if not present
            sample_size: Number of samples to use per category
        """
        self.root_dir = Path(root_dir)
        self.categories = categories
        self.transform = transform
        self.train = train
        self.sample_size = sample_size

        # Create directory if it doesn't exist
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if download:
            self._download_data()

        self.data, self.targets = self._load_data()

    def _download_data(self):
        """Download QuickDraw dataset if not present"""
        base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

        # Create session with custom SSL context
        session = requests.Session()
        session.verify = certifi.where()

        for category in tqdm(self.categories, desc="Downloading categories"):
            filename = f"{category}.npy"
            filepath = self.root_dir / filename

            if not filepath.exists():
                try:
                    url = f"{base_url}/{filename}"
                    response = session.get(url, stream=True)
                    response.raise_for_status()

                    # Get total file size for progress bar
                    total_size = int(response.headers.get('content-length', 0))

                    with open(filepath, 'wb') as f:
                        with tqdm(total=total_size, unit='iB', unit_scale=True,
                                  desc=f"Downloading {category}") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                pbar.update(size)

                    logger.info(f"Successfully downloaded {category}")

                except Exception as e:
                    logger.error(f"Error downloading {category}: {str(e)}")
                    if filepath.exists():
                        filepath.unlink()  # Remove partial download

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data"""
        data_list = []
        target_list = []

        for idx, category in enumerate(self.categories):
            filepath = self.root_dir / f"{category}.npy"
            if not filepath.exists():
                logger.warning(
                    f"No data file found for {category}, skipping...")
                continue

            try:
                # Load the category data
                category_data = np.load(filepath)

                # Limit samples per category
                if len(category_data) > self.sample_size:
                    indices = np.random.choice(
                        len(category_data),
                        self.sample_size,
                        replace=False
                    )
                    category_data = category_data[indices]

                # Split into train/test
                split_idx = int(len(category_data) * 0.9)
                if self.train:
                    category_data = category_data[:split_idx]
                else:
                    category_data = category_data[split_idx:]

                # Reshape to 28x28 and normalize
                category_data = category_data.reshape(
                    -1, 28, 28).astype(np.float32) / 255.0

                data_list.append(category_data)
                target_list.append(np.full(len(category_data), idx))

                logger.info(
                    f"Successfully loaded {category} with {len(category_data)} samples")

            except Exception as e:
                logger.error(f"Error loading {category}: {str(e)}")

        if not data_list:
            raise RuntimeError("No data could be loaded from any category")

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
    """
    try:
        # Create datasets with smaller sample size for testing
        train_dataset = QuickDrawDataset(
            root_dir=root_dir,
            categories=categories,
            train=True,
            download=True,
            sample_size=10000  # Limit samples for testing
        )

        test_dataset = QuickDrawDataset(
            root_dir=root_dir,
            categories=categories,
            train=False,
            download=False,
            sample_size=10000  # Limit samples for testing
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

    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        raise
