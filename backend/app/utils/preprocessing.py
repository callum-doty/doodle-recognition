# app/utils/preprocessing.py

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
import cv2


class DoodlePreprocessor:
    def __init__(self, image_size: Tuple[int, int] = (28, 28)):
        self.image_size = image_size

    def process_drawing(self, points: List[List[float]]) -> torch.Tensor:
        """
        Process drawing points into a tensor ready for model input.

        Args:
            points: List of [x, y] coordinates from the drawing

        Returns:
            torch.Tensor: Processed image tensor of shape (1, 1, 28, 28)
        """
        # Create blank canvas (white background)
        canvas_size = 256
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

        if points:
            # Convert points to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw lines connecting points
            for i in range(len(points) - 1):
                cv2.line(
                    canvas,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    color=0,  # Black lines
                    thickness=3
                )

        # Resize to target size
        canvas = cv2.resize(canvas, self.image_size)

        # Normalize to [0, 1]
        canvas = canvas.astype(np.float32) / 255.0

        # Convert to tensor and add batch and channel dimensions
        tensor = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)

        return tensor
