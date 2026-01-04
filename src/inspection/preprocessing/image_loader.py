"""
Image loading and preprocessing utilities.

Provides consistent image loading and preprocessing for the inspection pipeline.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image


class ImageLoader:
    """Image loader with preprocessing capabilities.

    Handles image loading from various sources and applies consistent
    preprocessing for model inference.

    Attributes:
        target_size: Target size for resizing (width, height).
        maintain_aspect: Whether to maintain aspect ratio when resizing.

    Example:
        >>> loader = ImageLoader(target_size=(640, 640))
        >>> image = loader.load(Path("image.jpg"))
        >>> print(image.shape)
        (640, 640, 3)
    """

    def __init__(
        self,
        target_size: Optional[tuple[int, int]] = (640, 640),
        maintain_aspect: bool = True,
    ) -> None:
        """Initialize the image loader.

        Args:
            target_size: Target size (width, height) for resizing.
                If None, images are not resized.
            maintain_aspect: Whether to maintain aspect ratio when resizing.
                If True, image is padded to target size.
        """
        self.target_size = target_size
        self.maintain_aspect = maintain_aspect

    def load(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file.

        Reads image file and converts to RGB format.

        Args:
            image_path: Path to image file.

        Returns:
            Image as numpy array in RGB format with shape (H, W, 3).

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load image using OpenCV (faster than PIL for most formats)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply resize if target_size is specified
        if self.target_size is not None:
            image = self._resize(image, self.target_size)

        return image

    def load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes.

        Useful for processing images from HTTP requests or memory.

        Args:
            image_bytes: Image data as bytes.

        Returns:
            Image as numpy array in RGB format.

        Raises:
            ValueError: If bytes cannot be decoded as image.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image from bytes")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.target_size is not None:
            image = self._resize(image, self.target_size)

        return image

    def load_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array.

        Args:
            pil_image: PIL Image object.

        Returns:
            Image as numpy array in RGB format.
        """
        # Ensure RGB mode
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        image = np.array(pil_image)

        if self.target_size is not None:
            image = self._resize(image, self.target_size)

        return image

    def _resize(
        self,
        image: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Resize image to target size.

        Args:
            image: Input image array.
            target_size: Target size (width, height).

        Returns:
            Resized image array.
        """
        if self.maintain_aspect:
            return self._resize_with_padding(image, target_size)
        else:
            return cv2.resize(
                image, target_size, interpolation=cv2.INTER_LINEAR
            )

    def _resize_with_padding(
        self,
        image: np.ndarray,
        target_size: tuple[int, int],
        pad_color: int = 114,
    ) -> np.ndarray:
        """Resize image while maintaining aspect ratio with padding.

        The image is scaled to fit within target_size while maintaining
        its aspect ratio. The remaining space is filled with pad_color.

        Args:
            image: Input image array with shape (H, W, 3).
            target_size: Target size (width, height).
            pad_color: Grayscale value for padding (0-255).

        Returns:
            Resized and padded image with exact target_size dimensions.
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scale factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # Create padded image
        padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

        # Calculate offset to center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Place resized image in center
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            resized
        )

        return padded

    def get_original_size(self, image_path: Union[str, Path]) -> tuple[int, int]:
        """Get original image dimensions without loading full image.

        Args:
            image_path: Path to image file.

        Returns:
            Tuple of (width, height).
        """
        path = Path(image_path)
        with Image.open(path) as img:
            return img.size  # Returns (width, height)
