"""
Application settings using Pydantic Settings.

Provides centralized configuration management with environment variable support.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.

    All settings can be overridden via environment variables.
    Example: MODEL_PATH=/path/to/model.pt

    Attributes:
        model_path: Path to the YOLOv8 model weights.
        sam_model_type: SAM model variant to use.
        device: Compute device (cuda/cpu).
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        use_sam: Whether to use SAM for segmentation refinement.
        warning_threshold: Confidence threshold for WARNING severity.
        ng_threshold: Confidence threshold for NG severity.
        image_size: Target image size for inference.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model settings
    model_path: Path = Field(
        default=Path("models/yolov8n.pt"),
        description="Path to YOLOv8 model weights",
    )
    sam_model_type: Literal["sam2_t", "sam2_s", "sam2_b", "sam2_l"] = Field(
        default="sam2_b",
        description="SAM 2 model variant (sam2_t is fastest, sam2_l is most accurate)",
    )

    # Device settings
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Compute device for inference",
    )

    # Detection thresholds
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for detections",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for Non-Maximum Suppression",
    )

    # SAM settings
    use_sam: bool = Field(
        default=True,
        description="Whether to use SAM for segmentation refinement",
    )

    # Severity thresholds
    warning_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for WARNING severity",
    )
    ng_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for NG severity",
    )

    # Image settings
    image_size: int = Field(
        default=640,
        ge=32,
        description="Target image size for inference",
    )

    def get_device(self) -> str:
        """Get the actual device to use.

        If device is 'auto', automatically detect CUDA availability.

        Returns:
            Device string ('cuda' or 'cpu').
        """
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses LRU cache to ensure only one Settings instance is created.

    Returns:
        Settings instance.
    """
    return Settings()
