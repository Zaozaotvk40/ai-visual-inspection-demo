"""
Manufacturing Visual Inspection AI Package.

This package provides tools for automated visual inspection
using YOLOv8 for defect detection and SAM 2 for segmentation.

Example:
    >>> from inspection import InspectionPipeline, PipelineConfig
    >>> config = PipelineConfig(detector_model_path="yolov8n.pt")
    >>> pipeline = InspectionPipeline(config)
    >>> pipeline.initialize()
    >>> result = pipeline.inspect("image.jpg")
    >>> print(f"Severity: {result.severity}")
"""

__version__ = "0.1.0"

# Main pipeline
from .pipeline import InspectionPipeline, PipelineConfig

# Models
from .models import (
    DefectClass,
    DetectionResult,
    InspectionResult,
    SAMSegmenter,
    Severity,
    YOLODefectDetector,
)

# Preprocessing
from .preprocessing import ImageLoader

# Visualization
from .visualization import ResultVisualizer

# Configuration
from .config import Settings, get_settings

__all__ = [
    # Pipeline
    "InspectionPipeline",
    "PipelineConfig",
    # Models
    "DefectClass",
    "DetectionResult",
    "InspectionResult",
    "SAMSegmenter",
    "Severity",
    "YOLODefectDetector",
    # Preprocessing
    "ImageLoader",
    # Visualization
    "ResultVisualizer",
    # Configuration
    "Settings",
    "get_settings",
]
