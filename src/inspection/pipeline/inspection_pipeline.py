"""
Inspection pipeline for visual defect detection.

Provides an integrated pipeline that combines image loading, detection,
optional segmentation refinement, and severity assessment.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..config.settings import Settings, get_settings
from ..models.detector import YOLODefectDetector
from ..models.segmenter import DefectSegmentationRefiner, SAMSegmenter
from ..models.types import DetectionResult, InspectionResult, Severity
from ..preprocessing.image_loader import ImageLoader


@dataclass
class PipelineConfig:
    """Configuration for the inspection pipeline.

    Attributes:
        detector_model_path: Path to YOLOv8 model weights.
        sam_model_type: SAM 2 model variant to use.
        use_sam_refinement: Whether to use SAM 2 for mask refinement.
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        warning_threshold: Threshold for WARNING severity.
        ng_threshold: Threshold for NG severity.
        device: Compute device (cuda/cpu/auto).
        image_size: Target image size for inference.
    """

    detector_model_path: Path
    sam_model_type: str = "sam2_b"
    use_sam_refinement: bool = True
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    warning_threshold: float = 0.3
    ng_threshold: float = 0.7
    device: str = "auto"
    image_size: int = 640

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "PipelineConfig":
        """Create config from application settings.

        Args:
            settings: Settings instance. If None, uses default settings.

        Returns:
            PipelineConfig instance.
        """
        if settings is None:
            settings = get_settings()

        return cls(
            detector_model_path=settings.model_path,
            sam_model_type=settings.sam_model_type,
            use_sam_refinement=settings.use_sam,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            warning_threshold=settings.warning_threshold,
            ng_threshold=settings.ng_threshold,
            device=settings.device,
            image_size=settings.image_size,
        )


class InspectionPipeline:
    """Integrated visual inspection pipeline.

    Combines all components for end-to-end defect detection:
    1. Image loading and preprocessing
    2. YOLOv8 defect detection
    3. (Optional) SAM segmentation refinement
    4. Severity assessment

    Attributes:
        config: Pipeline configuration.
        detector: YOLOv8 detector instance.
        image_loader: Image loader instance.
        is_initialized: Whether pipeline is ready for inference.

    Example:
        >>> config = PipelineConfig(detector_model_path=Path("yolov8n.pt"))
        >>> pipeline = InspectionPipeline(config)
        >>> pipeline.initialize()
        >>> result = pipeline.inspect(Path("test_image.jpg"))
        >>> print(f"Severity: {result.severity}, Defects: {result.defect_count}")
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the inspection pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self._detector: Optional[YOLODefectDetector] = None
        self._image_loader: Optional[ImageLoader] = None
        self._segmenter: Optional[SAMSegmenter] = None
        self._refiner: Optional[DefectSegmentationRefiner] = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized and ready."""
        return self._is_initialized

    @property
    def detector(self) -> YOLODefectDetector:
        """Get the detector instance."""
        if self._detector is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._detector

    @property
    def image_loader(self) -> ImageLoader:
        """Get the image loader instance."""
        if self._image_loader is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._image_loader

    def initialize(self) -> None:
        """Initialize all pipeline components.

        Loads models and prepares components for inference.
        Must be called before using inspect() methods.

        Raises:
            RuntimeError: If initialization fails.
        """
        # Resolve device
        device = self._resolve_device()

        # Initialize image loader
        self._image_loader = ImageLoader(
            target_size=(self.config.image_size, self.config.image_size),
            maintain_aspect=True,
        )

        # Initialize detector
        self._detector = YOLODefectDetector(
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            device=device,
        )
        self._detector.load_model(self.config.detector_model_path)

        # Initialize SAM for segmentation refinement
        if self.config.use_sam_refinement:
            self._segmenter = SAMSegmenter(
                model_type=self.config.sam_model_type,
                device=device,
            )
            self._segmenter.load_model()
            self._refiner = DefectSegmentationRefiner(self._segmenter)

        self._is_initialized = True

    def _resolve_device(self) -> str:
        """Resolve the compute device to use.

        Returns:
            Device string ('cuda' or 'cpu').
        """
        if self.config.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def inspect(self, image_path: Union[str, Path]) -> InspectionResult:
        """Inspect a single image for defects.

        Args:
            image_path: Path to the image file.

        Returns:
            InspectionResult containing detection results and severity.

        Raises:
            RuntimeError: If pipeline is not initialized.
            FileNotFoundError: If image file doesn't exist.
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.time()

        # Load and preprocess image
        image = self.image_loader.load(image_path)

        # Run detection
        detections = self.detector.detect(image)

        # Apply SAM refinement if enabled
        if self.config.use_sam_refinement and self._refiner and detections:
            detections = self._refiner.refine_detections(image, detections)

        # Determine severity
        severity = self._determine_severity(detections)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        return InspectionResult(
            image_path=str(image_path),
            is_defective=len(detections) > 0,
            detections=detections,
            processing_time_ms=processing_time,
            severity=severity,
        )

    def inspect_image(self, image: np.ndarray, image_name: str = "image") -> InspectionResult:
        """Inspect a numpy array image for defects.

        Useful for processing images already in memory.

        Args:
            image: Image as numpy array (RGB format).
            image_name: Name identifier for the image.

        Returns:
            InspectionResult containing detection results and severity.
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.time()

        # Run detection directly on the image
        detections = self.detector.detect(image)

        # Apply SAM refinement if enabled
        if self.config.use_sam_refinement and self._refiner and detections:
            detections = self._refiner.refine_detections(image, detections)

        # Determine severity
        severity = self._determine_severity(detections)

        processing_time = (time.time() - start_time) * 1000

        return InspectionResult(
            image_path=image_name,
            is_defective=len(detections) > 0,
            detections=detections,
            processing_time_ms=processing_time,
            severity=severity,
        )

    def inspect_batch(
        self, image_paths: list[Union[str, Path]]
    ) -> list[InspectionResult]:
        """Inspect multiple images.

        Args:
            image_paths: List of paths to image files.

        Returns:
            List of InspectionResult, one per image.
        """
        return [self.inspect(path) for path in image_paths]

    def _determine_severity(self, detections: list[DetectionResult]) -> Severity:
        """Determine overall severity based on detections.

        Severity levels:
        - OK: No defects or all detections below warning threshold
        - WARNING: At least one detection above warning but below NG
        - NG: At least one detection above NG threshold

        Args:
            detections: List of detection results.

        Returns:
            Severity enum value.
        """
        if not detections:
            return Severity.OK

        max_confidence = max(det.confidence for det in detections)

        if max_confidence >= self.config.ng_threshold:
            return Severity.NG
        elif max_confidence >= self.config.warning_threshold:
            return Severity.WARNING
        else:
            return Severity.OK

    def get_statistics(self, results: list[InspectionResult]) -> dict:
        """Calculate statistics from multiple inspection results.

        Args:
            results: List of inspection results.

        Returns:
            Dictionary containing statistics.
        """
        if not results:
            return {
                "total": 0,
                "defective": 0,
                "ok": 0,
                "warning": 0,
                "ng": 0,
                "defect_rate": 0.0,
                "avg_processing_time_ms": 0.0,
            }

        total = len(results)
        defective = sum(1 for r in results if r.is_defective)
        severity_counts = {s: 0 for s in Severity}
        for r in results:
            severity_counts[r.severity] += 1

        avg_time = sum(r.processing_time_ms for r in results) / total

        return {
            "total": total,
            "defective": defective,
            "ok": severity_counts[Severity.OK],
            "warning": severity_counts[Severity.WARNING],
            "ng": severity_counts[Severity.NG],
            "defect_rate": defective / total if total > 0 else 0.0,
            "avg_processing_time_ms": avg_time,
        }
