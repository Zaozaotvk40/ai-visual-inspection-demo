"""
YOLOv8-based defect detector.

Provides wrapper classes for YOLOv8 object detection with
manufacturing defect detection capabilities.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .types import DefectClass, DetectionResult


class BaseDetector(ABC):
    """Abstract base class for defect detectors.

    Defines the interface that all detector implementations must follow.
    """

    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """Load model weights from file.

        Args:
            model_path: Path to model weights file.
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """Detect defects in an image.

        Args:
            image: Input image as numpy array (RGB format).

        Returns:
            List of detection results.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model metadata.
        """
        pass


class YOLODefectDetector(BaseDetector):
    """YOLOv8-based defect detector.

    Wraps the Ultralytics YOLOv8 model for manufacturing defect detection.
    Supports both pretrained and custom-trained models.

    Attributes:
        model: The loaded YOLOv8 model instance.
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        device: Compute device (cuda/cpu).

    Example:
        >>> detector = YOLODefectDetector(confidence_threshold=0.5)
        >>> detector.load_model(Path("models/yolov8n.pt"))
        >>> results = detector.detect(image)
        >>> for det in results:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")
    """

    # Default class mapping for defect detection
    DEFECT_CLASSES = DefectClass.get_name_mapping()

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ) -> None:
        """Initialize the YOLOv8 detector.

        Args:
            model_path: Optional path to model weights. If provided, model
                is loaded immediately.
            confidence_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for Non-Maximum Suppression.
            device: Compute device ('cuda' or 'cpu').
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        self._class_names: dict[int, str] = {}

        if model_path is not None:
            self.load_model(model_path)

    @property
    def model(self):
        """Get the loaded YOLO model.

        Returns:
            The YOLO model instance.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self._model is not None

    def load_model(self, model_path: Path) -> None:
        """Load YOLOv8 model from file.

        Supports both pretrained models (yolov8n.pt, etc.) and custom-trained
        models. For pretrained models, uses default COCO classes. For custom
        models, reads class names from the model.

        Args:
            model_path: Path to model weights (.pt file).

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)

        # Load the model
        self._model = YOLO(str(self.model_path))
        self._model.to(self.device)

        # Get class names from model
        if hasattr(self._model, "names"):
            self._class_names = self._model.names
        else:
            # Fallback to defect classes for custom models
            self._class_names = self.DEFECT_CLASSES

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """Detect defects in an image.

        Runs YOLOv8 inference on the input image and returns structured
        detection results.

        Args:
            image: Input image as numpy array. Expected shape: (H, W, 3)
                in RGB format.

        Returns:
            List of DetectionResult objects, one per detected defect.
            Empty list if no defects detected.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image format is invalid.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")

        # Run inference
        results = self._model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        # Parse results
        detections: list[DetectionResult] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Extract bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = tuple(map(int, xyxy))

                # Extract confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Get class name
                class_name = self._class_names.get(class_id, f"class_{class_id}")

                detection = DetectionResult(
                    bbox=bbox,  # type: ignore
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
                detections.append(detection)

        return detections

    def detect_batch(
        self, images: list[np.ndarray]
    ) -> list[list[DetectionResult]]:
        """Detect defects in multiple images.

        Processes images sequentially. For true batch inference,
        consider using the YOLO model's native batch capabilities.

        Args:
            images: List of input images.

        Returns:
            List of detection result lists, one per image.
        """
        return [self.detect(img) for img in images]

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary containing:
                - type: Model type (YOLOv8)
                - path: Path to model weights
                - classes: Class name mapping
                - confidence_threshold: Current confidence threshold
                - iou_threshold: Current IoU threshold
                - device: Current compute device
        """
        return {
            "type": "YOLOv8",
            "path": str(self.model_path) if self.model_path else None,
            "classes": self._class_names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "is_loaded": self.is_loaded,
        }

    def set_thresholds(
        self,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> None:
        """Update detection thresholds.

        Args:
            confidence: New confidence threshold (0-1).
            iou: New IoU threshold (0-1).

        Raises:
            ValueError: If threshold values are out of range.
        """
        if confidence is not None:
            if not 0 <= confidence <= 1:
                raise ValueError(f"Confidence must be 0-1, got {confidence}")
            self.confidence_threshold = confidence

        if iou is not None:
            if not 0 <= iou <= 1:
                raise ValueError(f"IoU must be 0-1, got {iou}")
            self.iou_threshold = iou
