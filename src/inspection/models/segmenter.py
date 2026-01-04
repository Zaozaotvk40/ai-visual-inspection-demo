"""
SAM 2 (Segment Anything Model 2) based segmentation.

Provides wrapper classes for SAM 2 segmentation with
YOLO detection integration for defect mask refinement.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .types import DetectionResult


class SAMSegmenter:
    """SAM 2 based segmentation for defect refinement.

    Wraps the Segment Anything Model 2 (SAM 2) for generating precise
    segmentation masks from YOLO detection boxes.

    Uses Ultralytics SAM 2 integration for seamless YOLO+SAM workflows.

    Attributes:
        model_type: SAM 2 model variant (sam2_t, sam2_s, sam2_b, sam2_l).
        device: Compute device (cuda/cpu).

    Example:
        >>> segmenter = SAMSegmenter(model_type="sam2_b", device="cuda")
        >>> segmenter.load_model()
        >>> masks = segmenter.segment_with_boxes(image, [(10, 10, 100, 100)])
    """

    # Available SAM 2 model types
    MODEL_TYPES = {
        "sam2_t": "sam2_t.pt",   # Tiny model (fastest)
        "sam2_s": "sam2_s.pt",   # Small model
        "sam2_b": "sam2_b.pt",   # Base model
        "sam2_l": "sam2_l.pt",   # Large model (most accurate)
    }

    # Default models directory (relative to project root)
    MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"

    def __init__(
        self,
        model_type: str = "sam2_b",
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the SAM 2 segmenter.

        Args:
            model_type: SAM 2 model variant ('sam2_t', 'sam2_s', 'sam2_b', or 'sam2_l').
            checkpoint_path: Optional path to custom checkpoint.
            device: Compute device ('cuda' or 'cpu').

        Raises:
            ValueError: If model_type is not valid.
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Must be one of {list(self.MODEL_TYPES.keys())}"
            )

        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def load_model(self) -> None:
        """Load SAM 2 model.

        Uses Ultralytics SAM 2 wrapper for easy integration.
        Model weights are downloaded to models/ directory if not present.

        Raises:
            RuntimeError: If model loading fails.
        """
        from ultralytics import SAM

        # Ensure models directory exists
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Use custom checkpoint or default model in models/ directory
        if self.checkpoint_path:
            model_path = str(self.checkpoint_path)
        else:
            model_path = str(self.MODELS_DIR / self.MODEL_TYPES[self.model_type])

        self._model = SAM(model_path)

    def segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """Generate segmentation masks from bounding boxes.

        Uses bounding boxes as prompts to generate precise masks.
        This is the primary method for YOLO+SAM integration.

        Args:
            image: Input image as numpy array (RGB format).
            boxes: List of bounding boxes (x1, y1, x2, y2).

        Returns:
            List of binary masks, one per bounding box.
            Each mask has shape (H, W) with values 0 or 1.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not boxes:
            return []

        # Run SAM with bounding box prompts
        results = self._model.predict(
            image,
            bboxes=boxes,
            verbose=False,
        )

        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks:
                    # Convert mask to binary numpy array
                    mask_array = mask.data.cpu().numpy().squeeze()
                    masks.append((mask_array > 0.5).astype(np.uint8))

        return masks

    def segment_with_points(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> list[np.ndarray]:
        """Generate segmentation masks from point prompts.

        Args:
            image: Input image as numpy array.
            points: List of (x, y) point coordinates.
            labels: List of labels (1 for foreground, 0 for background).

        Returns:
            List of binary masks.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not points:
            return []

        results = self._model.predict(
            image,
            points=points,
            labels=labels,
            verbose=False,
        )

        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks:
                    mask_array = mask.data.cpu().numpy().squeeze()
                    masks.append((mask_array > 0.5).astype(np.uint8))

        return masks

    def segment_everything(self, image: np.ndarray) -> list[np.ndarray]:
        """Segment all objects in the image automatically.

        Generates masks for all detected objects without prompts.

        Args:
            image: Input image as numpy array.

        Returns:
            List of binary masks for all detected objects.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self._model.predict(
            image,
            verbose=False,
        )

        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks:
                    mask_array = mask.data.cpu().numpy().squeeze()
                    masks.append((mask_array > 0.5).astype(np.uint8))

        return masks


class DefectSegmentationRefiner:
    """Refines YOLO detections with SAM segmentation.

    Combines YOLO detection boxes with SAM to produce
    precise segmentation masks for each detected defect.

    Attributes:
        segmenter: SAM segmenter instance.

    Example:
        >>> segmenter = SAMSegmenter(model_type="vit_b")
        >>> segmenter.load_model()
        >>> refiner = DefectSegmentationRefiner(segmenter)
        >>> refined = refiner.refine_detections(image, detections)
    """

    def __init__(self, segmenter: SAMSegmenter) -> None:
        """Initialize the refinement module.

        Args:
            segmenter: Loaded SAM segmenter instance.
        """
        self.segmenter = segmenter

    def refine_detections(
        self,
        image: np.ndarray,
        detections: list[DetectionResult],
    ) -> list[DetectionResult]:
        """Add segmentation masks to detection results.

        Takes YOLO detection results and uses their bounding boxes
        to generate precise SAM masks.

        Args:
            image: Original image as numpy array.
            detections: List of detection results from YOLO.

        Returns:
            List of detection results with masks added.
        """
        if not detections:
            return detections

        # Extract bounding boxes
        boxes = [det.bbox for det in detections]

        # Generate masks
        masks = self.segmenter.segment_with_boxes(image, boxes)

        # Create new detections with masks
        refined_detections = []
        for i, det in enumerate(detections):
            mask = masks[i] if i < len(masks) else None
            refined_det = DetectionResult(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                mask=mask,
            )
            refined_detections.append(refined_det)

        return refined_detections

    def calculate_mask_area(self, mask: np.ndarray) -> int:
        """Calculate the area of a binary mask.

        Args:
            mask: Binary mask array.

        Returns:
            Number of pixels in the mask.
        """
        return int(np.sum(mask > 0))

    def calculate_mask_iou(
        self, mask1: np.ndarray, mask2: np.ndarray
    ) -> float:
        """Calculate IoU between two masks.

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.

        Returns:
            Intersection over Union score (0-1).
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)
