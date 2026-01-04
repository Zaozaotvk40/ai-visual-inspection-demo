"""
Data types for the inspection package.

Defines core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class DefectClass(Enum):
    """Enumeration of defect types.

    Each defect type has an associated ID for model compatibility.

    Attributes:
        SCRATCH: Surface scratch defect.
        CHIP: Chipped or broken edge defect.
        CONTAMINATION: Foreign material contamination.
        CRACK: Crack or fracture defect.
        DENT: Dent or indentation defect.
        DISCOLORATION: Color anomaly defect.
    """

    SCRATCH = 0
    CHIP = 1
    CONTAMINATION = 2
    CRACK = 3
    DENT = 4
    DISCOLORATION = 5

    @classmethod
    def from_id(cls, class_id: int) -> "DefectClass":
        """Get DefectClass from class ID.

        Args:
            class_id: Integer class ID.

        Returns:
            Corresponding DefectClass enum member.

        Raises:
            ValueError: If class_id is not valid.
        """
        for member in cls:
            if member.value == class_id:
                return member
        raise ValueError(f"Invalid class_id: {class_id}")

    @classmethod
    def get_name_mapping(cls) -> dict[int, str]:
        """Get mapping from class ID to name.

        Returns:
            Dictionary mapping class IDs to lowercase names.
        """
        return {member.value: member.name.lower() for member in cls}


class Severity(Enum):
    """Inspection severity levels.

    Attributes:
        OK: No defects or low-confidence detections.
        WARNING: Medium-confidence defects requiring review.
        NG: High-confidence defects, product rejected.
    """

    OK = "OK"
    WARNING = "WARNING"
    NG = "NG"


@dataclass
class DetectionResult:
    """Result of a single defect detection.

    Represents one detected defect with its location, classification,
    and optional segmentation mask.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        confidence: Detection confidence score (0-1).
        class_id: Integer class ID.
        class_name: Human-readable class name.
        mask: Optional segmentation mask from SAM.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate detection result after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if len(self.bbox) != 4:
            raise ValueError(f"Bbox must have 4 elements, got {len(self.bbox)}")

    @property
    def area(self) -> int:
        """Calculate bounding box area.

        Returns:
            Area in pixels.
        """
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> tuple[int, int]:
        """Calculate bounding box center.

        Returns:
            Center coordinates (cx, cy).
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class InspectionResult:
    """Result of a complete inspection.

    Represents the full inspection result for a single image,
    including all detections and the overall severity assessment.

    Attributes:
        image_path: Path to the inspected image.
        is_defective: Whether any defects were detected.
        detections: List of individual detection results.
        processing_time_ms: Total processing time in milliseconds.
        severity: Overall severity assessment.
    """

    image_path: str
    is_defective: bool
    detections: list[DetectionResult]
    processing_time_ms: float
    severity: Severity

    @property
    def defect_count(self) -> int:
        """Get total number of detected defects.

        Returns:
            Number of detections.
        """
        return len(self.detections)

    @property
    def max_confidence(self) -> float:
        """Get maximum confidence among all detections.

        Returns:
            Maximum confidence score, or 0.0 if no detections.
        """
        if not self.detections:
            return 0.0
        return max(det.confidence for det in self.detections)

    def get_defects_by_class(self, class_name: str) -> list[DetectionResult]:
        """Filter detections by class name.

        Args:
            class_name: Class name to filter by.

        Returns:
            List of detections matching the class name.
        """
        return [det for det in self.detections if det.class_name == class_name]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation (without numpy arrays).
        """
        return {
            "image_path": self.image_path,
            "is_defective": self.is_defective,
            "defect_count": self.defect_count,
            "processing_time_ms": self.processing_time_ms,
            "severity": self.severity.value,
            "detections": [
                {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "area": det.area,
                }
                for det in self.detections
            ],
        }
