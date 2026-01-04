"""
Visualization utilities for inspection results.

Provides tools for drawing detection results on images
and creating summary dashboards.
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..models.types import DetectionResult, InspectionResult, Severity


class ResultVisualizer:
    """Visualizer for inspection results.

    Draws bounding boxes, masks, and annotations on images.
    Also provides dashboard generation for batch results.

    Attributes:
        font_scale: Scale factor for text annotations.
        line_thickness: Thickness of bounding box lines.

    Example:
        >>> visualizer = ResultVisualizer()
        >>> annotated = visualizer.draw_results(image, result)
        >>> visualizer.save_image(annotated, Path("output.jpg"))
    """

    # Color scheme for severity levels (BGR format for OpenCV)
    SEVERITY_COLORS = {
        Severity.OK: (0, 255, 0),       # Green
        Severity.WARNING: (0, 165, 255),  # Orange
        Severity.NG: (0, 0, 255),        # Red
    }

    # Color scheme for defect classes (BGR format)
    DEFECT_COLORS = {
        "scratch": (255, 0, 0),       # Blue
        "chip": (0, 255, 0),          # Green
        "contamination": (0, 0, 255), # Red
        "crack": (0, 255, 255),       # Yellow
        "dent": (255, 0, 255),        # Magenta
        "discoloration": (255, 255, 0), # Cyan
    }

    # Default color for unknown classes
    DEFAULT_COLOR = (128, 128, 128)  # Gray

    def __init__(
        self,
        font_scale: float = 0.5,
        line_thickness: int = 2,
    ) -> None:
        """Initialize the visualizer.

        Args:
            font_scale: Scale factor for text annotations.
            line_thickness: Thickness of bounding box lines.
        """
        self.font_scale = font_scale
        self.line_thickness = line_thickness

    def draw_results(
        self,
        image: np.ndarray,
        result: InspectionResult,
        show_masks: bool = True,
        show_labels: bool = True,
        show_severity: bool = True,
        show_time: bool = True,
    ) -> np.ndarray:
        """Draw detection results on an image.

        Args:
            image: Input image (RGB format).
            result: Inspection result to visualize.
            show_masks: Whether to draw segmentation masks.
            show_labels: Whether to show class labels and confidence.
            show_severity: Whether to show severity indicator.
            show_time: Whether to show processing time.

        Returns:
            Annotated image (RGB format).
        """
        # Create a copy to avoid modifying the original
        output = image.copy()

        # Draw each detection
        for det in result.detections:
            color = self.DEFECT_COLORS.get(det.class_name, self.DEFAULT_COLOR)

            # Draw mask if available
            if show_masks and det.mask is not None:
                output = self._draw_mask(output, det.mask, color)

            # Draw bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.line_thickness)

            # Draw label
            if show_labels:
                label = f"{det.class_name}: {det.confidence:.2f}"
                self._draw_label(output, label, (x1, y1 - 10), color)

        # Draw severity indicator
        if show_severity:
            self._draw_severity_indicator(output, result.severity)

        # Draw processing time
        if show_time:
            time_text = f"{result.processing_time_ms:.1f}ms"
            self._draw_label(output, time_text, (10, output.shape[0] - 20), (255, 255, 255))

        return output

    def _draw_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Draw a semi-transparent mask overlay.

        Args:
            image: Input image.
            mask: Binary mask array.
            color: RGB color for the mask.
            alpha: Transparency level (0-1).

        Returns:
            Image with mask overlay.
        """
        # Create colored overlay
        overlay = image.copy()

        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Apply color to mask region
        overlay[mask > 0.5] = color

        # Blend with original image
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw text label with background.

        Args:
            image: Image to draw on.
            text: Text to display.
            position: (x, y) position for text.
            color: Text color.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, self.font_scale, 1
        )

        x, y = position

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x, y - text_h - baseline),
            (x + text_w, y + baseline),
            color,
            -1,  # Filled
        )

        # Draw text (white on colored background)
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            self.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_severity_indicator(
        self,
        image: np.ndarray,
        severity: Severity,
    ) -> None:
        """Draw severity indicator in corner.

        Args:
            image: Image to draw on.
            severity: Severity level to display.
        """
        color = self.SEVERITY_COLORS[severity]

        # Draw filled rectangle
        cv2.rectangle(image, (10, 10), (120, 50), color, -1)

        # Draw severity text
        cv2.putText(
            image,
            severity.value,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def save_image(
        self,
        image: np.ndarray,
        path: Path,
    ) -> None:
        """Save image to file.

        Args:
            image: Image to save (RGB format).
            path: Output file path.
        """
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

    def create_comparison(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
    ) -> np.ndarray:
        """Create side-by-side comparison image.

        Args:
            original: Original image.
            annotated: Annotated image.

        Returns:
            Combined image showing both side by side.
        """
        # Ensure same height
        h1, w1 = original.shape[:2]
        h2, w2 = annotated.shape[:2]

        if h1 != h2:
            # Resize to match heights
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            annotated = cv2.resize(annotated, (new_w2, h1))

        # Concatenate horizontally
        return np.concatenate([original, annotated], axis=1)

    def create_summary_dashboard(
        self,
        results: list[InspectionResult],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Create summary dashboard for multiple results.

        Generates a matplotlib figure with:
        - Severity distribution pie chart
        - Defect type distribution bar chart
        - Processing time histogram
        - Confidence distribution

        Args:
            results: List of inspection results.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Severity distribution (pie chart)
        severities = [r.severity.value for r in results]
        severity_counts = {}
        for s in ["OK", "WARNING", "NG"]:
            severity_counts[s] = severities.count(s)

        colors = ["green", "orange", "red"]
        ax1 = axes[0, 0]
        ax1.pie(
            severity_counts.values(),
            labels=severity_counts.keys(),
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("Severity Distribution")

        # 2. Defect type distribution (bar chart)
        defect_types: list[str] = []
        for r in results:
            defect_types.extend([d.class_name for d in r.detections])

        ax2 = axes[0, 1]
        if defect_types:
            unique_types = list(set(defect_types))
            type_counts = [defect_types.count(t) for t in unique_types]
            ax2.bar(unique_types, type_counts, color="steelblue")
            ax2.set_title("Defect Type Distribution")
            ax2.set_xlabel("Defect Type")
            ax2.set_ylabel("Count")
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        else:
            ax2.text(0.5, 0.5, "No defects detected", ha="center", va="center")
            ax2.set_title("Defect Type Distribution")

        # 3. Processing time histogram
        times = [r.processing_time_ms for r in results]
        ax3 = axes[1, 0]
        ax3.hist(times, bins=20, edgecolor="black", color="lightblue")
        ax3.set_xlabel("Processing Time (ms)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Processing Time Distribution")
        ax3.axvline(np.mean(times), color="red", linestyle="--", label=f"Mean: {np.mean(times):.1f}ms")
        ax3.legend()

        # 4. Confidence distribution
        confidences: list[float] = []
        for r in results:
            confidences.extend([d.confidence for d in r.detections])

        ax4 = axes[1, 1]
        if confidences:
            ax4.hist(confidences, bins=20, edgecolor="black", color="lightgreen")
            ax4.set_xlabel("Confidence")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Detection Confidence Distribution")
        else:
            ax4.text(0.5, 0.5, "No detections", ha="center", va="center")
            ax4.set_title("Detection Confidence Distribution")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
