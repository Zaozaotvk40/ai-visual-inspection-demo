#!/usr/bin/env python
"""
Inference demo script for the visual inspection pipeline.

Usage:
    python scripts/inference_demo.py --image path/to/image.jpg
    python scripts/inference_demo.py --image path/to/image.jpg --no-sam
    python scripts/inference_demo.py --image path/to/image.jpg --output result.jpg
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inspection.config.settings import get_settings
from inspection.models import Severity
from inspection.pipeline import InspectionPipeline, PipelineConfig
from inspection.preprocessing import ImageLoader
from inspection.visualization import ResultVisualizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual Inspection AI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/yolov8n.pt"),
        help="Path to YOLOv8 model weights (default: models/yolov8n.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save annotated output image",
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Disable SAM segmentation refinement",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default=None,
        choices=["sam2_t", "sam2_s", "sam2_b", "sam2_l"],
        help="SAM model variant (default: .env or sam2_b)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result image using matplotlib",
    )

    return parser.parse_args()


def main() -> None:
    """Run inference demo."""
    args = parse_args()

    # Validate input
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Get settings from .env, use CLI args to override
    settings = get_settings()
    sam_model_type = args.sam_model if args.sam_model else settings.sam_model_type

    print("=" * 60)
    print("Visual Inspection AI Demo")
    print("=" * 60)
    print(f"Input image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Use SAM: {not args.no_sam}")
    print(f"SAM Model: {sam_model_type}")
    print(f"Device: {args.device}")
    print("-" * 60)

    # Create pipeline config
    config = PipelineConfig(
        detector_model_path=args.model,
        sam_model_type=sam_model_type,
        use_sam_refinement=not args.no_sam,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = InspectionPipeline(config)

    try:
        pipeline.initialize()
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nNote: If using pretrained YOLO model, it will be downloaded automatically.")
        sys.exit(1)

    # Run inspection
    print(f"\nInspecting: {args.image.name}")
    result = pipeline.inspect(args.image)

    # Print results
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Severity: {result.severity.value}")
    print(f"Is Defective: {result.is_defective}")
    print(f"Defect Count: {result.defect_count}")
    print(f"Processing Time: {result.processing_time_ms:.1f} ms")

    if result.detections:
        print("\nDetections:")
        for i, det in enumerate(result.detections, 1):
            print(f"  {i}. {det.class_name}")
            print(f"     Confidence: {det.confidence:.2%}")
            print(f"     Bounding Box: {det.bbox}")
            print(f"     Area: {det.area} px")
            if det.mask is not None:
                print(f"     Has Mask: Yes")

    # Visualize results
    if args.output or args.show:
        print("\nGenerating visualization...")

        # Load original image
        loader = ImageLoader(target_size=None)
        image = loader.load(args.image)

        # Draw results
        visualizer = ResultVisualizer()
        annotated = visualizer.draw_results(image, result)

        # Save if output path provided
        if args.output:
            visualizer.save_image(annotated, args.output)
            print(f"Saved annotated image to: {args.output}")

        # Display if requested
        if args.show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            plt.imshow(annotated)
            plt.axis("off")
            plt.title(f"Severity: {result.severity.value} | Defects: {result.defect_count}")
            plt.tight_layout()
            plt.show()

    print("-" * 60)
    print("Demo complete!")

    # Return appropriate exit code
    if result.severity == Severity.NG:
        sys.exit(2)  # Critical defects
    elif result.severity == Severity.WARNING:
        sys.exit(1)  # Warnings
    else:
        sys.exit(0)  # OK


if __name__ == "__main__":
    main()
