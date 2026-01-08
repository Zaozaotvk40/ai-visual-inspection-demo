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
import time
from pathlib import Path

# Supported image extensions for folder processing
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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
        help="Path to input image or folder containing images",
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


def process_single_image(
    pipeline: "InspectionPipeline",
    image_path: Path,
    output_path: Path | None,
    show: bool,
    verbose: bool = True,
) -> "Severity":
    """Process a single image and return its severity.

    Args:
        pipeline: Initialized inspection pipeline.
        image_path: Path to input image.
        output_path: Path to save annotated output (optional).
        show: Whether to display result using matplotlib.
        verbose: Whether to print detailed results.

    Returns:
        Severity level of the inspection result.
    """
    result = pipeline.inspect(image_path)

    if verbose:
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
                    print("     Has Mask: Yes")

    # Visualize results
    if output_path or show:
        if verbose:
            print("\nGenerating visualization...")

        loader = ImageLoader(target_size=None)
        image = loader.load(image_path)

        visualizer = ResultVisualizer()
        annotated = visualizer.draw_results(image, result)

        if output_path:
            visualizer.save_image(annotated, output_path)
            if verbose:
                print(f"Saved annotated image to: {output_path}")

        if show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            plt.imshow(annotated)
            plt.axis("off")
            plt.title(f"Severity: {result.severity.value} | Defects: {result.defect_count}")
            plt.tight_layout()
            plt.show()

    return result.severity


def process_folder(
    pipeline: "InspectionPipeline",
    folder_path: Path,
    output_folder: Path | None,
) -> "Severity":
    """Process all images in a folder and return the worst severity.

    Args:
        pipeline: Initialized inspection pipeline.
        folder_path: Path to folder containing images.
        output_folder: Path to folder for saving annotated outputs (optional).

    Returns:
        Worst (most severe) severity level among all processed images.
    """
    # Collect supported images
    images = [
        f for f in sorted(folder_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        print(f"No supported images found in: {folder_path}")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return Severity.OK

    print(f"Found {len(images)} images")
    print("-" * 60)

    # Create output folder if needed
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)

    # Process each image
    stats = {"OK": 0, "WARNING": 0, "NG": 0}
    worst_severity = Severity.OK
    start_time = time.time()

    for i, image_path in enumerate(images, 1):
        print(f"Processing: {image_path.name} ({i}/{len(images)})")

        # Determine output path
        output_path = None
        if output_folder:
            stem = image_path.stem
            output_path = output_folder / f"{stem}_result.jpg"

        # Process image (non-verbose mode)
        result = pipeline.inspect(image_path)
        severity = result.severity

        # Update stats
        stats[severity.value] += 1
        if severity == Severity.NG:
            worst_severity = Severity.NG
        elif severity == Severity.WARNING and worst_severity != Severity.NG:
            worst_severity = Severity.WARNING

        # Save visualization if output folder specified
        if output_path:
            loader = ImageLoader(target_size=None)
            image = loader.load(image_path)
            visualizer = ResultVisualizer()
            annotated = visualizer.draw_results(image, result)
            visualizer.save_image(annotated, output_path)

        # Print compact result
        print(f"  -> {severity.value} ({result.defect_count} defects, {result.processing_time_ms:.1f} ms)")

    total_time = time.time() - start_time

    # Print summary
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total: {len(images)} images")
    print(f"  OK:      {stats['OK']}")
    print(f"  WARNING: {stats['WARNING']}")
    print(f"  NG:      {stats['NG']}")
    print(f"Total Time: {total_time:.1f} s")
    if output_folder:
        print(f"Results saved to: {output_folder}")

    return worst_severity


def main() -> None:
    """Run inference demo."""
    args = parse_args()

    # Validate input
    if not args.image.exists():
        print(f"Error: Path not found: {args.image}")
        sys.exit(1)

    # Determine if folder or single image mode
    is_folder_mode = args.image.is_dir()

    # Get settings from .env, use CLI args to override
    settings = get_settings()
    sam_model_type = args.sam_model if args.sam_model else settings.sam_model_type

    # Print header
    print("=" * 60)
    if is_folder_mode:
        print("Visual Inspection AI Demo - Folder Mode")
    else:
        print("Visual Inspection AI Demo")
    print("=" * 60)

    if is_folder_mode:
        print(f"Input folder: {args.image}")
        if args.show:
            print("Warning: --show is ignored for folder processing")
    else:
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

    # Process based on mode
    if is_folder_mode:
        # Folder mode
        worst_severity = process_folder(pipeline, args.image, args.output)
    else:
        # Single image mode
        print(f"\nInspecting: {args.image.name}")
        worst_severity = process_single_image(
            pipeline, args.image, args.output, args.show
        )

    print("-" * 60)
    print("Demo complete!")

    # Return appropriate exit code based on worst severity
    if worst_severity == Severity.NG:
        sys.exit(2)  # Critical defects
    elif worst_severity == Severity.WARNING:
        sys.exit(1)  # Warnings
    else:
        sys.exit(0)  # OK


if __name__ == "__main__":
    main()
