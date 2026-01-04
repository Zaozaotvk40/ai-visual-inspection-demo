"""
可視化（overlay.py）のユニットテスト.

ResultVisualizer クラスの主要機能をテストします。
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from inspection.models.types import Severity
from inspection.visualization.overlay import ResultVisualizer


# ========================================
# ResultVisualizer テスト
# ========================================


class TestResultVisualizer:
    """ResultVisualizer クラスのテスト."""

    def test_visualizer_init_defaults(self):
        """デフォルト初期化が正しいことを確認."""
        viz = ResultVisualizer()

        assert viz.font_scale == 0.5
        assert viz.line_thickness == 2

    def test_visualizer_init_custom(self):
        """カスタム値で初期化できることを確認."""
        viz = ResultVisualizer(font_scale=1.0, line_thickness=3)

        assert viz.font_scale == 1.0
        assert viz.line_thickness == 3

    def test_visualizer_draw_results_basic(
        self, sample_image, sample_inspection_result
    ):
        """結果を画像に描画できることを確認."""
        viz = ResultVisualizer()
        result_image = viz.draw_results(sample_image, sample_inspection_result)

        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == sample_image.shape
        assert result_image.dtype == np.uint8

    def test_visualizer_draw_results_no_detections(self, sample_image):
        """検出なしの場合も正常に描画できることを確認."""
        from inspection.models.types import InspectionResult

        result = InspectionResult(
            image_path="test.jpg",
            is_defective=False,
            detections=[],
            processing_time_ms=30.0,
            severity=Severity.OK,
        )

        viz = ResultVisualizer()
        result_image = viz.draw_results(sample_image, result)

        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == sample_image.shape

    def test_visualizer_draw_results_with_mask(
        self, sample_image, sample_detection_with_mask
    ):
        """マスク付き検出結果を描画できることを確認."""
        from inspection.models.types import InspectionResult

        result = InspectionResult(
            image_path="test.jpg",
            is_defective=True,
            detections=[sample_detection_with_mask],
            processing_time_ms=50.0,
            severity=Severity.NG,
        )

        viz = ResultVisualizer()
        result_image = viz.draw_results(sample_image, result)

        assert isinstance(result_image, np.ndarray)

    def test_visualizer_draw_results_options(self, sample_image, sample_inspection_result):
        """表示オプションが正しく機能することを確認."""
        viz = ResultVisualizer()

        # ラベルのみ表示
        result_image = viz.draw_results(
            sample_image,
            sample_inspection_result,
            show_labels=True,
            show_severity=False,
            show_time=False,
        )
        assert isinstance(result_image, np.ndarray)

        # 重要度のみ表示
        result_image = viz.draw_results(
            sample_image,
            sample_inspection_result,
            show_labels=False,
            show_severity=True,
            show_time=False,
        )
        assert isinstance(result_image, np.ndarray)

    def test_visualizer_severity_colors(self):
        """各重要度に対応する色が定義されていることを確認."""
        viz = ResultVisualizer()

        assert hasattr(viz, "severity_colors") or hasattr(viz, "SEVERITY_COLORS")

    def test_visualizer_save_image(self, sample_image, sample_inspection_result):
        """画像を保存できることを確認."""
        viz = ResultVisualizer()
        result_image = viz.draw_results(sample_image, sample_inspection_result)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            output_path = Path(f.name)

        try:
            viz.save_image(result_image, output_path)
            assert output_path.exists()
        finally:
            output_path.unlink(missing_ok=True)
