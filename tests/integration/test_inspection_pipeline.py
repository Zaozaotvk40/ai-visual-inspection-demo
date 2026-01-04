"""
検査パイプライン（inspection_pipeline.py）の統合テスト.

InspectionPipeline と PipelineConfig の統合動作をテストします。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inspection.models.types import Severity
from inspection.pipeline.inspection_pipeline import InspectionPipeline, PipelineConfig


# ========================================
# モックのヘルパー関数
# ========================================


def create_mock_yolo_model(detections_data: list[dict] = None):
    """YOLOモデルのモックを作成するヘルパー関数."""
    mock_model = MagicMock()

    mock_model.names = {
        0: "scratch",
        1: "chip",
        2: "contamination",
        3: "crack",
        4: "dent",
        5: "discoloration",
    }

    if detections_data is None:
        detections_data = [
            {"bbox": (100, 100, 200, 200), "confidence": 0.85, "class_id": 0}
        ]

    if not detections_data:
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]
    else:
        mock_boxes = []
        for data in detections_data:
            mock_box = MagicMock()
            mock_box.xyxy = [MagicMock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array(data["bbox"])
            mock_box.conf = [MagicMock()]
            mock_box.conf[0].cpu.return_value.numpy.return_value = data["confidence"]
            mock_box.cls = [MagicMock()]
            mock_box.cls[0].cpu.return_value.numpy.return_value = data["class_id"]
            mock_boxes.append(mock_box)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

    mock_model.to.return_value = mock_model

    return mock_model


def create_mock_sam_model():
    """SAMモデルのモックを作成するヘルパー関数."""
    mock_model = MagicMock()

    mock_mask = np.zeros((640, 640), dtype=np.float32)
    mock_mask[100:200, 100:200] = 1.0

    mock_mask_obj = MagicMock()
    mock_mask_obj.data.cpu.return_value.numpy.return_value.squeeze.return_value = mock_mask

    mock_result = MagicMock()
    mock_result.masks = [mock_mask_obj]
    mock_model.predict.return_value = [mock_result]

    return mock_model


# ========================================
# PipelineConfig テスト
# ========================================


class TestPipelineConfig:
    """PipelineConfig データクラスのテスト."""

    def test_pipeline_config_defaults(self):
        """デフォルト値が正しいことを確認."""
        config = PipelineConfig(detector_model_path=Path("test_model.pt"))

        assert config.detector_model_path == Path("test_model.pt")
        assert config.sam_model_type == "sam2_b"
        assert config.use_sam_refinement is True
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.warning_threshold == 0.3
        assert config.ng_threshold == 0.7
        assert config.device == "auto"
        assert config.image_size == 640

    def test_pipeline_config_custom_values(self):
        """カスタム値で初期化できることを確認."""
        config = PipelineConfig(
            detector_model_path=Path("custom_model.pt"),
            sam_model_type="sam2_s",
            use_sam_refinement=False,
            confidence_threshold=0.6,
            iou_threshold=0.5,
            warning_threshold=0.4,
            ng_threshold=0.8,
            device="cpu",
            image_size=1024,
        )

        assert config.sam_model_type == "sam2_s"
        assert config.use_sam_refinement is False
        assert config.confidence_threshold == 0.6
        assert config.device == "cpu"
        assert config.image_size == 1024

    def test_pipeline_config_from_settings(self):
        """Settingsから設定を生成できることを確認."""
        from inspection.config.settings import Settings

        settings = Settings(
            model_path=Path("models/yolov8n.pt"),
            sam_model_type="sam2_l",
            confidence_threshold=0.7,
        )

        config = PipelineConfig.from_settings(settings)

        assert config.detector_model_path == Path("models/yolov8n.pt")
        assert config.sam_model_type == "sam2_l"
        assert config.confidence_threshold == 0.7


# ========================================
# InspectionPipeline テスト
# ========================================


class TestInspectionPipeline:
    """InspectionPipeline クラスのテスト."""

    def test_pipeline_init(self):
        """パイプラインが正常に初期化できることを確認."""
        config = PipelineConfig(detector_model_path=Path("test_model.pt"))
        pipeline = InspectionPipeline(config)

        assert pipeline.config == config
        assert pipeline.is_initialized is False

    def test_pipeline_not_initialized_error(self):
        """初期化前にdetectorアクセスするとRuntimeError."""
        config = PipelineConfig(detector_model_path=Path("test_model.pt"))
        pipeline = InspectionPipeline(config)

        with pytest.raises(RuntimeError, match="Pipeline not initialized"):
            _ = pipeline.detector

    @patch("ultralytics.SAM")
    @patch("ultralytics.YOLO")
    def test_pipeline_initialize(self, mock_yolo_class, mock_sam_class):
        """パイプラインの初期化が正常に動作することを確認."""
        mock_yolo_class.return_value = create_mock_yolo_model()
        mock_sam_class.return_value = create_mock_sam_model()

        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            use_sam_refinement=True,
        )
        pipeline = InspectionPipeline(config)

        with patch.object(pipeline, "_resolve_device", return_value="cpu"):
            pipeline.initialize()

        assert pipeline.is_initialized is True
        assert pipeline.detector is not None
        assert pipeline.image_loader is not None

    @patch("ultralytics.YOLO")
    def test_pipeline_initialize_without_sam(self, mock_yolo_class):
        """SAM使用なしでパイプラインを初期化できることを確認."""
        mock_yolo_class.return_value = create_mock_yolo_model()

        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            use_sam_refinement=False,
        )
        pipeline = InspectionPipeline(config)

        with patch.object(pipeline, "_resolve_device", return_value="cpu"):
            pipeline.initialize()

        assert pipeline.is_initialized is True

    @patch("inspection.pipeline.inspection_pipeline.time")
    @patch("ultralytics.YOLO")
    def test_pipeline_inspect_image_array(
        self, mock_yolo_class, mock_time, sample_image
    ):
        """numpy配列の画像を検査できることを確認."""
        mock_yolo_class.return_value = create_mock_yolo_model()
        # time.time()が呼ばれるたびに異なる値を返す（10ms経過をシミュレート）
        mock_time.time.side_effect = [0.0, 0.01]

        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            use_sam_refinement=False,
        )
        pipeline = InspectionPipeline(config)

        with patch.object(pipeline, "_resolve_device", return_value="cpu"):
            pipeline.initialize()

        result = pipeline.inspect_image(sample_image, image_name="test.jpg")

        assert result.image_path == "test.jpg"
        assert result.is_defective is True
        assert len(result.detections) > 0
        assert result.severity in [Severity.OK, Severity.WARNING, Severity.NG]
        assert result.processing_time_ms > 0

    @patch("ultralytics.YOLO")
    def test_pipeline_severity_ok(self, mock_yolo_class, sample_image):
        """検出なしの場合にseverity=OKとなることを確認."""
        mock_yolo_class.return_value = create_mock_yolo_model([])  # 検出なし

        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            use_sam_refinement=False,
        )
        pipeline = InspectionPipeline(config)

        with patch.object(pipeline, "_resolve_device", return_value="cpu"):
            pipeline.initialize()

        result = pipeline.inspect_image(sample_image)

        assert result.severity == Severity.OK
        assert result.is_defective is False

    @patch("ultralytics.YOLO")
    def test_pipeline_severity_ng(self, mock_yolo_class, sample_image):
        """高信頼度検出でseverity=NGとなることを確認."""
        mock_yolo_class.return_value = create_mock_yolo_model()

        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            use_sam_refinement=False,
            ng_threshold=0.7,
        )
        pipeline = InspectionPipeline(config)

        with patch.object(pipeline, "_resolve_device", return_value="cpu"):
            pipeline.initialize()

        # mock_yolo_modelは信頼度0.85の検出を返すので、NG判定になるはず
        result = pipeline.inspect_image(sample_image)

        assert result.severity == Severity.NG

    @patch("torch.cuda.is_available")
    def test_pipeline_resolve_device_auto_cuda(self, mock_cuda):
        """device='auto'でCUDA利用可能時にcudaを返すことを確認."""
        mock_cuda.return_value = True

        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            device="auto",
        )
        pipeline = InspectionPipeline(config)

        device = pipeline._resolve_device()

        assert device == "cuda"

    @patch("torch.cuda.is_available")
    def test_pipeline_resolve_device_auto_cpu(self, mock_cuda):
        """device='auto'でCUDA利用不可時にcpuを返すことを確認."""
        mock_cuda.return_value = False

        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            device="auto",
        )
        pipeline = InspectionPipeline(config)

        device = pipeline._resolve_device()

        assert device == "cpu"

    def test_pipeline_resolve_device_explicit(self):
        """deviceが明示的に指定されている場合、そのまま返すことを確認."""
        config = PipelineConfig(
            detector_model_path=Path("test_model.pt"),
            device="cpu",
        )
        pipeline = InspectionPipeline(config)

        device = pipeline._resolve_device()

        assert device == "cpu"
