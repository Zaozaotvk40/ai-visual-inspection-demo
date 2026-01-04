"""
検出器（detector.py）のユニットテスト.

YOLODefectDetector クラスをモックを使ってテストします。
実際のYOLOモデルは使用せず、軽量・高速なテストを実現します。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inspection.models.detector import YOLODefectDetector
from inspection.models.types import DetectionResult


# ========================================
# YOLOモックのヘルパー関数
# ========================================


def create_mock_yolo_model(detections_data: list[dict] = None):
    """YOLOモデルのモックを作成するヘルパー関数."""
    mock_model = MagicMock()

    # クラス名マッピング
    mock_model.names = {
        0: "scratch",
        1: "chip",
        2: "contamination",
        3: "crack",
        4: "dent",
        5: "discoloration",
    }

    if detections_data is None:
        # デフォルト: 1件の検出
        detections_data = [
            {"bbox": (100, 100, 200, 200), "confidence": 0.85, "class_id": 0}
        ]

    if not detections_data:
        # 検出なし
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]
    else:
        # 検出あり
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


# ========================================
# YOLODefectDetector 初期化テスト
# ========================================


class TestYOLODefectDetectorInit:
    """YOLODefectDetector初期化のテスト."""

    def test_yolo_detector_init_defaults(self):
        """デフォルト初期化が正しいことを確認."""
        detector = YOLODefectDetector()

        assert detector.model_path is None
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45
        assert detector.device == "cuda"
        assert detector.is_loaded is False

    def test_yolo_detector_init_with_custom_thresholds(self):
        """カスタム閾値で初期化できることを確認."""
        detector = YOLODefectDetector(
            confidence_threshold=0.6,
            iou_threshold=0.5,
            device="cpu",
        )

        assert detector.confidence_threshold == 0.6
        assert detector.iou_threshold == 0.5
        assert detector.device == "cpu"

    @patch("ultralytics.YOLO")
    def test_yolo_detector_init_with_model_path(self, mock_yolo_class):
        """model_pathを指定した場合、自動的にロードされることを確認."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector(model_path=Path("test_model.pt"))

        assert detector.model_path == Path("test_model.pt")
        assert detector.is_loaded is True
        mock_yolo_class.assert_called_once()


# ========================================
# YOLODefectDetector モデルロードテスト
# ========================================


class TestYOLODefectDetectorLoadModel:
    """YOLODefectDetector.load_model()のテスト."""

    @patch("ultralytics.YOLO")
    def test_load_model_success(self, mock_yolo_class):
        """モデルが正常にロードできることを確認."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        assert detector.is_loaded is True
        assert detector.model_path == Path("test_model.pt")
        mock_yolo_class.assert_called_once_with("test_model.pt")
        mock_yolo_model.to.assert_called_once_with("cuda")

    @patch("ultralytics.YOLO")
    def test_load_model_sets_class_names(self, mock_yolo_class):
        """モデルからクラス名が取得されることを確認."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        # _class_namesがモデルのnamesに設定される
        assert detector._class_names == mock_yolo_model.names

    def test_model_property_not_loaded(self):
        """モデルがロードされていない状態でmodelプロパティにアクセスするとRuntimeError."""
        detector = YOLODefectDetector()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = detector.model

    @patch("ultralytics.YOLO")
    def test_model_property_loaded(self, mock_yolo_class):
        """モデルロード後にmodelプロパティでアクセスできることを確認."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        assert detector.model is mock_yolo_model


# ========================================
# YOLODefectDetector 検出テスト
# ========================================


class TestYOLODefectDetectorDetect:
    """YOLODefectDetector.detect()のテスト."""

    @patch("ultralytics.YOLO")
    def test_detect_success(self, mock_yolo_class, sample_image):
        """検出が正常に動作することを確認."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        results = detector.detect(sample_image)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)
        assert results[0].class_name == "scratch"
        assert results[0].confidence == 0.85
        assert results[0].bbox == (100, 100, 200, 200)

        # predict が呼ばれたことを確認
        mock_yolo_model.predict.assert_called_once()

    @patch("ultralytics.YOLO")
    def test_detect_multiple_detections(self, mock_yolo_class, sample_image):
        """複数検出結果が正しく返されることを確認."""
        detections_data = [
            {"bbox": (100, 100, 200, 200), "confidence": 0.85, "class_id": 0},
            {"bbox": (300, 300, 400, 400), "confidence": 0.65, "class_id": 1},
            {"bbox": (50, 50, 100, 100), "confidence": 0.25, "class_id": 2},
        ]
        mock_yolo_model = create_mock_yolo_model(detections_data)
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        results = detector.detect(sample_image)

        assert len(results) == 3
        assert results[0].class_name == "scratch"
        assert results[1].class_name == "chip"
        assert results[2].class_name == "contamination"

    @patch("ultralytics.YOLO")
    def test_detect_no_results(self, mock_yolo_class, sample_image):
        """検出なしの場合に空リストが返されることを確認."""
        mock_yolo_model = create_mock_yolo_model([])  # 空のリスト = 検出なし
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        results = detector.detect(sample_image)

        assert isinstance(results, list)
        assert len(results) == 0

    @patch("ultralytics.YOLO")
    def test_detect_invalid_image_none(self, mock_yolo_class):
        """Noneが渡された場合にValueErrorが発生."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect(None)

    @patch("ultralytics.YOLO")
    def test_detect_invalid_image_empty(self, mock_yolo_class):
        """空の配列が渡された場合にValueErrorが発生."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        empty_image = np.array([])

        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect(empty_image)

    def test_detect_model_not_loaded(self, sample_image):
        """モデルがロードされていない状態でdetect()を呼ぶとAttributeError."""
        detector = YOLODefectDetector()

        # detect()は直接self._model.predict()を呼ぶため、
        # モデル未ロード時はAttributeErrorが発生
        with pytest.raises(AttributeError):
            detector.detect(sample_image)


# ========================================
# YOLODefectDetector バッチ検出テスト
# ========================================


class TestYOLODefectDetectorDetectBatch:
    """YOLODefectDetector.detect_batch()のテスト."""

    @patch("ultralytics.YOLO")
    def test_detect_batch(self, mock_yolo_class, sample_image, small_image):
        """バッチ検出が正常に動作することを確認."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector()
        detector.load_model(Path("test_model.pt"))

        images = [sample_image, small_image]
        results = detector.detect_batch(images)

        assert isinstance(results, list)
        assert len(results) == 2
        # 各画像に対して検出結果が返される
        assert isinstance(results[0], list)
        assert isinstance(results[1], list)


# ========================================
# YOLODefectDetector モデル情報テスト
# ========================================


class TestYOLODefectDetectorGetModelInfo:
    """YOLODefectDetector.get_model_info()のテスト."""

    @patch("ultralytics.YOLO")
    def test_get_model_info(self, mock_yolo_class):
        """モデル情報が正しく返されることを確認."""
        mock_yolo_model = create_mock_yolo_model()
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODefectDetector(confidence_threshold=0.6, iou_threshold=0.5)
        detector.load_model(Path("test_model.pt"))

        info = detector.get_model_info()

        assert info["type"] == "YOLOv8"
        assert info["path"] == "test_model.pt"
        assert info["confidence_threshold"] == 0.6
        assert info["iou_threshold"] == 0.5
        assert info["device"] == "cuda"
        assert info["is_loaded"] is True

    def test_get_model_info_not_loaded(self):
        """モデルがロードされていない場合のget_model_info()."""
        detector = YOLODefectDetector()

        info = detector.get_model_info()

        assert info["type"] == "YOLOv8"
        assert info["path"] is None
        assert info["is_loaded"] is False


# ========================================
# YOLODefectDetector 閾値設定テスト
# ========================================


class TestYOLODefectDetectorSetThresholds:
    """YOLODefectDetector.set_thresholds()のテスト."""

    def test_set_thresholds_confidence(self):
        """confidence閾値が更新できることを確認."""
        detector = YOLODefectDetector()

        detector.set_thresholds(confidence=0.7)

        assert detector.confidence_threshold == 0.7

    def test_set_thresholds_iou(self):
        """IoU閾値が更新できることを確認."""
        detector = YOLODefectDetector()

        detector.set_thresholds(iou=0.6)

        assert detector.iou_threshold == 0.6

    def test_set_thresholds_both(self):
        """両方の閾値が同時に更新できることを確認."""
        detector = YOLODefectDetector()

        detector.set_thresholds(confidence=0.75, iou=0.55)

        assert detector.confidence_threshold == 0.75
        assert detector.iou_threshold == 0.55

    @pytest.mark.parametrize("invalid_conf", [-0.1, 1.5, 2.0])
    def test_set_thresholds_invalid_confidence(self, invalid_conf):
        """無効なconfidence値でValueError."""
        detector = YOLODefectDetector()

        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            detector.set_thresholds(confidence=invalid_conf)

    @pytest.mark.parametrize("invalid_iou", [-0.1, 1.5, 2.0])
    def test_set_thresholds_invalid_iou(self, invalid_iou):
        """無効なIoU値でValueError."""
        detector = YOLODefectDetector()

        with pytest.raises(ValueError, match="IoU must be 0-1"):
            detector.set_thresholds(iou=invalid_iou)
