"""
共通テストフィクスチャ

このモジュールは全テストで共有されるフィクスチャを提供します。
- サンプル画像データ
- モックMLモデル（YOLO、SAM）
- テスト用データ構造
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from inspection.models.types import DefectClass, DetectionResult, InspectionResult, Severity


# ========================================
# 画像フィクスチャ
# ========================================


@pytest.fixture
def sample_image() -> np.ndarray:
    """テスト用RGB画像 (640x640x3) を生成."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image() -> np.ndarray:
    """リサイズテスト用の小さいRGB画像 (100x100x3) を生成."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def large_image() -> np.ndarray:
    """リサイズテスト用の大きいRGB画像 (1920x1080x3) を生成."""
    return np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """テスト用バイナリマスク (640x640) を生成."""
    mask = np.zeros((640, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 1  # 正方形の欠陥領域
    return mask


@pytest.fixture
def temp_image_file(sample_image) -> Path:
    """一時画像ファイルを作成."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # RGB to BGR for OpenCV
        bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f.name, bgr)
        temp_path = Path(f.name)

    yield temp_path

    # クリーンアップ
    temp_path.unlink(missing_ok=True)


# ========================================
# 検出結果フィクスチャ
# ========================================


@pytest.fixture
def sample_detection() -> DetectionResult:
    """単一の検出結果を生成."""
    return DetectionResult(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
        class_name="scratch",
    )


@pytest.fixture
def sample_detection_with_mask(sample_mask) -> DetectionResult:
    """マスク付き検出結果を生成."""
    return DetectionResult(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
        class_name="scratch",
        mask=sample_mask,
    )


@pytest.fixture
def sample_detections() -> list[DetectionResult]:
    """複数の検出結果を生成（異なる重要度レベル）."""
    return [
        DetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_id=0,
            class_name="scratch",
        ),
        DetectionResult(
            bbox=(300, 300, 400, 400),
            confidence=0.65,
            class_id=1,
            class_name="chip",
        ),
        DetectionResult(
            bbox=(50, 50, 100, 100),
            confidence=0.25,
            class_id=2,
            class_name="contamination",
        ),
    ]


@pytest.fixture
def sample_inspection_result(sample_detections) -> InspectionResult:
    """検査結果を生成."""
    return InspectionResult(
        image_path="test_image.jpg",
        is_defective=True,
        detections=sample_detections,
        processing_time_ms=50.0,
        severity=Severity.NG,
    )


# ========================================
# モックYOLOモデル
# ========================================


def create_mock_yolo_boxes(detections_data: list[dict]) -> list:
    """
    YOLOの検出結果をモック化するヘルパー関数.

    Args:
        detections_data: 検出データのリスト。各要素は以下のキーを持つ辞書:
            - bbox: (x1, y1, x2, y2)
            - confidence: 信頼度 (0.0-1.0)
            - class_id: クラスID

    Returns:
        モックされたYOLO Boxesオブジェクトのリスト
    """
    mock_boxes_list = []

    for data in detections_data:
        mock_box = MagicMock()

        # xyxy (バウンディングボックス座標)
        xyxy_array = np.array([data["bbox"]])
        mock_box.xyxy.cpu.return_value.numpy.return_value = xyxy_array

        # conf (信頼度)
        conf_array = np.array([data["confidence"]])
        mock_box.conf.cpu.return_value.numpy.return_value = conf_array

        # cls (クラスID)
        cls_array = np.array([data["class_id"]])
        mock_box.cls.cpu.return_value.numpy.return_value = cls_array

        mock_boxes_list.append(mock_box)

    return mock_boxes_list


@pytest.fixture
def mock_yolo_model():
    """単一検出結果を返すYOLOモデルのモック."""
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

    # 検出結果のモック
    detections_data = [
        {"bbox": (100, 100, 200, 200), "confidence": 0.85, "class_id": 0}
    ]
    mock_boxes = create_mock_yolo_boxes(detections_data)

    # Resultオブジェクトのモック
    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    # predictメソッドの戻り値
    mock_model.predict.return_value = [mock_result]

    # toメソッド（デバイス移動）
    mock_model.to.return_value = mock_model

    return mock_model


@pytest.fixture
def mock_yolo_multiple_detections():
    """複数検出結果を返すYOLOモデルのモック."""
    mock_model = MagicMock()

    mock_model.names = {
        0: "scratch",
        1: "chip",
        2: "contamination",
        3: "crack",
        4: "dent",
        5: "discoloration",
    }

    # 3つの検出結果
    detections_data = [
        {"bbox": (100, 100, 200, 200), "confidence": 0.85, "class_id": 0},
        {"bbox": (300, 300, 400, 400), "confidence": 0.65, "class_id": 1},
        {"bbox": (50, 50, 100, 100), "confidence": 0.25, "class_id": 2},
    ]
    mock_boxes = create_mock_yolo_boxes(detections_data)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_model.predict.return_value = [mock_result]
    mock_model.to.return_value = mock_model

    return mock_model


@pytest.fixture
def mock_yolo_no_detections():
    """検出結果なしを返すYOLOモデルのモック."""
    mock_model = MagicMock()

    mock_model.names = {
        0: "scratch",
        1: "chip",
        2: "contamination",
        3: "crack",
        4: "dent",
        5: "discoloration",
    }

    # 検出なし
    mock_result = MagicMock()
    mock_result.boxes = None

    mock_model.predict.return_value = [mock_result]
    mock_model.to.return_value = mock_model

    return mock_model


# ========================================
# モックSAMモデル
# ========================================


@pytest.fixture
def mock_sam_model(sample_mask):
    """マスクを返すSAMモデルのモック."""
    mock_model = MagicMock()

    # マスクのモック
    mock_mask_obj = MagicMock()
    mock_mask_obj.data.cpu.return_value.numpy.return_value.squeeze.return_value = (
        sample_mask.astype(float)
    )

    # Resultオブジェクトのモック
    mock_result = MagicMock()
    mock_result.masks = [mock_mask_obj]

    # predictメソッドの戻り値
    mock_model.predict.return_value = [mock_result]

    return mock_model


@pytest.fixture
def mock_sam_no_masks():
    """マスクなしを返すSAMモデルのモック."""
    mock_model = MagicMock()

    mock_result = MagicMock()
    mock_result.masks = None

    mock_model.predict.return_value = [mock_result]

    return mock_model
