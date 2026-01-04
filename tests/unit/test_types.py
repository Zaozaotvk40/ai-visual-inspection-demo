"""
データ型（types.py）のユニットテスト.

DefectClass, Severity, DetectionResult, InspectionResultの
各クラス・メソッド・プロパティをテストします。
"""

import numpy as np
import pytest

from inspection.models.types import DefectClass, DetectionResult, InspectionResult, Severity


# ========================================
# DefectClass テスト
# ========================================


class TestDefectClass:
    """DefectClass Enumのテスト."""

    def test_defect_class_values(self):
        """全6クラスが正しいIDを持つことを確認."""
        assert DefectClass.SCRATCH.value == 0
        assert DefectClass.CHIP.value == 1
        assert DefectClass.CONTAMINATION.value == 2
        assert DefectClass.CRACK.value == 3
        assert DefectClass.DENT.value == 4
        assert DefectClass.DISCOLORATION.value == 5

    @pytest.mark.parametrize(
        "class_id,expected_name",
        [
            (0, "scratch"),
            (1, "chip"),
            (2, "contamination"),
            (3, "crack"),
            (4, "dent"),
            (5, "discoloration"),
        ],
    )
    def test_defect_class_from_id_valid(self, class_id, expected_name):
        """有効なIDから正しいDefectClassを取得."""
        result = DefectClass.from_id(class_id)
        assert result.name.lower() == expected_name
        assert result.value == class_id

    @pytest.mark.parametrize("invalid_id", [-1, 6, 10, 100])
    def test_defect_class_from_id_invalid(self, invalid_id):
        """無効なIDでValueErrorが発生することを確認."""
        with pytest.raises(ValueError, match=f"Invalid class_id: {invalid_id}"):
            DefectClass.from_id(invalid_id)

    def test_defect_class_get_name_mapping(self):
        """ID→名前のマッピングが正しいことを確認."""
        mapping = DefectClass.get_name_mapping()

        assert len(mapping) == 6
        assert mapping[0] == "scratch"
        assert mapping[1] == "chip"
        assert mapping[2] == "contamination"
        assert mapping[3] == "crack"
        assert mapping[4] == "dent"
        assert mapping[5] == "discoloration"


# ========================================
# Severity テスト
# ========================================


class TestSeverity:
    """Severity Enumのテスト."""

    def test_severity_values(self):
        """3つの重要度レベルが正しい値を持つことを確認."""
        assert Severity.OK.value == "OK"
        assert Severity.WARNING.value == "WARNING"
        assert Severity.NG.value == "NG"

    def test_severity_members(self):
        """全メンバーが存在することを確認."""
        members = [s.value for s in Severity]
        assert "OK" in members
        assert "WARNING" in members
        assert "NG" in members
        assert len(members) == 3


# ========================================
# DetectionResult テスト
# ========================================


class TestDetectionResult:
    """DetectionResultデータクラスのテスト."""

    def test_detection_result_creation(self):
        """DetectionResultが正しく生成されることを確認."""
        detection = DetectionResult(
            bbox=(10, 20, 100, 120),
            confidence=0.85,
            class_id=0,
            class_name="scratch",
        )

        assert detection.bbox == (10, 20, 100, 120)
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "scratch"
        assert detection.mask is None

    def test_detection_result_with_mask(self):
        """マスク付きDetectionResultが正しく生成されることを確認."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        detection = DetectionResult(
            bbox=(10, 20, 100, 120),
            confidence=0.75,
            class_id=1,
            class_name="chip",
            mask=mask,
        )

        assert detection.mask is not None
        assert detection.mask.shape == (100, 100)

    @pytest.mark.parametrize("invalid_conf", [-0.1, 1.5, 2.0, -1.0])
    def test_detection_result_invalid_confidence(self, invalid_conf):
        """信頼度が0-1の範囲外でValueErrorが発生することを確認."""
        with pytest.raises(
            ValueError, match="Confidence must be between 0 and 1"
        ):
            DetectionResult(
                bbox=(10, 20, 100, 120),
                confidence=invalid_conf,
                class_id=0,
                class_name="scratch",
            )

    @pytest.mark.parametrize("invalid_bbox", [(10, 20, 100), (10, 20), (10, 20, 30, 40, 50)])
    def test_detection_result_invalid_bbox(self, invalid_bbox):
        """バウンディングボックスが4要素でない場合にValueErrorが発生することを確認."""
        with pytest.raises(ValueError, match="Bbox must have 4 elements"):
            DetectionResult(
                bbox=invalid_bbox,
                confidence=0.5,
                class_id=0,
                class_name="scratch",
            )

    def test_detection_result_area(self):
        """面積計算が正しいことを確認."""
        detection = DetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.5,
            class_id=0,
            class_name="scratch",
        )

        # (200-100) * (200-100) = 100 * 100 = 10000
        assert detection.area == 10000

    def test_detection_result_area_non_square(self):
        """非正方形のバウンディングボックスの面積計算."""
        detection = DetectionResult(
            bbox=(50, 100, 150, 300),
            confidence=0.5,
            class_id=0,
            class_name="scratch",
        )

        # (150-50) * (300-100) = 100 * 200 = 20000
        assert detection.area == 20000

    def test_detection_result_center(self):
        """中心座標計算が正しいことを確認."""
        detection = DetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.5,
            class_id=0,
            class_name="scratch",
        )

        # center = ((100+200)//2, (100+200)//2) = (150, 150)
        assert detection.center == (150, 150)

    def test_detection_result_center_odd_coordinates(self):
        """奇数座標での中心計算（整数除算の確認）."""
        detection = DetectionResult(
            bbox=(10, 20, 101, 121),
            confidence=0.5,
            class_id=0,
            class_name="scratch",
        )

        # center = ((10+101)//2, (20+121)//2) = (55, 70)
        assert detection.center == (55, 70)


# ========================================
# InspectionResult テスト
# ========================================


class TestInspectionResult:
    """InspectionResultデータクラスのテスト."""

    def test_inspection_result_creation(self):
        """InspectionResultが正しく生成されることを確認."""
        detections = [
            DetectionResult(
                bbox=(100, 100, 200, 200),
                confidence=0.85,
                class_id=0,
                class_name="scratch",
            )
        ]

        result = InspectionResult(
            image_path="test.jpg",
            is_defective=True,
            detections=detections,
            processing_time_ms=50.0,
            severity=Severity.NG,
        )

        assert result.image_path == "test.jpg"
        assert result.is_defective is True
        assert len(result.detections) == 1
        assert result.processing_time_ms == 50.0
        assert result.severity == Severity.NG

    def test_inspection_result_no_defects(self):
        """欠陥なしのInspectionResult."""
        result = InspectionResult(
            image_path="clean.jpg",
            is_defective=False,
            detections=[],
            processing_time_ms=30.0,
            severity=Severity.OK,
        )

        assert result.is_defective is False
        assert len(result.detections) == 0
        assert result.severity == Severity.OK

    def test_inspection_result_defect_count(self):
        """defect_countプロパティが正しいことを確認."""
        detections = [
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
        ]

        result = InspectionResult(
            image_path="test.jpg",
            is_defective=True,
            detections=detections,
            processing_time_ms=50.0,
            severity=Severity.NG,
        )

        assert result.defect_count == 2

    def test_inspection_result_max_confidence_with_detections(self):
        """検出ありの場合のmax_confidence."""
        detections = [
            DetectionResult(
                bbox=(100, 100, 200, 200),
                confidence=0.85,
                class_id=0,
                class_name="scratch",
            ),
            DetectionResult(
                bbox=(300, 300, 400, 400),
                confidence=0.95,  # 最大
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

        result = InspectionResult(
            image_path="test.jpg",
            is_defective=True,
            detections=detections,
            processing_time_ms=50.0,
            severity=Severity.NG,
        )

        assert result.max_confidence == 0.95

    def test_inspection_result_max_confidence_empty(self):
        """検出なしの場合のmax_confidenceが0.0であることを確認."""
        result = InspectionResult(
            image_path="clean.jpg",
            is_defective=False,
            detections=[],
            processing_time_ms=30.0,
            severity=Severity.OK,
        )

        assert result.max_confidence == 0.0

    def test_inspection_result_get_defects_by_class(self):
        """クラス名でフィルタリングが正しく動作することを確認."""
        detections = [
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
                confidence=0.75,
                class_id=0,
                class_name="scratch",
            ),
        ]

        result = InspectionResult(
            image_path="test.jpg",
            is_defective=True,
            detections=detections,
            processing_time_ms=50.0,
            severity=Severity.NG,
        )

        scratches = result.get_defects_by_class("scratch")
        assert len(scratches) == 2
        assert all(det.class_name == "scratch" for det in scratches)

        chips = result.get_defects_by_class("chip")
        assert len(chips) == 1
        assert chips[0].class_name == "chip"

        cracks = result.get_defects_by_class("crack")
        assert len(cracks) == 0

    def test_inspection_result_to_dict(self):
        """辞書変換が正しく動作することを確認."""
        detections = [
            DetectionResult(
                bbox=(100, 100, 200, 200),
                confidence=0.85,
                class_id=0,
                class_name="scratch",
            )
        ]

        result = InspectionResult(
            image_path="test.jpg",
            is_defective=True,
            detections=detections,
            processing_time_ms=50.5,
            severity=Severity.NG,
        )

        result_dict = result.to_dict()

        assert result_dict["image_path"] == "test.jpg"
        assert result_dict["is_defective"] is True
        assert result_dict["defect_count"] == 1
        assert result_dict["processing_time_ms"] == 50.5
        assert result_dict["severity"] == "NG"
        assert len(result_dict["detections"]) == 1

        # 検出データの確認
        det_dict = result_dict["detections"][0]
        assert det_dict["bbox"] == (100, 100, 200, 200)
        assert det_dict["confidence"] == 0.85
        assert det_dict["class_id"] == 0
        assert det_dict["class_name"] == "scratch"
        assert det_dict["area"] == 10000

    def test_inspection_result_to_dict_empty_detections(self):
        """検出なしの場合の辞書変換."""
        result = InspectionResult(
            image_path="clean.jpg",
            is_defective=False,
            detections=[],
            processing_time_ms=30.0,
            severity=Severity.OK,
        )

        result_dict = result.to_dict()

        assert result_dict["defect_count"] == 0
        assert result_dict["severity"] == "OK"
        assert result_dict["detections"] == []
