"""
セグメンター（segmenter.py）のユニットテスト.

SAMSegmenter と DefectSegmentationRefiner の主要機能をテストします。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inspection.models.segmenter import DefectSegmentationRefiner, SAMSegmenter


# ========================================
# SAMSegmenter テスト
# ========================================


class TestSAMSegmenter:
    """SAMSegmenter クラスのテスト."""

    def test_sam_segmenter_init_valid_type(self):
        """有効なmodel_typeで初期化できることを確認."""
        for model_type in ["sam2_t", "sam2_s", "sam2_b", "sam2_l"]:
            segmenter = SAMSegmenter(model_type=model_type)
            assert segmenter.model_type == model_type
            assert segmenter.device == "cuda"
            assert segmenter.is_loaded is False

    def test_sam_segmenter_init_invalid_type(self):
        """無効なmodel_typeでValueErrorが発生."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            SAMSegmenter(model_type="invalid_model")

    def test_sam_segmenter_init_custom_device(self):
        """デバイスをカスタマイズできることを確認."""
        segmenter = SAMSegmenter(device="cpu")
        assert segmenter.device == "cpu"

    def test_sam_segmenter_is_loaded_false(self):
        """モデルロード前はis_loadedがFalse."""
        segmenter = SAMSegmenter()
        assert segmenter.is_loaded is False

    @patch("ultralytics.SAM")
    def test_sam_segmenter_load_model(self, mock_sam_class):
        """モデルが正常にロードできることを確認."""
        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        segmenter = SAMSegmenter(model_type="sam2_b")
        segmenter.load_model()

        assert segmenter.is_loaded is True
        # モデルパスはMODELS_DIR配下に保存される
        expected_path = str(SAMSegmenter.MODELS_DIR / "sam2_b.pt")
        mock_sam_class.assert_called_once_with(expected_path)

    @patch("ultralytics.SAM")
    def test_sam_segmenter_segment_with_boxes(self, mock_sam_class, sample_image, sample_mask):
        """バウンディングボックスからマスクを生成できることを確認."""
        # モックのセットアップ
        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        # マスク結果のモック
        mock_mask_obj = MagicMock()
        mock_mask_obj.data.cpu.return_value.numpy.return_value.squeeze.return_value = (
            sample_mask.astype(float)
        )
        mock_result = MagicMock()
        mock_result.masks = [mock_mask_obj]
        mock_sam_model.predict.return_value = [mock_result]

        segmenter = SAMSegmenter()
        segmenter.load_model()

        boxes = [(100, 100, 200, 200)]
        masks = segmenter.segment_with_boxes(sample_image, boxes)

        assert isinstance(masks, list)
        assert len(masks) == 1
        assert isinstance(masks[0], np.ndarray)
        assert masks[0].dtype == np.uint8

    @patch("ultralytics.SAM")
    def test_sam_segmenter_segment_with_boxes_empty(self, mock_sam_class, sample_image):
        """空のボックスリストで空リストを返すことを確認."""
        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        segmenter = SAMSegmenter()
        segmenter.load_model()

        masks = segmenter.segment_with_boxes(sample_image, [])

        assert masks == []

    def test_sam_segmenter_segment_not_loaded(self, sample_image):
        """モデル未ロード時にRuntimeErrorが発生."""
        segmenter = SAMSegmenter()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            segmenter.segment_with_boxes(sample_image, [(10, 10, 100, 100)])


# ========================================
# DefectSegmentationRefiner テスト
# ========================================


class TestDefectSegmentationRefiner:
    """DefectSegmentationRefiner クラスのテスト."""

    @patch("ultralytics.SAM")
    def test_defect_refiner_init(self, mock_sam_class):
        """DefectSegmentationRefinerが正常に初期化できることを確認."""
        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        segmenter = SAMSegmenter(model_type="sam2_b", device="cpu")
        refiner = DefectSegmentationRefiner(segmenter)

        assert refiner.segmenter.model_type == "sam2_b"
        assert refiner.segmenter.device == "cpu"

    @patch("ultralytics.SAM")
    def test_defect_refiner_refine_detections(
        self, mock_sam_class, sample_image, sample_detections, sample_mask
    ):
        """検出結果にマスクを追加できることを確認."""
        # モックのセットアップ
        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        # マスク結果のモック（複数マスクを返す）
        mock_masks = []
        for _ in sample_detections:
            mock_mask_obj = MagicMock()
            mock_mask_obj.data.cpu.return_value.numpy.return_value.squeeze.return_value = (
                sample_mask.astype(float)
            )
            mock_masks.append(mock_mask_obj)

        mock_result = MagicMock()
        mock_result.masks = mock_masks
        mock_sam_model.predict.return_value = [mock_result]

        segmenter = SAMSegmenter()
        segmenter.load_model()
        refiner = DefectSegmentationRefiner(segmenter)

        refined = refiner.refine_detections(sample_image, sample_detections)

        # マスクが追加されたことを確認
        assert len(refined) == len(sample_detections)
        # 最初の検出結果にマスクが追加されている
        assert refined[0].mask is not None

    @patch("ultralytics.SAM")
    def test_defect_refiner_refine_empty(self, mock_sam_class, sample_image):
        """空の検出結果でも正常に処理できることを確認."""
        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        segmenter = SAMSegmenter()
        segmenter.load_model()
        refiner = DefectSegmentationRefiner(segmenter)

        refined = refiner.refine_detections(sample_image, [])

        assert refined == []

    @patch("ultralytics.SAM")
    def test_defect_refiner_calculate_mask_area(self, mock_sam_class):
        """マスク面積の計算が正しいことを確認."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1  # 20 * 30 = 600ピクセル

        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        segmenter = SAMSegmenter()
        refiner = DefectSegmentationRefiner(segmenter)
        area = refiner.calculate_mask_area(mask)

        assert area == 600

    @patch("ultralytics.SAM")
    def test_defect_refiner_calculate_mask_iou(self, mock_sam_class):
        """IoU計算が正しいことを確認."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10:50, 10:50] = 1  # 40 * 40 = 1600ピクセル

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[30:70, 30:70] = 1  # 40 * 40 = 1600ピクセル

        # 重なり領域: (30,30)-(50,50) = 20 * 20 = 400ピクセル
        # 合計領域: 1600 + 1600 - 400 = 2800ピクセル
        # IoU = 400 / 2800 = 0.142857...

        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        segmenter = SAMSegmenter()
        refiner = DefectSegmentationRefiner(segmenter)
        iou = refiner.calculate_mask_iou(mask1, mask2)

        assert pytest.approx(iou, abs=0.01) == 0.1429

    @patch("ultralytics.SAM")
    def test_defect_refiner_calculate_mask_iou_no_overlap(self, mock_sam_class):
        """重なりがない場合のIoUが0であることを確認."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10:20, 10:20] = 1

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[80:90, 80:90] = 1

        mock_sam_model = MagicMock()
        mock_sam_class.return_value = mock_sam_model

        segmenter = SAMSegmenter()
        refiner = DefectSegmentationRefiner(segmenter)
        iou = refiner.calculate_mask_iou(mask1, mask2)

        assert iou == 0.0
