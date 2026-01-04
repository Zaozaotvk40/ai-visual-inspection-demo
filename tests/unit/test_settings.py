"""
設定（settings.py）のユニットテスト.

Settings クラスと get_settings 関数のテストを実施します。
環境変数による設定上書きと、デバイス自動選択のテストも含みます。
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from inspection.config.settings import Settings, get_settings


# ========================================
# Settings テスト
# ========================================


class TestSettings:
    """Settings クラスのテスト."""

    def test_settings_defaults(self):
        """デフォルト値が正しく設定されることを確認."""
        settings = Settings()

        assert settings.model_path == Path("models/yolov8n.pt")
        assert settings.sam_model_type == "sam2_b"
        assert settings.device == "auto"
        assert settings.confidence_threshold == 0.5
        assert settings.iou_threshold == 0.45
        assert settings.use_sam is True
        assert settings.warning_threshold == 0.3
        assert settings.ng_threshold == 0.7
        assert settings.image_size == 640

    def test_settings_custom_values(self):
        """カスタム値で初期化できることを確認."""
        settings = Settings(
            model_path=Path("custom/model.pt"),
            sam_model_type="sam2_s",
            device="cpu",
            confidence_threshold=0.6,
            iou_threshold=0.5,
            use_sam=False,
            warning_threshold=0.4,
            ng_threshold=0.8,
            image_size=1024,
        )

        assert settings.model_path == Path("custom/model.pt")
        assert settings.sam_model_type == "sam2_s"
        assert settings.device == "cpu"
        assert settings.confidence_threshold == 0.6
        assert settings.iou_threshold == 0.5
        assert settings.use_sam is False
        assert settings.warning_threshold == 0.4
        assert settings.ng_threshold == 0.8
        assert settings.image_size == 1024

    def test_settings_env_override(self, monkeypatch):
        """環境変数で設定を上書きできることを確認."""
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.75")
        monkeypatch.setenv("IOU_THRESHOLD", "0.6")
        monkeypatch.setenv("USE_SAM", "false")
        monkeypatch.setenv("IMAGE_SIZE", "800")

        # 新しいインスタンスを作成（環境変数を読み込む）
        settings = Settings()

        assert settings.confidence_threshold == 0.75
        assert settings.iou_threshold == 0.6
        assert settings.use_sam is False
        assert settings.image_size == 800

    @pytest.mark.parametrize("invalid_conf", [-0.1, 1.5, 2.0])
    def test_settings_validation_confidence_threshold(self, invalid_conf):
        """confidence_threshold が0-1の範囲外でValidationErrorが発生."""
        with pytest.raises(ValidationError):
            Settings(confidence_threshold=invalid_conf)

    @pytest.mark.parametrize("invalid_iou", [-0.1, 1.5, 2.0])
    def test_settings_validation_iou_threshold(self, invalid_iou):
        """iou_threshold が0-1の範囲外でValidationErrorが発生."""
        with pytest.raises(ValidationError):
            Settings(iou_threshold=invalid_iou)

    @pytest.mark.parametrize("invalid_warning", [-0.1, 1.5, 2.0])
    def test_settings_validation_warning_threshold(self, invalid_warning):
        """warning_threshold が0-1の範囲外でValidationErrorが発生."""
        with pytest.raises(ValidationError):
            Settings(warning_threshold=invalid_warning)

    @pytest.mark.parametrize("invalid_ng", [-0.1, 1.5, 2.0])
    def test_settings_validation_ng_threshold(self, invalid_ng):
        """ng_threshold が0-1の範囲外でValidationErrorが発生."""
        with pytest.raises(ValidationError):
            Settings(ng_threshold=invalid_ng)

    @pytest.mark.parametrize("invalid_size", [0, 10, 31, -100])
    def test_settings_validation_image_size(self, invalid_size):
        """image_size が32未満でValidationErrorが発生."""
        with pytest.raises(ValidationError):
            Settings(image_size=invalid_size)

    def test_settings_validation_sam_model_type_invalid(self):
        """無効なsam_model_typeでValidationErrorが発生."""
        with pytest.raises(ValidationError):
            Settings(sam_model_type="invalid_model")

    def test_settings_validation_device_invalid(self):
        """無効なdeviceでValidationErrorが発生."""
        with pytest.raises(ValidationError):
            Settings(device="invalid_device")

    @patch("torch.cuda.is_available")
    def test_get_device_auto_cuda(self, mock_cuda):
        """device='auto'でCUDA利用可能時にcudaを返す."""
        mock_cuda.return_value = True

        settings = Settings(device="auto")
        device = settings.get_device()

        assert device == "cuda"
        mock_cuda.assert_called_once()

    @patch("torch.cuda.is_available")
    def test_get_device_auto_cpu(self, mock_cuda):
        """device='auto'でCUDA利用不可時にcpuを返す."""
        mock_cuda.return_value = False

        settings = Settings(device="auto")
        device = settings.get_device()

        assert device == "cpu"
        mock_cuda.assert_called_once()

    def test_get_device_explicit_cuda(self):
        """device='cuda'が明示的に設定されている場合."""
        settings = Settings(device="cuda")
        device = settings.get_device()

        assert device == "cuda"

    def test_get_device_explicit_cpu(self):
        """device='cpu'が明示的に設定されている場合."""
        settings = Settings(device="cpu")
        device = settings.get_device()

        assert device == "cpu"


# ========================================
# get_settings() テスト
# ========================================


class TestGetSettings:
    """get_settings() 関数のテスト."""

    def test_get_settings_returns_settings(self):
        """get_settings()がSettingsインスタンスを返すことを確認."""
        # キャッシュをクリア
        get_settings.cache_clear()

        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_get_settings_cached(self):
        """get_settings()が同じインスタンスを返すことを確認（キャッシュ動作）."""
        # キャッシュをクリア
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        # 同一インスタンス
        assert settings1 is settings2

    def test_get_settings_with_env_override(self, monkeypatch):
        """環境変数を使ったget_settings()の動作確認."""
        # キャッシュをクリア
        get_settings.cache_clear()

        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")
        monkeypatch.setenv("IMAGE_SIZE", "1024")

        settings = get_settings()

        assert settings.confidence_threshold == 0.8
        assert settings.image_size == 1024

        # テスト後にキャッシュをクリア
        get_settings.cache_clear()
