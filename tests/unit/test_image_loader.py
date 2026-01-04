"""
画像読み込み（image_loader.py）のユニットテスト.

ImageLoaderクラスの各メソッドをテストします。
"""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from inspection.preprocessing.image_loader import ImageLoader


# ========================================
# ImageLoader 初期化テスト
# ========================================


class TestImageLoaderInit:
    """ImageLoader初期化のテスト."""

    def test_image_loader_init_defaults(self):
        """デフォルト初期化が正しいことを確認."""
        loader = ImageLoader()

        assert loader.target_size == (640, 640)
        assert loader.maintain_aspect is True

    def test_image_loader_init_custom_size(self):
        """カスタムサイズで初期化."""
        loader = ImageLoader(target_size=(1024, 768))

        assert loader.target_size == (1024, 768)

    def test_image_loader_init_no_resize(self):
        """target_size=Noneでリサイズなし."""
        loader = ImageLoader(target_size=None)

        assert loader.target_size is None

    def test_image_loader_init_no_maintain_aspect(self):
        """maintain_aspect=Falseでアスペクト比維持なし."""
        loader = ImageLoader(maintain_aspect=False)

        assert loader.maintain_aspect is False


# ========================================
# ImageLoader.load() テスト
# ========================================


class TestImageLoaderLoad:
    """ImageLoader.load()のテスト."""

    def test_load_file_success(self, temp_image_file):
        """ファイルから正常に読み込めることを確認."""
        loader = ImageLoader(target_size=None)
        image = loader.load(temp_image_file)

        assert isinstance(image, np.ndarray)
        assert image.shape == (640, 640, 3)
        assert image.dtype == np.uint8

    def test_load_file_with_resize(self, temp_image_file):
        """リサイズが適用されることを確認."""
        loader = ImageLoader(target_size=(320, 320), maintain_aspect=False)
        image = loader.load(temp_image_file)

        assert image.shape == (320, 320, 3)

    def test_load_file_not_found(self):
        """ファイルが存在しない場合にFileNotFoundErrorが発生."""
        loader = ImageLoader()
        non_existent_path = Path("nonexistent_image_12345.jpg")

        with pytest.raises(FileNotFoundError, match="Image not found"):
            loader.load(non_existent_path)

    def test_load_invalid_file(self, tmp_path):
        """無効な画像ファイルでValueErrorが発生."""
        # テキストファイルを画像として読み込もうとする
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("This is not an image")

        loader = ImageLoader()

        with pytest.raises(ValueError, match="Failed to load image"):
            loader.load(invalid_file)


# ========================================
# ImageLoader.load_from_bytes() テスト
# ========================================


class TestImageLoaderLoadFromBytes:
    """ImageLoader.load_from_bytes()のテスト."""

    def test_load_from_bytes_success(self, sample_image):
        """バイト列から正常に読み込めることを確認."""
        # 画像をバイト列に変換
        bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", bgr)
        assert success
        image_bytes = buffer.tobytes()

        loader = ImageLoader(target_size=None)
        image = loader.load_from_bytes(image_bytes)

        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # RGB
        assert image.dtype == np.uint8

    def test_load_from_bytes_with_resize(self, sample_image):
        """バイト列読み込みでリサイズが適用されることを確認."""
        bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", bgr)
        assert success
        image_bytes = buffer.tobytes()

        loader = ImageLoader(target_size=(320, 320), maintain_aspect=False)
        image = loader.load_from_bytes(image_bytes)

        assert image.shape == (320, 320, 3)

    def test_load_from_bytes_invalid(self):
        """無効なバイト列でValueErrorが発生."""
        invalid_bytes = b"This is not an image"

        loader = ImageLoader()

        with pytest.raises(ValueError, match="Failed to decode image from bytes"):
            loader.load_from_bytes(invalid_bytes)


# ========================================
# ImageLoader.load_from_pil() テスト
# ========================================


class TestImageLoaderLoadFromPIL:
    """ImageLoader.load_from_pil()のテスト."""

    def test_load_from_pil_rgb(self, sample_image):
        """PIL Image（RGB）から正常に変換できることを確認."""
        pil_image = Image.fromarray(sample_image)

        loader = ImageLoader(target_size=None)
        image = loader.load_from_pil(pil_image)

        assert isinstance(image, np.ndarray)
        assert image.shape == sample_image.shape
        assert np.array_equal(image, sample_image)

    def test_load_from_pil_non_rgb(self):
        """非RGB PIL Image（L, RGBA等）が正しくRGBに変換されることを確認."""
        # グレースケール画像を作成
        grayscale_pil = Image.new("L", (100, 100), color=128)

        loader = ImageLoader(target_size=None)
        image = loader.load_from_pil(grayscale_pil)

        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8

    def test_load_from_pil_with_resize(self):
        """PIL Image読み込みでリサイズが適用されることを確認."""
        pil_image = Image.new("RGB", (1000, 800), color=(255, 0, 0))

        loader = ImageLoader(target_size=(200, 200), maintain_aspect=False)
        image = loader.load_from_pil(pil_image)

        assert image.shape == (200, 200, 3)


# ========================================
# ImageLoader._resize() テスト
# ========================================


class TestImageLoaderResize:
    """ImageLoader._resize()のテスト."""

    def test_resize_without_maintain_aspect(self, large_image):
        """maintain_aspect=Falseで直接リサイズされることを確認."""
        loader = ImageLoader(target_size=(640, 640), maintain_aspect=False)
        resized = loader._resize(large_image, (640, 640))

        assert resized.shape == (640, 640, 3)

    def test_resize_with_maintain_aspect(self, large_image):
        """maintain_aspect=Trueでパディング付きリサイズされることを確認."""
        loader = ImageLoader(target_size=(640, 640), maintain_aspect=True)
        resized = loader._resize(large_image, (640, 640))

        # パディングされて正確に(640, 640)になる
        assert resized.shape == (640, 640, 3)


# ========================================
# ImageLoader._resize_with_padding() テスト
# ========================================


class TestImageLoaderResizeWithPadding:
    """ImageLoader._resize_with_padding()のテスト."""

    def test_resize_with_padding_landscape(self):
        """横長画像のパディング付きリサイズ."""
        # 横長画像 (1000x500)
        image = np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8)

        loader = ImageLoader(target_size=(640, 640), maintain_aspect=True)
        resized = loader._resize_with_padding(image, (640, 640))

        # 正確に640x640になる
        assert resized.shape == (640, 640, 3)

        # 上下にパディングがあることを確認（中央にパディング）
        # 元画像のアスペクト比は2:1なので、幅640に合わせると高さは320
        # 残り320はパディング（上下160ずつ）

    def test_resize_with_padding_portrait(self):
        """縦長画像のパディング付きリサイズ."""
        # 縦長画像 (500x1000)
        image = np.random.randint(0, 255, (1000, 500, 3), dtype=np.uint8)

        loader = ImageLoader(target_size=(640, 640), maintain_aspect=True)
        resized = loader._resize_with_padding(image, (640, 640))

        # 正確に640x640になる
        assert resized.shape == (640, 640, 3)

    def test_resize_with_padding_square(self, sample_image):
        """正方形画像のパディング付きリサイズ."""
        loader = ImageLoader(target_size=(800, 800), maintain_aspect=True)
        resized = loader._resize_with_padding(sample_image, (800, 800))

        # 正方形なのでパディングなしでリサイズ
        assert resized.shape == (800, 800, 3)

    def test_resize_with_padding_custom_pad_color(self, sample_image):
        """カスタムパディングカラーが適用されることを確認."""
        loader = ImageLoader(target_size=(1000, 1000), maintain_aspect=True)
        resized = loader._resize_with_padding(sample_image, (1000, 1000), pad_color=0)

        # パディング部分が0（黒）であることを確認
        # sample_imageは640x640、1000x1000にリサイズするとパディングが発生
        # 上下左右にパディングがある
        assert resized.shape == (1000, 1000, 3)


# ========================================
# ImageLoader.get_original_size() テスト
# ========================================


class TestImageLoaderGetOriginalSize:
    """ImageLoader.get_original_size()のテスト."""

    def test_get_original_size(self, temp_image_file):
        """元のサイズを取得できることを確認."""
        loader = ImageLoader()
        width, height = loader.get_original_size(temp_image_file)

        # temp_image_fileは640x640の画像
        assert width == 640
        assert height == 640

    def test_get_original_size_different_size(self, tmp_path):
        """異なるサイズの画像でも正しく取得できることを確認."""
        # 1920x1080の画像を作成
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        image_path = tmp_path / "test_1920x1080.jpg"

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), bgr)

        loader = ImageLoader()
        width, height = loader.get_original_size(image_path)

        assert width == 1920
        assert height == 1080
