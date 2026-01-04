# AI Visual Inspection Demo

製造業向け外観検査AIデモンストレーション
YOLOv8 + SAM 2による欠陥検出・セグメンテーション

## 概要

このプロジェクトは、製造業における製品外観検査の自動化を実現するAIシステムのデモです。
YOLOv8による高速な欠陥検出と、SAM 2（Segment Anything Model 2）による高精度なセグメンテーションを組み合わせています。

### 主な機能

- **欠陥検出**: YOLOv8による6種類の欠陥クラス検出（傷、欠け、異物混入、ひび割れ、へこみ、変色）
- **セグメンテーション**: SAM 2による欠陥領域の高精度マスク生成
- **3段階判定**: 製造ラインを想定したOK/WARNING/NG判定
- **可視化**: 検出結果の視覚的レポート生成
- **設定の柔軟性**: .envファイルによるパラメータ調整

### 技術スタック

- **検出**: Ultralytics YOLOv8
- **セグメンテーション**: Ultralytics SAM 2
- **深層学習**: PyTorch + CUDA
- **画像処理**: OpenCV, Pillow
- **可視化**: Matplotlib

## 重要な注意事項

⚠️ **このプロジェクトはデモンストレーション用です**

- 欠陥検出を行うには、**事前にトレーニング済みのYOLOv8モデル**（.pt形式）が必要です
- デフォルトの `models/yolov8n.pt` は一般物体検出用モデルなので、製造欠陥は検出できません
- 実際の運用には、製造ラインの画像データを用いたカスタムモデルの学習が必須です

## 動作確認環境

### 必須環境
- **OS**: Windows 10/11
- **Python**: 3.10
- **CUDA**: 12.1

### 検証済みバージョン
```
Python:       3.10.x
torch:        2.5.1+cu121
torchvision:  0.20.1+cu121
```

## セットアップ

### 1. リポジトリのクローン

```cmd
git clone https://github.com/yourusername/ai-visual-inspection-demo.git
cd ai-visual-inspection-demo
```

### 2. 仮想環境の作成・有効化

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. 依存関係のインストール

```cmd
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### 4. (オプション) 開発環境のセットアップ

テストやコード品質チェックを行う場合：

```cmd
pip install -e ".[dev]"
```

## 使い方

**注意**: 初回起動時、指定したYOLOモデルやSAMモデルが`models/`フォルダに存在しない場合、Ultralyticsが自動的にダウンロードします。ネットワーク接続が必要です。

### 基本的な推論

```cmd
python scripts/inference_demo.py --image data/samples/test.jpg
```

### よく使うオプション

**結果画像を保存**
```cmd
python scripts/inference_demo.py --image test.jpg --output result.jpg
```

**カスタムモデルを使用**
```cmd
python scripts/inference_demo.py --image test.jpg --model models/defect_detector.pt
```

**信頼度閾値を変更**
```cmd
python scripts/inference_demo.py --image test.jpg --confidence 0.7
```

**SAMセグメンテーションを無効化（高速化）**
```cmd
python scripts/inference_demo.py --image test.jpg --no-sam
```

**CPU使用を強制**
```cmd
python scripts/inference_demo.py --image test.jpg --device cpu
```

**結果を画面表示**
```cmd
python scripts/inference_demo.py --image test.jpg --show
```

### 出力例

```
============================================================
Visual Inspection AI Demo
============================================================
Input image: test.jpg
Model: models/yolov8n.pt
Use SAM: True
Device: cuda
------------------------------------------------------------
Initializing pipeline...
Pipeline initialized successfully!

Inspecting: test.jpg
------------------------------------------------------------
RESULTS
------------------------------------------------------------
Severity: WARNING
Is Defective: True
Defect Count: 2
Processing Time: 234.5 ms

Detections:
  1. scratch
     Confidence: 85.23%
     Bounding Box: (120, 45, 180, 90)
     Area: 2700 px
     Has Mask: Yes
```

### 終了コード

スクリプトは検査結果に応じた終了コードを返します（自動化システムとの連携用）：

| 終了コード | 判定 | 説明 |
|-----------|------|------|
| 0 | OK | 欠陥なし or 軽微な欠陥 |
| 1 | WARNING | 注意が必要な欠陥 |
| 2 | NG | 重大な欠陥（不良品） |

## 環境設定（.envファイル）

プロジェクトルートに `.env` ファイルを作成することで、コードを変更せずにパラメータを調整できます。

### .envファイル例

```bash
# モデル設定
MODEL_PATH=models/defect_detector.pt
SAM_MODEL_TYPE=sam2_b

# デバイス設定
DEVICE=auto

# 検出パラメータ
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# SAMセグメンテーション
USE_SAM=true

# 重要度判定閾値
WARNING_THRESHOLD=0.3
NG_THRESHOLD=0.7

# 画像サイズ
IMAGE_SIZE=640
```

### 設定可能な環境変数

| 環境変数 | デフォルト値 | 説明 |
|---------|-------------|------|
| `MODEL_PATH` | models/yolov8n.pt | YOLOv8モデルパス |
| `SAM_MODEL_TYPE` | sam2_b | SAMモデルバリアント (sam2_t/s/b/l) |
| `DEVICE` | auto | 使用デバイス (cuda/cpu/auto) |
| `CONFIDENCE_THRESHOLD` | 0.5 | 検出の最小信頼度 (0.0-1.0) |
| `IOU_THRESHOLD` | 0.45 | NMSのIoU閾値 |
| `USE_SAM` | true | SAMセグメンテーション使用有無 |
| `WARNING_THRESHOLD` | 0.3 | WARNING判定の閾値 |
| `NG_THRESHOLD` | 0.7 | NG判定の閾値 |
| `IMAGE_SIZE` | 640 | 推論時の画像サイズ |

## プロジェクト構造

```
ai-visual-inspection-demo/
├── src/inspection/          # メインパッケージ
│   ├── models/              # YOLODetector, SAMSegmenter
│   ├── pipeline/            # InspectionPipeline
│   ├── preprocessing/       # ImageLoader
│   ├── visualization/       # ResultVisualizer
│   └── config/              # Settings管理
├── scripts/
│   └── inference_demo.py    # デモスクリプト
├── tests/                   # pytestテストコード
├── models/                  # 学習済みモデル (.pt)
├── data/                    # サンプル画像
├── .env                     # 環境設定（gitignore）
├── pyproject.toml           # プロジェクト定義
└── README.md
```

## 欠陥クラス定義

| ID | 名称 | 説明 |
|----|------|------|
| 0 | scratch | 傷 |
| 1 | chip | 欠け |
| 2 | contamination | 異物混入 |
| 3 | crack | ひび割れ |
| 4 | dent | へこみ |
| 5 | discoloration | 変色 |

## 重要度判定ロジック

- **OK**: 欠陥なし、または信頼度 < 0.3
- **WARNING**: 信頼度 0.3～0.7（要確認）
- **NG**: 信頼度 ≥ 0.7（不良品）

## テスト実行

**全テスト実行**
```cmd
pytest
```

**カバレッジ付き**
```cmd
pytest --cov=src/inspection --cov-report=html
```

**カバレッジレポート表示**
```cmd
start htmlcov\index.html
```

## 設計思想

### アーキテクチャ原則

1. **レイヤードアーキテクチャ**: 責務の明確な分離
2. **依存性注入**: テスタビリティの向上
3. **遅延初期化**: メモリ効率化
4. **型安全性**: データクラス + Type hints
5. **設定外部化**: .envによる柔軟な構成管理

## ライセンス

このプロジェクトはポートフォリオ用デモです。商用利用には制限があります。

## 参考文献

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
