# Chest X-ray 肺炎分類（ResNet 遷移學習 / PyTorch）

此專案使用 Kaggle 的 Chest X-Ray Pneumonia 資料集，透過 ResNet（`resnet18` 或 `resnet50`）進行遷移學習，完成二元分類：

- `NORMAL`: 正常胸腔 X 光
- `PNEUMONIA`: 肺炎胸腔 X 光

資料集連結：
`https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data`

## 1) 安裝套件

在 `C:\Users\Edward\Desktop\Classifier` 下執行：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 下載與放置資料集

期望資料夾結構如下（重點是 `train/`, `val/`, `test/`）：

```text
C:\Users\Edward\Desktop\Classifier\chest_xray
├── train
│   ├── NORMAL
│   └── PNEUMONIA
├── val
│   ├── NORMAL
│   └── PNEUMONIA
└── test
    ├── NORMAL
    └── PNEUMONIA
```

## 3) 開始訓練

### 基本版（resnet18）

```powershell
python train_resnet_pneumonia.py --dataset_root "C:\Users\Edward\Desktop\Classifier\chest_xray"
```

### 進階版（先凍結 backbone，再解凍微調）

```powershell
python train_resnet_pneumonia.py `
  --dataset_root "C:\Users\Edward\Desktop\Classifier\chest_xray" `
  --model_name resnet50 `
  --epochs 15 `
  --batch_size 32 `
  --freeze_backbone `
  --unfreeze_epoch 4
```

訓練輸出預設在：
`C:\Users\Edward\Desktop\Classifier\outputs`

輸出檔案包含：

- `best_model.pth`：最佳驗證 F1 的模型權重
- `metadata.json`：模型設定、類別映射、影像大小
- `history.json`：每個 epoch 的訓練/驗證指標

## 4) 單張影像推論

```powershell
python infer_single.py --image_path "C:\path\to\your\xray_image.jpeg"
```

可自行指定模型與 metadata 路徑：

```powershell
python infer_single.py `
  --image_path "C:\path\to\your\xray_image.jpeg" `
  --model_path "C:\Users\Edward\Desktop\Classifier\outputs\best_model.pth" `
  --metadata_path "C:\Users\Edward\Desktop\Classifier\outputs\metadata.json"
```

## 5) 備註

- 本訓練程式已使用 ImageNet 預訓練權重做遷移學習。
- 由於資料集有類別不平衡，loss 會自動使用 class weights。
- 在 Windows 上若 `num_workers` 過大可能不穩定，建議先使用 `0`。

