# Chest X-ray 肺炎分類（EfficientNet_B3 遷移學習 / PyTorch）

此專案使用 Kaggle 的 Chest X-Ray Pneumonia 資料集，透過 EfficientNet_B3 進行遷移學習，完成二元分類：

- `NORMAL`: 正常胸腔 X 光
- `PNEUMONIA`: 肺炎胸腔 X 光

資料集連結：
`https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data`

## 1) 安裝套件

在專案根目錄下執行：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 模組說明

- `constant.py`：共用常數（ImageNet mean/std、模型名稱對照表、正類別名稱）。
- `data/dataset.py`：資料集下載（`kagglehub`）、影像 transform、`ImageFolder` 建構、`DataLoader` 建構。
- `engine/trainer.py`：單一 epoch 訓練/驗證迴圈、完整訓練迴圈（含 backbone 解凍排程、early stopping）、指標計算。
- `engine/visualize.py`：訓練曲線、混淆矩陣繪圖。
- `models/builder.py`：模型建構（`resnet18`/`resnet50`/`efficientnet_b3`，可切換 backbone 凍結）。
- `models/inference.py`：批次推論（softmax + argmax）。
- `utils.py`：亂數種子、計時器、輸出目錄與結果（權重/metadata/history/圖表）的儲存。

## 3) 下載與放置資料集

資料集透過 notebook 內的 `download_dataset()` 以 `kagglehub` 自動下載，也可以手動放置成以下結構（重點是 `train/`, `val/`, `test/`）：

```text
chest_xray/
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

## 4) 開始訓練

訓練以 `model/efficient_train.ipynb` 進行（本專案主要以 notebook 訓練，無 CLI 訓練腳本）。打開 notebook 後依序執行：

- **Cell 1**：環境設定（import 共用模組、選擇裝置）
- **Cell 2**：實驗參數（要換模型/超參數只改這裡，改完重跑 Cell 4 之後）
- **Cell 3**：下載資料集
- **Cell 4**：建立資料集、DataLoader、模型、optimizer
- **Cell 5**：呼叫 `train_model(...)` 開始訓練
- **Cell 6**：測試集評估
- **Cell 7**：訓練曲線與混淆矩陣
- **Cell 8**：儲存結果

訓練輸出預設在 `outputs/<experiment_name>_<timestamp>/`，包含：

- `best_model.pth`：最佳驗證 F1 的模型權重
- `metadata.json`：模型設定、類別映射、影像大小、測試集指標
- `history.json`：每個 epoch 的訓練/驗證指標
- `training_curves.png`、`confusion_matrix.png`：視覺化圖表

## 5) 單張影像推論

```powershell
python infer_single.py `
  --image_path "C:\path\to\your\xray_image.jpeg" `
  --model_path "outputs\<experiment_name>_<timestamp>\best_model.pth" `
  --metadata_path "outputs\<experiment_name>_<timestamp>\metadata.json"
```

`infer_single.py` 會依 `metadata.json` 裡的 `model_name` 自動選擇正確的模型架構（`resnet18`/`resnet50`/`efficientnet_b3`）。

## 6) 備註

- 本訓練程式已使用 ImageNet 預訓練權重做遷移學習。
- 由於資料集有類別不平衡，loss 會自動使用 class weights。
- 在 Windows 上若 `num_workers` 過大可能不穩定，建議先使用 `0`。
