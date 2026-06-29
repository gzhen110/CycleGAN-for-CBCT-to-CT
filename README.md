# CycleGAN with Self-Attention for CBCT-to-CT Image Synthesis

## 專案來源 (Original Source)

本專案的原始碼來自 **[victor369basu/CycleGAN-with-Self-Attention](https://github.com/victor369basu/CycleGAN-with-Self-Attention)**，由 **Victor Basu** 所開發。

原作者在 U-Net 架構的 CycleGAN 中嵌入了 Self-Attention 層，靈感來自論文 [Self-Attention Generative Adversarial Networks（SAGAN）](https://arxiv.org/pdf/1805.08318v2.pdf) 與 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)。Self-Attention 層被加入 Generator 的 bottleneck 與最終輸出卷積層之前，以及 Discriminator 的 zero-padding 層與輸出層之前，以強化模型對全局特徵（而非僅局部卷積特徵）的捕捉能力。

原作者以此架構解決了三個不同的自然影像轉換任務：
- 臉部草圖上色（Sketch Colorization）
- 男女臉部轉換（Gender Transformation）
- 眼鏡移除（Glasses Removal）

模型以 256×256 的影像大小、訓練 800 個 epoch，使用 TensorFlow 實作，整體架構不針對單一任務設計，而是以相同的網路架構處理多種任務。

---

## 本專案修改動機 (Motivation for This Fork)

本專案將上述架構應用於**醫學影像合成**領域，具體任務為：

- **輸入（Domain A）**：錐束電腦斷層掃描（CBCT，Cone Beam CT）
- **輸出（Domain B）**：模擬標準電腦斷層掃描（sCT，Synthetic CT）

CBCT 影像雖然成本低、輻射劑量小，但影像品質（對比度、HU 值精準度）遠不如標準 CT，直接用於放射治療計劃會導致劑量計算誤差。因此本研究目標為訓練一個模型，將 CBCT 影像轉換為接近真實 CT 品質的 sCT 影像，以提升後續放射治療計劃的準確性。

---

## 主要修改項目 (Modifications)

相較於原始碼，本專案針對醫學影像任務進行了以下調整：

### 1. 輸入通道調整（Grayscale Single-Channel Input）
原始碼預設處理 RGB 三通道影像；CBCT/CT 為單通道灰階影像，因此將 Generator 與 Discriminator 的輸入通道數修改為 **1 channel**，並更新 `config.py` 對應設定。

### 2. 資料集讀取器（Custom DatasetLoader）
針對本研究的資料格式（`.npy` 切片檔案）重新撰寫資料讀取流程，包含：
- **`DatasetLoader.py`**：全域 HU 值正規化 + zero-padding（保留原始長寬比，避免形變），搭配 `--height 320 --width 320`

HU 值範圍固定為 **-1000 ~ 3000**，以確保不同病患間的一致性。

### 3. 配對資料生成器（Paired Data Generator）
原本的 `GANDataGenerator` 為非配對（unpaired）隨機抽樣；本研究新增 `GANDataGeneratorXY`，確保評估時 CBCT 與 CT 切片索引正確對應。

### 4. 訓練監控與 Log 控制
實作自定義 Keras Callback **`EpochLogger`**，每個 epoch 僅輸出一行摘要（含各項 loss 數值），方便長時間訓練監控。

### 5. Checkpoint 機制調整
採用 `tf.train.Checkpoint` 搭配具名鍵值（`c_gen`、`s_gen` 等）進行模型存檔，每 10 個 epoch 儲存完整模型（`epoch_010.ckpt`，約 346MB），另外單獨儲存 `c_generator` 的權重（`model_weights`，約 45MB）供推論使用。

載入方式統一使用 `.expect_partial()`，以避免物件圖不匹配或 HDF5 格式造成的卡住問題。

### 6. 評估腳本（Evaluation Script）
撰寫 `temp_evaluate.py` 針對固定測試切片（4 位病患，放置於 `test_samples/`）進行定量評估，輸出指標如下：

| 指標 | 說明 |
|------|------|
| PSNR | Peak Signal-to-Noise Ratio |
| SSIM | Structural Similarity Index |
| MAE（HU） | Mean Absolute Error |
| RMSE（HU） | Root Mean Squared Error |

同時將 CBCT / sCT / 真實 CT 並排輸出為 PNG，儲存至 `./evaluation_results/`。

### 7. 學習率衰減（Linear LR Decay）
實作線性學習率衰減 Callback，於訓練後半段逐步降低學習率，以利模型收斂。

### 8. 判別器學習率調整（Anti Mode Collapse）
原始設定 `dis_lr = 0.0002` 導致 Discriminator 過強而發生 Mode Collapse（輸出黑白影像）；將判別器學習率調降為 `0.00002` 後訓練恢復正常。

---

## 資料集說明 (Dataset)

| 項目 | 內容 |
|------|------|
| 病患數 | 169 位 |
| 切片總數 | 約 18,532 張 |
| 資料格式 | `.npy`（NumPy array，單通道） |
| 命名規則 | `{PatientID}_{slice_number:03d}.npy` |
| 目錄結構 | `trainA/`（CBCT）、`trainB/`（CT） |
| HU 值範圍 | -1000 ~ 3000（固定） |


---

## 環境需求 (Requirements)

請參考 `requirements.txt`。

```
tensorflow>=2.X.X
numpy<2.0
scikit-image
matplotlib
```

---

## 訓練方式 (Training)

```bash
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --dataset_path /path/to/dataset \
    --channels 1 \
    --batch_size [填入] \
    --epochs [填入] \
    --height 320 \
    --width 320 \
    --train > train.log 2>&1 &
```

---

## 評估方式 (Evaluation)

```bash
python temp_evaluate.py --checkpoint_path ./checkpoints/epoch_XXX.ckpt
```

---

## 專案結構 (Project Structure)

```
.
├── main.py
├── config.py
├── trainer.py
├── CycleGAN.py
├── generator.py
├── discriminator.py
├── SelfAttention.py
├── DatasetLoader.py
├── Datasetlabels.py
├── evaluate_testset.py
├── temp_evaluate.py
├── test_samples/           # 固定測試切片
├── evaluation_results/     # 評估輸出圖片
└── checkpoints/            # 模型 checkpoint
```

---

## 引用 (Citation)

原始碼來源：

```
@misc{victor369basu_cyclegan_attention,
  author = {Victor Basu},
  title  = {CycleGAN with Self-Attention Layers},
  year   = {2021},
  url    = {https://github.com/victor369basu/CycleGAN-with-Self-Attention}
}
```


