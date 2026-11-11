import sys
import tensorflow as tf
import numpy as np
import os

PAD_VALUE = -1.0   # 背景值：資料集已套用 mask，身體外為 -1

def _loadNpy_core(path, dim):
    '''
    核心讀取邏輯 — 資料已在前處理階段歸一化到 [-1, 1]，這裡不再做任何數值轉換
    尺寸統一：四周補 -1.0（與背景相同），原始像素完全不變
    '''
    # 讀取 numpy 檔案
    img = np.load(path).astype(np.float32)

    # --- shape 正規化：統一轉為 (H, W) ---
    if img.ndim == 3:
        img = img[..., 0]   # (H, W, 1) 或 (H, W, 3) → 取第一個 channel

    # --- 安全檢查：資料必須已經歸一化到 [-1, 1] ---
    # 若誤用尚未歸一化的資料（例如原始 HU），直接報錯，避免默默訓練出錯誤的模型
    if img.min() < -1.01 or img.max() > 1.01:
        raise ValueError(
            f"{path} 的數值範圍 {img.min():.2f} ~ {img.max():.2f} 不在 [-1, 1]，"
            f"請確認資料集已完成歸一化"
        )

    # --- Padding 到指定尺寸（原始像素不動）---
    target_h, target_w = dim
    h, w = img.shape

    # 修改：移除原本的 resize 保護（會讓影像變形、內插混合 HU）
    # 切片超出目標尺寸時直接報錯，請調大 --height / --width（需為 16 的倍數）
    if h > target_h or w > target_w:
        raise ValueError(
            f"{path} 的大小 {h}x{w} 超過目標尺寸 {target_h}x{target_w}，"
            f"請調大 --height / --width"
        )

    pad_top    = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left   = (target_w - w) // 2
    pad_right  = target_w - w - pad_left
    img = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=PAD_VALUE
    )

    # 增加通道維度 (H, W) -> (H, W, 1)
    img = np.expand_dims(img, axis=-1)
    return img

class GANDataGenerator(tf.keras.utils.Sequence):
    '''
    補回原本缺失的單一資料讀取器 (已適配醫學 NPY 格式)
    '''
    def __init__(self, data, image_path, batch_size=1, dim=(304, 304), n_channels=1, shuffle=True):
        self.data = data
        self.image_path = image_path
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.data[k] for k in indexes]
        X, y = self.__data_generation(batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        for i, fname in enumerate(batch):
            full_path = os.path.join(self.image_path, fname)
            img_data = _loadNpy_core(full_path, self.dim)
            X[i,] = img_data
            y[i,] = img_data  # 單圖輸入時，輸出暫時對齊
        return X, y


class GANDataGeneratorXY(tf.keras.utils.Sequence):
    '''
    成對 / 非成對 資料讀取器 (已適配醫學 NPY 格式)

    paired=True  (預設，與原本行為完全一致)：
        data_1[k]、data_2[k] 永遠共用同一個 index k，
        即使每個 epoch 打亂順序，A[k]-B[k] 的對應關係也不會被打散
        （適用於已配準的資料，例如 SynthRAD2023 剛性配準後的 CBCT-CT）

    paired=False (unpaired，對應原論文 "Random Shuffling" 的做法)：
        data_1、data_2 各自擁有獨立的 index 陣列，
        每個 epoch 開始前分別各自 shuffle，
        徹底切斷 A[k]-B[k] 的空間對應關係，
        讓模型純粹依賴 cycle-consistency loss 學習跨模態映射
    '''
    def __init__(self, data_1, data_2, image_path1, image_path2, batch_size=1,
                 dim=(304, 304), n_channels=1, shuffle=True, paired=True):
        self.data_1 = data_1
        self.data_2 = data_2
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.paired = paired

        # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔，跟專案其他訊息一致
        mode_str = "Paired" if self.paired else "Unpaired"
        print(f"[GANDataGeneratorXY] 目前訓練模式：{mode_str} | "
              f"domain1={len(self.data_1)} 筆，domain2={len(self.data_2)} 筆",
              file=sys.stderr, flush=True)

        self.indexes_1 = np.arange(len(self.data_1))
        self.indexes_2 = np.arange(len(self.data_2))
        self.on_epoch_end()

    def __len__(self):
        # unpaired 時以較短的 domain 為準，避免其中一個 domain index 越界
        return int(np.floor(min(len(self.data_1), len(self.data_2)) / self.batch_size))

    def __getitem__(self, index):
        idx_1 = self.indexes_1[index * self.batch_size:(index + 1) * self.batch_size]
        idx_2 = self.indexes_2[index * self.batch_size:(index + 1) * self.batch_size]
        batch_1 = [self.data_1[k] for k in idx_1]
        batch_2 = [self.data_2[k] for k in idx_2]
        X, y = self.__data_generation(batch_1, batch_2)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            if self.paired:
                # paired：A、B 共用同一組打亂後的 index，對應關係不變
                np.random.shuffle(self.indexes_1)
                self.indexes_2 = self.indexes_1.copy()
            else:
                # unpaired：A、B 各自獨立打亂，徹底切斷一對一對應
                np.random.shuffle(self.indexes_1)
                np.random.shuffle(self.indexes_2)

    def __data_generation(self, batch_1, batch_2):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        for i, (fname1, fname2) in enumerate(zip(batch_1, batch_2)):
            X[i,] = _loadNpy_core(os.path.join(self.image_path1, fname1), self.dim)
            y[i,] = _loadNpy_core(os.path.join(self.image_path2, fname2), self.dim)
        return X, y