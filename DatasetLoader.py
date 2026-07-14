import sys
import tensorflow as tf
import numpy as np
import os
import cv2

HU_MIN = -1000.0
HU_MAX =  3000.0

def _loadNpy_core(path, dim):
    '''
    核心讀取與歸一化邏輯 — Padding 版本
    Global 歸一化：固定 HU 範圍 -1000 ~ 3000，所有切片可比
    尺寸統一：四周補 -1.0（對應空氣 HU 值），原始像素完全不變
    '''
    # 讀取 numpy 檔案
    img = np.load(path).astype(np.float32)

    # --- shape 正規化：統一轉為 (H, W) ---
    if img.ndim == 3:
        img = img[..., 0]   # (H, W, 1) 或 (H, W, 3) → 取第一個 channel

    # --- Global 歸一化：固定範圍 -1000 ~ 3000 → -1 ~ 1 ---
    img = np.clip(img, HU_MIN, HU_MAX)
    img = 2.0 * (img - HU_MIN) / (HU_MAX - HU_MIN) - 1.0

    # --- Padding 到指定尺寸（原始像素不動）---
    target_h, target_w = dim
    h, w = img.shape

    # 保護性處理：若切片超出 target，先 resize 縮小
    if h > target_h or w > target_w:
        img = cv2.resize(img, (target_w, target_h))
    else:
        pad_top    = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left   = (target_w - w) // 2
        pad_right  = target_w - w - pad_left
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=-1.0   # -1.0 對應歸一化後的空氣/背景值
        )

    # 增加通道維度 (H, W) -> (H, W, 1)
    img = np.expand_dims(img, axis=-1)
    return img

class GANDataGenerator(tf.keras.utils.Sequence):
    '''
    補回原本缺失的單一資料讀取器 (已適配醫學 NPY 格式)
    '''
    def __init__(self, data, image_path, batch_size=1, dim=(256, 256), n_channels=1, shuffle=True):
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
                 dim=(256, 256), n_channels=1, shuffle=True, paired=True):
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