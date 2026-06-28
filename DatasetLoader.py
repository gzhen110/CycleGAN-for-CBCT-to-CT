import tensorflow as tf
import numpy as np
import os
import cv2

HU_MIN = -1000.0
HU_MAX =  3000.0

def _loadNpy_core(path, dim):
    '''
    核心讀取與歸一化邏輯 — Resize 版本
    Global 歸一化：固定 HU 範圍 -1000 ~ 3000，所有切片可比
    尺寸統一：cv2.resize 雙線性插值縮放至 dim
    '''
    # 讀取 numpy 檔案
    img = np.load(path).astype(np.float32)

    # --- shape 正規化：統一轉為 (H, W) ---
    if img.ndim == 3:
        img = img[..., 0]   # (H, W, 1) 或 (H, W, 3) → 取第一個 channel

    # --- Global 歸一化：固定範圍 -1000 ~ 3000 → -1 ~ 1 ---
    img = np.clip(img, HU_MIN, HU_MAX)
    img = 2.0 * (img - HU_MIN) / (HU_MAX - HU_MIN) - 1.0

    # --- Resize 到指定尺寸（雙線性插值）---
    img = cv2.resize(img, (dim[1], dim[0]))  # cv2 順序是 (width, height)

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
    成對資料讀取器 (已適配醫學 NPY 格式)
    '''
    def __init__(self, data_1, data_2, image_path1, image_path2, batch_size=1, dim=(256, 256), n_channels=1, shuffle=True):
        self.data_1 = data_1
        self.data_2 = data_2
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_1))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_1) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_1 = [self.data_1[k] for k in indexes]
        batch_2 = [self.data_2[k] for k in indexes]
        X, y = self.__data_generation(batch_1, batch_2)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_1, batch_2):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        for i, (fname1, fname2) in enumerate(zip(batch_1, batch_2)):
            X[i,] = _loadNpy_core(os.path.join(self.image_path1, fname1), self.dim)
            y[i,] = _loadNpy_core(os.path.join(self.image_path2, fname2), self.dim)
        return X, y