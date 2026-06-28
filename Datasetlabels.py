import numpy as np
import os
import config  # 引入你設定好的 config.py

# 直接使用 config 裡面的 basedir
DATASET_ROOT = config.basedir 

def get_filtered_files(subfolder, target_patients):
    """
    這個函式會去目標資料夾抓檔案，並且只保留在 target_patients 名單內的病人切片
    """
    folder = os.path.join(DATASET_ROOT, subfolder)
    files = []
    
    # 遍歷資料夾內的所有檔案
    for f in sorted(os.listdir(folder)):
        if f.endswith('.npy'):
            # 抓出檔名前面的病人 ID (例如 "2BA001")
            patient_id = f.split('_')[0]
            
            # 判斷有沒有在訓練名單中
            if patient_id in target_patients:
                files.append(os.path.join(subfolder, f))
                
    return files

# ========================================================
# 核心執行區：透過 config.TRAIN_PATIENTS 過濾檔案
# ========================================================
domainA = get_filtered_files('trainA', config.TRAIN_PATIENTS)
domainB = get_filtered_files('trainB', config.TRAIN_PATIENTS)

# 安全檢查：確保 A 和 B 數量一致
assert len(domainA) == len(domainB), \
    f"trainA ({len(domainA)}) 與 trainB ({len(domainB)}) 數量不一致！請檢查資料集。"

print("=" * 50)
print(f"成功載入 config.py 中的特定訓練病人名單！")
print(f"共篩選出 {len(domainA)} 對切片進行訓練。")
print("=" * 50)

# ========================================================
# 變數輸出區：提供給 trainer.py 讀取的變數
# ========================================================
train_imgs = domainA