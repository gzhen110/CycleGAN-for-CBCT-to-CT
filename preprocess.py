import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def process_all_patients():
    # 來源路徑：組員共用的 3D 原始資料路徑
    src_root = '/home/qjanet/dataset/CBCT2CT/brain'
    
    # 目標路徑：你個人 data 資料夾下的新目錄
    dst_root = '/home/qjanet/data/medical_npy' 
    
    # 建立訓練所需的子資料夾
    os.makedirs(os.path.join(dst_root, 'trainA'), exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'trainB'), exist_ok=True)

    # 1. 自動撈出 src_root 裡面的所有病患資料夾
    # 排除隱藏檔案，確保只抓到真正的病患目錄
    all_patients = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    all_patients.sort() # 排序一下，使整齊
    
    print(f"--- 找到總共 {len(all_patients)} 位患者，開始全量切片預處理 ---")

    # 2. 用大迴圈一個一個處理
    for idx, target_patient in enumerate(all_patients):
        print(f"\n[{idx+1}/{len(all_patients)}] 正在處理患者: {target_patient}")
        
        p_path = os.path.join(src_root, target_patient)
        cbct_file = os.path.join(p_path, 'cbct.nii.gz')
        ct_file = os.path.join(p_path, 'ct.nii.gz')
        
        if os.path.exists(cbct_file) and os.path.exists(ct_file):
            try:
                # 讀取 3D 影像數據
                cbct_data = nib.load(cbct_file).get_fdata()
                ct_data = nib.load(ct_file).get_fdata()

                # 沿著 Z 軸進行切片
                for z in tqdm(range(cbct_data.shape[2]), desc=f"{target_patient} 切片中", leave=False):
                    slice_cbct = cbct_data[:, :, z].astype(np.float32)
                    slice_ct = ct_data[:, :, z].astype(np.float32)
                    
                    # 過濾空氣區域（背景）
                    if np.max(slice_cbct) > np.min(slice_cbct):
                        np.save(os.path.join(dst_root, 'trainA', f"{target_patient}_{z:03d}.npy"), slice_cbct)
                        np.save(os.path.join(dst_root, 'trainB', f"{target_patient}_{z:03d}.npy"), slice_ct)
            except Exception as e:
                print(f"處理患者 {target_patient} 時發生預期外錯誤: {e}，跳過此患者。")
        else:
            print(f"警告：找不到 {target_patient} 的 cbct.nii.gz 或 ct.nii.gz，跳過。")
            
    print(f"\n所有病患資料處理完成！檔案已存入: {dst_root}")

if __name__ == "__main__":
    process_all_patients()