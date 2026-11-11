"""
把 3D NIfTI 資料集切成訓練用的 2D .npy 切片（數值完全不變，不做任何歸一化）。

輸入結構：
    src/
        2BA001/cbct.nii.gz, ct.nii.gz
        2BA002/...
輸出結構（DatasetLoader / Datasetlabels 讀的格式）：
    dst/
        trainA/2BA001_000.npy ...   (CBCT)
        trainB/2BA001_000.npy ...   (CT)

用法：
    python slice_dataset.py /path/to/dataset2_new /path/to/dataset2_npy
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='3D 資料集根目錄（底下是病人資料夾）')
    ap.add_argument('dst', help='輸出資料夾（會建立 trainA/ trainB/）')
    ap.add_argument('--bg', type=float, default=-1.0, help='背景值，整張都是背景的切片會跳過')
    args = ap.parse_args()

    dirA = os.path.join(args.dst, 'trainA')
    dirB = os.path.join(args.dst, 'trainB')
    if os.path.isdir(dirA) and os.listdir(dirA):
        sys.exit(f"[停止] {dirA} 已經有檔案，請換一個輸出資料夾或先清空，避免新舊切片混在一起")
    os.makedirs(dirA, exist_ok=True)
    os.makedirs(dirB, exist_ok=True)

    patients = sorted(d for d in os.listdir(args.src)
                      if os.path.isdir(os.path.join(args.src, d)))
    total, skipped_pat = 0, []

    for i, pid in enumerate(patients):
        fa = os.path.join(args.src, pid, 'cbct.nii.gz')
        fb = os.path.join(args.src, pid, 'ct.nii.gz')
        if not (os.path.exists(fa) and os.path.exists(fb)):
            skipped_pat.append((pid, '缺檔'))
            continue

        cbct = np.asanyarray(nib.load(fa).dataobj).astype(np.float32)
        ct = np.asanyarray(nib.load(fb).dataobj).astype(np.float32)
        if cbct.shape != ct.shape:
            skipped_pat.append((pid, f'形狀不同 {cbct.shape} vs {ct.shape}'))
            continue

        n = 0
        for z in range(cbct.shape[2]):          # 沿 z 軸切，與原本 preprocess.py 相同
            a, b = cbct[:, :, z], ct[:, :, z]
            # CBCT 和 CT 都整張是背景才跳過（兩邊用同一個條件，確保 A/B 檔案一一對應）
            if np.all(a <= args.bg + 1e-6) and np.all(b <= args.bg + 1e-6):
                continue
            name = f"{pid}_{z:03d}.npy"
            np.save(os.path.join(dirA, name), a)
            np.save(os.path.join(dirB, name), b)
            n += 1
        total += n
        print(f"[{i+1}/{len(patients)}] {pid}: {n} 張切片，平面大小 {cbct.shape[:2]}", flush=True)

    print("=" * 60)
    print(f"完成：{len(patients) - len(skipped_pat)} 位病人，共 {total} 對切片 → {args.dst}")
    if skipped_pat:
        print("⚠️ 跳過的病人：")
        for pid, why in skipped_pat:
            print(f"   {pid}: {why}")


if __name__ == '__main__':
    main()
