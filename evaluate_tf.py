"""
TF 版 3D 評估腳本（計算方式參考組員的 evaluate3D.py）

對測試病人的每張 CBCT 切片產生 sCT，依病人組成 3D 體積，計算 MAE / RMSE / SSIM / PSNR。

反歸一化（與 evaluate3D.py 相同）：
    值 <= -0.5  → 背景，設為 -1000 HU
    值 >  -0.5  → 組織，x * 300 - 150，再 clip 到 [-150, 150] HU
動態範圍 DATA_RANGE = 300（軟組織範圍）

輸出兩組指標：
    full ：整個影像（已去掉 padding），與 evaluate3D.py 的算法相同，方便與組員比較
    body ：只算身體內（真實 CT > -0.5 的區域），排除背景，建議論文使用

用法：
    python evaluate_tf.py \
        --model_dir ./models/dataset2/ \
        --data /work/jhz21803/dataset2_npy/ \
        --out ./results/dataset2/
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity

from generator import Generator
import config as cfg   # 讀取 TEST_PATIENTS

# ==================== 數值與動態範圍設定（與 evaluate3D.py 相同） ====================
HU_MIN = -150.0
HU_MAX = 150.0
BACKGROUND_HU = -1000.0
DATA_RANGE = 300.0
BODY_THRESHOLD = -0.5     # 歸一化值大於此值視為身體內
# ==============================================================================


def unnormalize_to_hu(arr):
    """與 evaluate3D.py 完全相同的雙段反歸一化"""
    arr_hu = np.full_like(arr, fill_value=BACKGROUND_HU, dtype=np.float32)
    body = arr > BODY_THRESHOLD
    if np.any(body):
        arr_hu[body] = np.clip(arr[body] * 300.0 - 150.0, HU_MIN, HU_MAX)
    return arr_hu


def metrics_full(sct, ct):
    """整個體積（與 evaluate3D.py 的 compute_similarity_metrics_3d 相同）"""
    diff = sct - ct
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float('inf') if mse == 0 else float(20 * np.log10(DATA_RANGE / rmse))
    ssim = float(structural_similarity(sct, ct, data_range=DATA_RANGE, channel_axis=None))
    return mae, rmse, ssim, psnr


def metrics_body(sct, ct, mask):
    """只在身體內計算：MAE/RMSE 取遮罩內體素；SSIM 取 SSIM map 在遮罩內的平均"""
    diff = (sct - ct)[mask]
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float('inf') if mse == 0 else float(20 * np.log10(DATA_RANGE / rmse))
    _, ssim_map = structural_similarity(sct, ct, data_range=DATA_RANGE, channel_axis=None, full=True)
    ssim = float(np.mean(ssim_map[mask]))
    return mae, rmse, ssim, psnr


def pad_to(img, h, w, value=-1.0):
    """與 DatasetLoader 相同的置中 padding，回傳 padding 後影像與裁回用的位置"""
    ih, iw = img.shape
    if ih > h or iw > w:
        raise ValueError(f"切片大小 {ih}x{iw} 超過 {h}x{w}")
    top, left = (h - ih) // 2, (w - iw) // 2
    out = np.full((h, w), value, dtype=np.float32)
    out[top:top + ih, left:left + iw] = img
    return out, (top, left, ih, iw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='訓練結果資料夾（內含 model_weights）')
    ap.add_argument('--weights', default='model_weights', help='權重檔名（預設最終權重 model_weights）')
    ap.add_argument('--data', required=True, help='切片資料夾（內含 trainA/ trainB/）')
    ap.add_argument('--out', required=True, help='輸出資料夾')
    ap.add_argument('--height', type=int, default=304)
    ap.add_argument('--width', type=int, default=304)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patients', nargs='*', default=None, help='要評估的病人（預設 config.TEST_PATIENTS）')
    ap.add_argument('--save_npy', action='store_true', help='另存每位病人的 sCT / realCT 3D 體積（HU）')
    args = ap.parse_args()

    patients = args.patients or list(cfg.TEST_PATIENTS)
    os.makedirs(args.out, exist_ok=True)

    # ---------- 載入模型（CBCT → CT 的生成器 c_generator）----------
    weights = os.path.join(args.model_dir, args.weights)
    if not os.path.exists(weights + '.index'):
        sys.exit(f"找不到權重：{weights}.index")
    model = Generator(args.height, args.width, args.alpha)
    model.load_weights(weights).expect_partial()
    print(f"已載入權重：{weights}")

    # ---------- 收集各病人的切片（依檔名排序 = 依 z 軸順序）----------
    dirA, dirB = os.path.join(args.data, 'trainA'), os.path.join(args.data, 'trainB')
    files = defaultdict(list)
    for f in sorted(os.listdir(dirA)):
        if f.endswith('.npy') and f.split('_')[0] in patients:
            files[f.split('_')[0]].append(f)
    missing = [p for p in patients if p not in files]
    if missing:
        print(f"⚠️ 以下病人在資料夾中找不到切片，略過：{missing}")

    rows = []
    out_of_range = []   # 檢查反歸一化假設用
    for pi, pid in enumerate(p for p in patients if p in files):
        names = files[pid]
        sct_slices, ct_slices = [], []
        for s in range(0, len(names), args.batch):
            chunk = names[s:s + args.batch]
            cbct_batch, ct_list, locs = [], [], []
            for n in chunk:
                a = np.load(os.path.join(dirA, n)).astype(np.float32)
                b = np.load(os.path.join(dirB, n)).astype(np.float32)
                a = a[..., 0] if a.ndim == 3 else a
                b = b[..., 0] if b.ndim == 3 else b
                pa, loc = pad_to(a, args.height, args.width)
                cbct_batch.append(pa)
                ct_list.append(b)
                locs.append(loc)
            pred = model(np.stack(cbct_batch)[..., None], training=False).numpy()[..., 0]
            for p, b, (top, left, ih, iw) in zip(pred, ct_list, locs):
                sct_slices.append(p[top:top + ih, left:left + iw])   # 裁掉 padding，回到原始大小
                ct_slices.append(b)

        if len({x.shape for x in ct_slices}) > 1:
            sys.exit(f"{pid} 的切片大小不一致，無法組成 3D 體積：{sorted({x.shape for x in ct_slices})}")
        sct_norm = np.stack(sct_slices)    # (D, H, W)，歸一化空間
        ct_norm = np.stack(ct_slices)

        # 檢查：真實 CT 在身體內是否落在 [0, 1]（反歸一化公式的前提）
        body = ct_norm > BODY_THRESHOLD
        out_of_range.append(float(np.mean((ct_norm > -0.99) & (ct_norm < -0.01))))

        sct_hu = unnormalize_to_hu(sct_norm)
        ct_hu = unnormalize_to_hu(ct_norm)

        f_mae, f_rmse, f_ssim, f_psnr = metrics_full(sct_hu, ct_hu)
        b_mae, b_rmse, b_ssim, b_psnr = metrics_body(sct_hu, ct_hu, body)
        rows.append({'patient': pid, 'slices': len(names),
                     'MAE_full': f_mae, 'RMSE_full': f_rmse, 'SSIM_full': f_ssim, 'PSNR_full': f_psnr,
                     'MAE_body': b_mae, 'RMSE_body': b_rmse, 'SSIM_body': b_ssim, 'PSNR_body': b_psnr})
        print(f"[{pi+1}/{len(files)}] {pid}（{len(names)} 張）| "
              f"full: MAE {f_mae:.2f} RMSE {f_rmse:.2f} SSIM {f_ssim:.4f} PSNR {f_psnr:.2f} | "
              f"body: MAE {b_mae:.2f} RMSE {b_rmse:.2f} SSIM {b_ssim:.4f} PSNR {b_psnr:.2f}", flush=True)

        if args.save_npy:
            d = os.path.join(args.out, 'npy_exports')
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f'{pid}_sCT.npy'), sct_hu)
            np.save(os.path.join(d, f'{pid}_realCT.npy'), ct_hu)

    # ---------- 輸出 CSV（Excel 可直接開啟）----------
    keys = ['MAE_full', 'RMSE_full', 'SSIM_full', 'PSNR_full', 'MAE_body', 'RMSE_body', 'SSIM_body', 'PSNR_body']
    mean = {'patient': '平均', 'slices': sum(r['slices'] for r in rows)}
    std = {'patient': '標準差', 'slices': ''}
    for k in keys:
        v = np.array([r[k] for r in rows])
        mean[k], std[k] = float(v.mean()), float(v.std())
    csv_path = os.path.join(args.out, 'evaluation_3d.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['patient', 'slices'] + keys)
        w.writeheader()
        for r in rows + [mean, std]:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print("=" * 70)
    print(f"共 {len(rows)} 位病人。平均 ± 標準差：")
    for tag in ('full', 'body'):
        print(f"  {tag:4s}: " + " | ".join(
            f"{m} {mean[f'{m}_{tag}']:.4f} ± {std[f'{m}_{tag}']:.4f}" for m in ('MAE', 'RMSE', 'SSIM', 'PSNR')))
    print(f"結果已存至 {csv_path}")
    print(f"設定：背景 {BACKGROUND_HU} HU，組織 [{HU_MIN}, {HU_MAX}] HU，Dynamic Range {DATA_RANGE}")

    # 反歸一化假設檢查
    frac = float(np.mean(out_of_range))
    if frac > 0.001:
        print(f"⚠️ 真實 CT 有 {frac:.2%} 的體素介於 -1 與 0 之間，"
              f"資料可能不是「背景 -1、組織 0~1」的格式，反歸一化公式可能不適用，請先確認前處理方式！")
    else:
        print(f"✔ 確認：真實 CT 身體內的值都在 0~1（-1~0 之間的比例 {frac:.4%}），反歸一化公式適用")


if __name__ == '__main__':
    main()
