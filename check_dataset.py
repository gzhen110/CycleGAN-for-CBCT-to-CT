"""
換資料集之前先跑這支，檢查新資料集跟模型的影像處理假設有沒有衝突。
資料夾結構：
    dataset_root/
        2BA001/
            cbct.nii.gz
            ct.nii.gz
        2BA002/
        ...
用法：
    python check_dataset.py /path/to/dataset2_new [--height 256 --width 256 --n 0]
    --n 0 代表檢查全部病人；給數字則隨機抽樣幾位
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib

HU_MIN, HU_MAX = -1000.0, 3000.0   # 與 DatasetLoader.py 相同


def summarize(vol):
    v = vol.astype(np.float32)
    # 每張切片（沿 z 軸，與 preprocess.py 一致）四個角落的值 → 估計背景
    corners = np.concatenate([v[0, 0, :], v[0, -1, :], v[-1, 0, :], v[-1, -1, :]])
    return {
        'min': float(v.min()), 'max': float(v.max()),
        'p1': float(np.percentile(v, 1)), 'p99': float(np.percentile(v, 99)),
        'bg': float(np.median(corners)),
        'frac_over': float(np.mean(v > HU_MAX)),
        'frac_under': float(np.mean(v < HU_MIN)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--n', type=int, default=0, help='抽樣病人數，0 = 全部')
    args = ap.parse_args()

    # 讓 config.py 的病人切分名單可以拿來比對（從目前目錄或本檔所在目錄 import）
    try:
        sys.path.insert(0, os.getcwd())
        import config
        splits = {'train': set(config.TRAIN_PATIENTS),
                  'val': set(config.VAL_PATIENTS),
                  'test': set(config.TEST_PATIENTS)}
    except Exception:
        splits = None

    patients = sorted(d for d in os.listdir(args.root)
                      if os.path.isdir(os.path.join(args.root, d)))
    if not patients:
        sys.exit(f"[錯誤] {args.root} 底下沒有病人資料夾")
    print(f"共 {len(patients)} 位病人資料夾")

    warn = []

    # 1. 檔案是否齊全
    missing = [p for p in patients
               if not (os.path.exists(os.path.join(args.root, p, 'cbct.nii.gz'))
                       and os.path.exists(os.path.join(args.root, p, 'ct.nii.gz')))]
    if missing:
        warn.append(f"{len(missing)} 位病人缺 cbct.nii.gz 或 ct.nii.gz：{missing[:5]}")
    patients = [p for p in patients if p not in missing]

    # 2. 跟 config.py 的切分名單比對
    if splits:
        have = set(patients)
        for k, s in splits.items():
            lack = sorted(s - have)
            print(f"  {k} 名單 {len(s)} 位，資料集中有 {len(s & have)} 位"
                  + (f"，缺 {lack[:5]}{'…' if len(lack) > 5 else ''}" if lack else ""))
            if lack:
                warn.append(f"config.py 的 {k} 名單有 {len(lack)} 位不在這個資料集裡")
    bad_id = [p for p in patients if len(p) != 6]
    if bad_id:
        warn.append(f"資料夾名稱不是 6 碼病人 ID：{bad_id[:3]}")

    if args.n and args.n < len(patients):
        rng = np.random.default_rng(0)
        patients = sorted(rng.choice(patients, size=args.n, replace=False))
        print(f"隨機抽樣 {len(patients)} 位檢查")

    # 3. 逐位病人檢查
    stats = {'cbct': [], 'ct': []}
    shapes, inplane = set(), set()
    for i, p in enumerate(patients):
        print(f"\r檢查中 {i+1}/{len(patients)} {p}", end='', file=sys.stderr, flush=True)
        imgs = {}
        for m in ('cbct', 'ct'):
            imgs[m] = nib.load(os.path.join(args.root, p, f'{m}.nii.gz'))
        a, b = imgs['cbct'], imgs['ct']

        if a.shape != b.shape:
            warn.append(f"{p} CBCT {a.shape} 與 CT {b.shape} 形狀不同 → 切片無法一對一配對")
        if not np.allclose(a.affine, b.affine, atol=1e-3):
            warn.append(f"{p} CBCT 與 CT 的 affine（spacing/方向/原點）不同 → 可能沒有配準好")
        if len(a.shape) != 3:
            warn.append(f"{p} CBCT 不是 3D 影像：{a.shape}")
            continue

        shapes.add(a.shape)
        inplane.add(a.shape[:2])
        for m in ('cbct', 'ct'):
            s = summarize(np.asanyarray(imgs[m].dataobj))
            s['id'] = p
            stats[m].append(s)
    print(file=sys.stderr)

    # 4. 尺寸
    print(f"\n== 尺寸（沿 z 軸切片後的平面大小）")
    print(f"  不同的切片平面大小：{sorted(inplane)}")
    print(f"  不同的 3D 形狀數：{len(shapes)}")
    over = [s for s in inplane if s[0] > args.height or s[1] > args.width]
    if over:
        warn.append(f"有切片平面 {over} 超過 {args.height}x{args.width} → 目前程式會 cv2.resize 變形")
    if len(inplane) > 1:
        warn.append(f"切片平面大小不一致（{len(inplane)} 種）→ 需要 padding，不能刪")
    else:
        h, w = next(iter(inplane))
        if h % 16 or w % 16:
            warn.append(f"切片大小 {h}x{w} 不是 16 的倍數 → 不能直接餵模型")
        elif (h, w) == (args.height, args.width):
            print(f"  ✔ 全部都是 {h}x{w}，padding 不會被執行，可以刪")

    # 5. 數值範圍
    for m in ('cbct', 'ct'):
        st = stats[m]
        if not st:
            continue
        g = lambda k: np.array([s[k] for s in st])
        print(f"\n== {m.upper()}（{len(st)} 位病人）")
        print(f"  最小值：中位數 {np.median(g('min')):.1f}（範圍 {g('min').min():.1f} ~ {g('min').max():.1f}）")
        print(f"  最大值：中位數 {np.median(g('max')):.1f}（範圍 {g('max').min():.1f} ~ {g('max').max():.1f}）")
        print(f"  1%~99% 分位數中位數：{np.median(g('p1')):.1f} ~ {np.median(g('p99')):.1f}")
        print(f"  背景值（四角）中位數：{np.median(g('bg')):.1f}")
        print(f"  超過 {HU_MAX:.0f} 的體素比例（平均）：{g('frac_over').mean():.4%}；"
              f"低於 {HU_MIN:.0f}：{g('frac_under').mean():.4%}")

        lo, hi = np.median(g('min')), np.median(g('max'))
        if -1.5 <= lo and hi <= 1.5:
            warn.append(f"{m} 值域約 {lo:.2f}~{hi:.2f}，已經歸一化過，不是 HU → 必須拿掉 DatasetLoader 的 HU clip")
        elif lo > -500:
            warn.append(f"{m} 最小值中位數 {lo:.1f}，空氣不在 -1000 附近 → 可能不是 HU 尺度（或有 offset）")
        bg = np.median(g('bg'))
        if abs(bg - HU_MIN) > 50 and not (-1.5 <= lo and hi <= 1.5):
            warn.append(f"{m} 背景約 {bg:.1f}，但 padding 補的是 -1.0（= -1000 HU）→ 補邊處會有假邊界")
        if g('frac_under').mean() > 0.01:
            warn.append(f"{m} 平均有 {g('frac_under').mean():.1%} 體素低於 -1000 → 會被 clip 成 -1000")
        # 個別離群病人（例如 RescaleIntercept 問題造成的 +1000 偏移）
        odd = [s['id'] for s in st if s['min'] > -500]
        if odd and len(odd) < len(st):
            warn.append(f"{m} 有 {len(odd)} 位病人最小值 > -500 HU，跟其他人不同：{odd[:5]}")

    print("\n" + "=" * 60)
    print("⚠️  另外注意：目前的 DatasetLoader 讀的是 trainA/ trainB/ 底下的 2D .npy 切片，")
    print("    這個資料集是 3D NIfTI，訓練前還需要切片，或改寫讀取方式。")
    if warn:
        print("\n⚠️  可能衝突：")
        for w in warn:
            print("  - " + w)
    else:
        print("\n✅ 沒有發現其他明顯衝突")


if __name__ == '__main__':
    main()