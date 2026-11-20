"""
比較不同前處理資料集的 CBCT 與 CT（不需要模型、不需要 GPU，在實驗室伺服器用 CPU 跑即可）

目的：找出 dataset 3 為什麼不如預期。前處理的目標是讓 CBCT 更接近 CT，
所以直接量「模型還沒介入前」CBCT 與 CT 的差距，以及兩者的遮罩是否一致。

每位病人計算（身體 = 真實 CT 不是 -1 的區域）：
  - mask Dice      ：CBCT 與 CT 的身體遮罩重疊程度（1 = 完全一致）
  - CBCT 多遮掉的比例：CT 有組織、但 CBCT 已經是背景的體素比例（模型被迫「憑空生成」的區域）
  - baseline MAE / RMSE（HU）：CBCT 直接當成 CT 時的誤差（反歸一化方式與 evaluate3D.py 相同）
  - CBCT / CT 中「剛好 0」「剛好 1」的比例（-150 / 150 HU 截斷的程度）

輸出：
  - 每位病人的 CSV（out/compare_per_patient.csv）
  - 各資料集 × 各中心的摘要（印在畫面上）
  - 箱形圖（out/compare_boxplot.png）

用法：
    python compare_datasets.py \
        --data dataset2=/home/qjanet/dataset/CBCT2CT/dataset2_new \
               dataset3=/home/qjanet/data/harmonized_quartile_newmask \
        --out ~/data/results/compare
    # 只看測試病人：加上 --split test（讀 config.py 的 TEST_PATIENTS）
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
import nibabel as nib

EXACT = -0.999
TOL = 0.001
BODY_THRESHOLD = -0.5


def to_hu(arr):
    """與 evaluate3D.py 相同：<= -0.5 → -1000 HU；其餘 x*300-150，clip 到 [-150, 150]"""
    hu = np.full_like(arr, -1000.0, dtype=np.float32)
    b = arr > BODY_THRESHOLD
    hu[b] = np.clip(arr[b] * 300.0 - 150.0, -150.0, 150.0)
    return hu


def analyze(cbct, ct):
    ct_body = ct > EXACT
    cb_body = cbct > EXACT
    inter = np.logical_and(ct_body, cb_body).sum()
    dice = 2 * inter / max(ct_body.sum() + cb_body.sum(), 1)
    missing = np.logical_and(ct_body, ~cb_body).sum() / max(ct_body.sum(), 1)     # CT 有、CBCT 沒有
    extra = np.logical_and(cb_body, ~ct_body).sum() / max(ct_body.sum(), 1)       # CBCT 有、CT 沒有
    d = (to_hu(cbct) - to_hu(ct))[ct_body]
    return {
        'dice': float(dice),
        'cbct_missing': float(missing),
        'cbct_extra': float(extra),
        'base_MAE': float(np.mean(np.abs(d))),
        'base_RMSE': float(np.sqrt(np.mean(d ** 2))),
        'ct_zero': float(np.mean(np.abs(ct[ct_body]) <= TOL)),
        'ct_one': float(np.mean(ct[ct_body] >= 1 - TOL)),
        'cbct_zero': float(np.mean(np.abs(cbct[ct_body]) <= TOL)),
        'cbct_one': float(np.mean(cbct[ct_body] >= 1 - TOL)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='名稱=路徑，例如 dataset2=/path/to/dataset2_new')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all')
    args = ap.parse_args()

    datasets = []
    for item in args.data:
        if '=' not in item:
            sys.exit(f"--data 格式應為 名稱=路徑：{item}")
        name, path = item.split('=', 1)
        datasets.append((name, os.path.expanduser(path)))

    wanted = None
    if args.split != 'all':
        sys.path.insert(0, os.getcwd())
        import config as cfg
        wanted = set({'train': cfg.TRAIN_PATIENTS, 'val': cfg.VAL_PATIENTS, 'test': cfg.TEST_PATIENTS}[args.split])

    os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    out = os.path.expanduser(args.out)
    rows = []
    for name, root in datasets:
        pids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if wanted is not None:
            pids = [p for p in pids if p in wanted]
        print(f"{name}：{len(pids)} 位病人（{root}）", flush=True)
        for i, pid in enumerate(pids):
            fa, fb = os.path.join(root, pid, 'cbct.nii.gz'), os.path.join(root, pid, 'ct.nii.gz')
            if not (os.path.exists(fa) and os.path.exists(fb)):
                print(f"  略過 {pid}：缺檔")
                continue
            cbct = np.asanyarray(nib.load(fa).dataobj).astype(np.float32)
            ct = np.asanyarray(nib.load(fb).dataobj).astype(np.float32)
            if cbct.shape != ct.shape:
                print(f"  略過 {pid}：形狀不同 {cbct.shape} vs {ct.shape}")
                continue
            r = analyze(cbct, ct)
            r.update({'dataset': name, 'patient': pid, 'center': pid[2] if len(pid) > 2 else '?'})
            rows.append(r)
            print(f"\r  {i+1}/{len(pids)} {pid}", end='', file=sys.stderr, flush=True)
        print(file=sys.stderr)

    keys = ['dice', 'cbct_missing', 'cbct_extra', 'base_MAE', 'base_RMSE', 'ct_zero', 'ct_one', 'cbct_zero', 'cbct_one']
    csv_path = os.path.join(out, 'compare_per_patient.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'center', 'patient'] + keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------- 摘要 ----------
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['center'])].append(r)
        groups[(r['dataset'], '全部')].append(r)
    print("\n" + "=" * 118)
    print(f"{'資料集':10s} {'中心':4s} {'人數':>4s} | {'mask Dice':>10s} {'CBCT多遮':>9s} {'CBCT多出':>9s} | "
          f"{'baseline MAE':>13s} {'baseline RMSE':>14s} | {'CT=0':>6s} {'CBCT=0':>7s} {'CT=1':>6s} {'CBCT=1':>7s}")
    print("=" * 118)
    for name, _ in datasets:
        for c in ['A', 'B', 'C', '全部']:
            g = groups.get((name, c))
            if not g:
                continue
            m = {k: np.array([r[k] for r in g]) for k in keys}
            print(f"{name:10s} {c:4s} {len(g):4d} | {m['dice'].mean():10.4f} {m['cbct_missing'].mean():9.2%} "
                  f"{m['cbct_extra'].mean():9.2%} | {m['base_MAE'].mean():6.2f} ± {m['base_MAE'].std():5.2f} "
                  f"{m['base_RMSE'].mean():7.2f} ± {m['base_RMSE'].std():5.2f} | {m['ct_zero'].mean():6.2%} "
                  f"{m['cbct_zero'].mean():7.2%} {m['ct_one'].mean():6.2%} {m['cbct_one'].mean():7.2%}")
        print("-" * 118)

    # 各資料集中，mask 最不一致、baseline 誤差最大的病人
    for name, _ in datasets:
        g = [r for r in rows if r['dataset'] == name]
        if not g:
            continue
        worst_dice = sorted(g, key=lambda r: r['dice'])[:5]
        worst_mae = sorted(g, key=lambda r: -r['base_MAE'])[:5]
        print(f"{name}：mask Dice 最低 → " + "、".join(f"{r['patient']}({r['dice']:.3f})" for r in worst_dice))
        print(f"{name}：baseline MAE 最高 → " + "、".join(f"{r['patient']}({r['base_MAE']:.1f})" for r in worst_mae))
    print(f"\n每位病人的結果：{csv_path}")

    # ---------- 箱形圖 ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("（未安裝 matplotlib，略過箱形圖）")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    centers = ['A', 'B', 'C']
    for ax, key, title in ((axes[0], 'base_MAE', 'Baseline MAE: CBCT vs CT (HU, body)'),
                           (axes[1], 'dice', 'Body mask Dice: CBCT vs CT')):
        data, labels, pos = [], [], []
        p = 0
        for name, _ in datasets:
            for c in centers:
                v = [r[key] for r in rows if r['dataset'] == name and r['center'] == c]
                if v:
                    data.append(v); labels.append(f"{name}\n{c}"); pos.append(p)
                p += 1
            p += 1
        ax.boxplot(data, positions=pos, widths=0.6)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    png = os.path.join(out, 'compare_boxplot.png')
    fig.savefig(png, dpi=150)
    print(f"箱形圖：{png}")


if __name__ == '__main__':
    main()
