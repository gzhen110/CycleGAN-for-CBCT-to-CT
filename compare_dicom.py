"""
DICOM 版的資料集比較（計算方式與 compare_datasets.py 相同，結果可直接放在一起比較）

資料夾結構（自動尋找）：
    root/
        2BA001/
            <名稱含 cbct 的資料夾>/   *.dcm（或沒有副檔名的 DICOM 檔）
            <名稱含 ct、但不含 cbct 的資料夾>/   *.dcm
        2BA002/ ...
  若病人資料夾底下多包了一層，會自動往下找。名稱規則可用 --cbct_key / --ct_key 調整。

數值格式（--mode，預設 auto 自動判斷）：
    norm   ：DICOM 存的是歸一化後的值（-1~1，背景 -1、身體內 0~1），與 NIfTI 版完全相同的分析
    uint16 ：值為 0~65535 且沒有換算參數 → 視為 -1~1 存成 16 位元整數，換回 -1~1 後同 norm
    hu     ：真正的 HU，使用下面的規則
  auto：範圍在 -1~1 → norm；範圍在 0~65535 且 RescaleIntercept 為 0 → uint16；其他 → hu

HU 模式的規則（與 NIfTI 版對應）：
    身體 = CT 的 HU > --body_hu（預設 -500，空氣與背景約 -1000）
    身體內 HU clip 到 [-150, 150]（同 evaluate3D.py），身體外 = -1000
    「剛好 0」= 身體內 HU <= -150；「剛好 1」= 身體內 HU >= 150

用法（實驗室伺服器，CPU 即可；需要 pydicom：pip install pydicom）：
    python compare_dicom.py --data dataset1=/path/to/dataset1_dicom --out ~/data/results/compare_dicom
    # 只看測試病人：--split test（在專案資料夾內執行，讀 config.py）
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_datasets import analyze as analyze_norm   # 與 NIfTI 版相同的分析

try:
    import pydicom
except ImportError:
    sys.exit("需要 pydicom：pip install pydicom")

HU_MIN, HU_MAX, BG_HU = -150.0, 150.0, -1000.0


def find_series_dirs(patient_dir, cbct_key, ct_key):
    """在病人資料夾（含子資料夾）中找出 CBCT 與 CT 的序列資料夾"""
    cbct_dir, ct_dir = None, None
    for cur, dirs, files in os.walk(patient_dir):
        name = os.path.basename(cur).lower()
        if not files:
            continue
        if cbct_key in name and cbct_dir is None:
            cbct_dir = cur
        elif ct_key in name and cbct_key not in name and ct_dir is None:
            ct_dir = cur
    return cbct_dir, ct_dir


def load_series(folder):
    """讀取一個 DICOM 序列 → (Z, H, W) 的 HU 陣列，依切片位置排序"""
    slices = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if not os.path.isfile(p):
            continue
        try:
            ds = pydicom.dcmread(p)
            arr = ds.pixel_array
        except Exception:
            continue      # 不是 DICOM 影像檔（例如 DICOMDIR、txt），略過
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        inter = float(getattr(ds, 'RescaleIntercept', 0.0))
        if hasattr(ds, 'ImagePositionPatient'):
            z = float(ds.ImagePositionPatient[2])
        else:
            z = float(getattr(ds, 'InstanceNumber', len(slices)))
        slices.append((z, arr.astype(np.float32) * slope + inter, inter))
    if not slices:
        return None, None
    slices.sort(key=lambda t: t[0])
    shapes = {s[1].shape for s in slices}
    if len(shapes) > 1:
        raise ValueError(f"切片大小不一致：{shapes}")
    intercepts = sorted({s[2] for s in slices})
    return np.stack([s[1] for s in slices]), intercepts


def to_window(hu, body):
    out = np.full_like(hu, BG_HU)
    out[body] = np.clip(hu[body], HU_MIN, HU_MAX)
    return out


def analyze(cbct, ct, body_hu):
    ct_body = ct > body_hu
    cb_body = cbct > body_hu
    inter = np.logical_and(ct_body, cb_body).sum()
    dice = 2 * inter / max(ct_body.sum() + cb_body.sum(), 1)
    missing = np.logical_and(ct_body, ~cb_body).sum() / max(ct_body.sum(), 1)
    extra = np.logical_and(cb_body, ~ct_body).sum() / max(ct_body.sum(), 1)
    d = (to_window(cbct, cb_body) - to_window(ct, ct_body))[ct_body]
    return {
        'dice': float(dice), 'cbct_missing': float(missing), 'cbct_extra': float(extra),
        'base_MAE': float(np.mean(np.abs(d))), 'base_RMSE': float(np.sqrt(np.mean(d ** 2))),
        'ct_zero': float(np.mean(ct[ct_body] <= HU_MIN)), 'ct_one': float(np.mean(ct[ct_body] >= HU_MAX)),
        'cbct_zero': float(np.mean(cbct[ct_body] <= HU_MIN)), 'cbct_one': float(np.mean(cbct[ct_body] >= HU_MAX)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='名稱=路徑')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all')
    ap.add_argument('--cbct_key', default='cbct', help='CBCT 資料夾名稱要包含的字（不分大小寫）')
    ap.add_argument('--ct_key', default='ct', help='CT 資料夾名稱要包含的字（不分大小寫）')
    ap.add_argument('--body_hu', type=float, default=-500.0, help='HU 大於此值視為身體（hu 模式）')
    ap.add_argument('--mode', choices=['auto', 'norm', 'uint16', 'hu'], default='auto')
    args = ap.parse_args()

    wanted = None
    if args.split != 'all':
        sys.path.insert(0, os.getcwd())
        import config as cfg
        wanted = set({'train': cfg.TRAIN_PATIENTS, 'val': cfg.VAL_PATIENTS, 'test': cfg.TEST_PATIENTS}[args.split])

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    rows, skipped = [], []
    for item in args.data:
        name, root = item.split('=', 1)
        root = os.path.expanduser(root)
        pids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if wanted is not None:
            pids = [p for p in pids if p in wanted]
        print(f"{name}：{len(pids)} 位病人（{root}）", flush=True)
        shown = False
        for i, pid in enumerate(pids):
            cdir, tdir = find_series_dirs(os.path.join(root, pid), args.cbct_key.lower(), args.ct_key.lower())
            if not cdir or not tdir:
                skipped.append((name, pid, f'找不到序列資料夾（cbct={cdir}, ct={tdir}）'))
                continue
            try:
                cbct, ic = load_series(cdir)
                ct, it = load_series(tdir)
            except ValueError as e:
                skipped.append((name, pid, str(e)))
                continue
            if cbct is None or ct is None:
                skipped.append((name, pid, '資料夾內沒有可讀的 DICOM 影像'))
                continue
            # 判斷數值格式（每位病人各自判斷，CBCT 與 CT 分開）
            def detect(v, icpt):
                if args.mode != 'auto':
                    return args.mode
                if v.min() >= -1.01 and v.max() <= 1.01:
                    return 'norm'
                if v.min() >= 0 and v.max() <= 65535 and icpt == [0.0]:
                    return 'uint16'
                return 'hu'
            mc, mt = detect(cbct, ic), detect(ct, it)
            if not shown:
                # 第一位病人：印出讀到的格式，方便確認資料是否正確
                print(f"  [格式確認] {pid}")
                for lab, v, d, icpt, md in (('CBCT', cbct, cdir, ic, mc), ('CT', ct, tdir, it, mt)):
                    print(f"    {lab}: {d}\n      形狀 {v.shape}，原始值範圍 {v.min():.3f} ~ {v.max():.3f}，"
                          f"不同的值 {len(np.unique(v))} 種，RescaleIntercept {icpt} → 判斷為 {md}")
            if mc != mt:
                skipped.append((name, pid, f'CBCT 與 CT 的格式不同（{mc} / {mt}）'))
                continue
            if mc == 'uint16':
                cbct = cbct / 65535.0 * 2.0 - 1.0
                ct = ct / 65535.0 * 2.0 - 1.0
            if not shown:
                if mc in ('norm', 'uint16'):
                    for lab, v in (('CBCT', cbct), ('CT', ct)):
                        print(f"      {lab} 換算後：範圍 {v.min():.3f} ~ {v.max():.3f}，"
                              f"-1 的比例 {np.mean(v <= -0.999):.1%}，"
                              f"-1~0 之間 {np.mean((v > -0.999) & (v < -0.001)):.2%}")
                shown = True
            if cbct.shape != ct.shape:
                skipped.append((name, pid, f'CBCT {cbct.shape} 與 CT {ct.shape} 形狀不同'))
                continue
            r = analyze_norm(cbct, ct) if mc in ('norm', 'uint16') else analyze(cbct, ct, args.body_hu)
            r.update({'dataset': name, 'patient': pid, 'center': pid[2] if len(pid) > 2 else '?'})
            rows.append(r)
            print(f"\r  {i+1}/{len(pids)} {pid}", end='', file=sys.stderr, flush=True)
        print(file=sys.stderr)

    if not rows:
        print("沒有任何病人成功讀取。")
        for s in skipped[:10]:
            print("  略過", *s)
        return

    keys = ['dice', 'cbct_missing', 'cbct_extra', 'base_MAE', 'base_RMSE', 'ct_zero', 'ct_one', 'cbct_zero', 'cbct_one']
    csv_path = os.path.join(out, 'compare_dicom_per_patient.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'center', 'patient'] + keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['center'])].append(r)
        groups[(r['dataset'], '全部')].append(r)
    print("\n" + "=" * 118)
    print(f"{'資料集':10s} {'中心':4s} {'人數':>4s} | {'mask Dice':>10s} {'CBCT多遮':>9s} {'CBCT多出':>9s} | "
          f"{'baseline MAE':>13s} {'baseline RMSE':>14s} | {'CT=0':>6s} {'CBCT=0':>7s} {'CT=1':>6s} {'CBCT=1':>7s}")
    print("=" * 118)
    for item in args.data:
        name = item.split('=', 1)[0]
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
    if skipped:
        print(f"略過 {len(skipped)} 位病人，例如：")
        for s in skipped[:5]:
            print("  ", *s)
    print(f"\n每位病人的結果：{csv_path}")


if __name__ == '__main__':
    main()