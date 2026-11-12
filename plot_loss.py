"""
畫 loss curve（在實驗室伺服器上跑即可，不需要 GPU）

讀取訓練時自動產生的 loss_log.csv（在 model_save_path 底下），畫出：
  1. 生成器總 loss   2. 判別器 loss   3. 生成器對抗 loss   4. cycle / identity loss

用法：
    # 單一實驗
    python plot_loss.py models/dataset2/loss_log.csv

    # 多個實驗疊在同一張圖比較（例如三個資料集）
    python plot_loss.py models/dataset1/loss_log.csv models/dataset2/loss_log.csv models/dataset3/loss_log.csv

    # 指定輸出檔名、平滑程度
    python plot_loss.py models/dataset2/loss_log.csv --out ds2_loss.png --smooth 5
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')            # 伺服器沒有螢幕，直接存成圖檔
import matplotlib.pyplot as plt

PANELS = [
    ("生成器總 loss (Generator total)", ["DomA_gen_loss", "DomB_gen_loss"]),
    ("判別器 loss (Discriminator)", ["DomA_disc_loss", "DomB_disc_loss"]),
    ("生成器對抗 loss (Adversarial)", ["DomA_adv_loss", "DomB_adv_loss"]),
    ("Cycle / Identity loss", ["cycle_loss", "DomA_id_loss", "DomB_id_loss"]),
]
STYLES = {"DomA": "-", "DomB": "--", "cycle": "-", }


def read_csv(path):
    with open(path) as f:
        header = f.readline().strip().split(',')
        rows = [line.strip().split(',') for line in f if line.strip()]
    data = np.array(rows, dtype=float)
    # 接續訓練時若有重複的 epoch（例如中斷前後各寫了一次），保留最後一筆
    epochs = data[:, 0]
    _, last_idx = np.unique(epochs[::-1], return_index=True)
    keep = np.sort(len(epochs) - 1 - last_idx)
    data = data[keep]
    return {h: data[:, i] for i, h in enumerate(header)}


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode='edge'), np.ones(k) / k, mode='valid')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+', help='一個或多個 loss_log.csv')
    ap.add_argument('--out', default='loss_curve.png', help='輸出圖檔名稱')
    ap.add_argument('--smooth', type=int, default=1, help='移動平均視窗（1 = 不平滑）')
    args = ap.parse_args()

    runs = []
    for p in args.csv:
        label = os.path.basename(os.path.dirname(os.path.abspath(p)))   # 用資料夾名稱當標籤
        runs.append((label, read_csv(p)))

    # 中文字型（找不到時退回預設字型，標題的中文可能顯示成方框，但不影響圖）
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft JhengHei',
                                       'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, (title, keys) in zip(axes.flat, PANELS):
        for r, (label, d) in enumerate(runs):
            for k in keys:
                if k not in d:
                    continue
                ls = '--' if k.startswith('DomB') else (':' if k.startswith('DomA_id') else '-')
                name = k if len(runs) == 1 else f"{label} · {k}"
                ax.plot(d['epoch'], smooth(d[k], args.smooth), ls,
                        color=colors[r % len(colors)] if len(runs) > 1 else None,
                        label=name, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(' vs '.join(label for label, _ in runs) +
                 (f'（移動平均 {args.smooth}）' if args.smooth > 1 else ''))
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"已儲存：{args.out}")
    for label, d in runs:
        print(f"  {label}：共 {len(d['epoch'])} 個 epoch（{int(d['epoch'][0])} ~ {int(d['epoch'][-1])}）")


if __name__ == '__main__':
    main()
