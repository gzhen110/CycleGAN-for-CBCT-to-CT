"""
檢查「背景值露餡」假設：
  真實 CT 的背景剛好是 -1，但 generator 最後一層是 tanh，輸出只能接近 -1。
  若判別器只要看背景是不是剛好 -1 就能分辨真假，就會一直贏，generator 的對抗 loss 也會一路上升。

這支腳本做兩件事：
  1. 統計 generator（CBCT→CT）原始輸出在「背景 / padding / 身體內」的數值
  2. 直接問判別器（判斷 CT 真假的那個）：
       真實 CT、生成的 CT、生成的 CT 把背景強制改成 -1
     各自被認為是真的機率。若第 3 種的機率明顯高於第 2 種，就代表判別器靠背景在分辨真假。

用法（在國網中心用 dev 分區跑，幾分鐘即可）：
    python check_background.py --model_dir ./models/dataset2/ --data /work/jhz21803/dataset2_npy/
"""
import os
import sys
import argparse
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
from CycleGAN import CycleGan
import config as cfg

EXACT = -0.999   # 小於等於這個值，視為「剛好是 -1」


def pad_to(img, h, w, value=-1.0):
    ih, iw = img.shape
    top, left = (h - ih) // 2, (w - iw) // 2
    out = np.full((h, w), value, dtype=np.float32)
    out[top:top + ih, left:left + iw] = img
    valid = np.zeros((h, w), dtype=bool)          # True = 原始影像範圍（非 padding）
    valid[top:top + ih, left:left + iw] = True
    return out, valid


def describe(name, v):
    if v.size == 0:
        print(f"  {name:28s}: （無）")
        return
    p = np.percentile(v, [1, 50, 99])
    print(f"  {name:28s}: 平均 {v.mean():+.4f} | 最小 {v.min():+.4f} | "
          f"1%/50%/99% {p[0]:+.4f} / {p[1]:+.4f} / {p[2]:+.4f} | "
          f"剛好 -1 的比例 {np.mean(v <= EXACT):6.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--ckpt', default='epoch_200.ckpt', help='整個 GAN 的存檔（含判別器），預設 epoch_200.ckpt')
    ap.add_argument('--data', required=True, help='切片資料夾（內含 trainA/ trainB/）')
    ap.add_argument('--patients', nargs='*', default=None, help='預設：測試名單前 4 位')
    ap.add_argument('--every', type=int, default=5, help='每幾張切片取 1 張')
    ap.add_argument('--height', type=int, default=304)
    ap.add_argument('--width', type=int, default=304)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    patients = args.patients or list(cfg.TEST_PATIENTS)[:4]

    # ---------- 建立與訓練時相同的結構，載入整個 GAN 的存檔 ----------
    c_gen = Generator(args.height, args.width, args.alpha)
    s_gen = Generator(args.height, args.width, args.alpha)
    c_disc = Discriminator(args.height, args.width, args.alpha)
    s_disc = Discriminator(args.height, args.width, args.alpha)
    gan = CycleGan(c_gen, s_gen, c_disc, s_disc, pool_size=0)
    dummy = tf.zeros((1, args.height, args.width, 1))
    for m in (c_gen, s_gen, c_disc, s_disc):
        m(dummy, training=False)
    ckpt = os.path.join(args.model_dir, args.ckpt)
    if not os.path.exists(ckpt + '.index'):
        sys.exit(f"找不到存檔：{ckpt}.index")
    gan.load_weights(ckpt).expect_partial()
    print(f"已載入：{ckpt}\n使用病人：{patients}（每 {args.every} 張取 1 張）\n")

    dirA, dirB = os.path.join(args.data, 'trainA'), os.path.join(args.data, 'trainB')
    names = [f for f in sorted(os.listdir(dirA))
             if f.endswith('.npy') and f.split('_')[0] in patients][::args.every]
    if not names:
        sys.exit("找不到這些病人的切片")

    vals = defaultdict(list)
    probs = defaultdict(list)
    for s in range(0, len(names), args.batch):
        chunk = names[s:s + args.batch]
        cb, ct, valid = [], [], []
        for n in chunk:
            a = np.load(os.path.join(dirA, n)).astype(np.float32)
            b = np.load(os.path.join(dirB, n)).astype(np.float32)
            a = a[..., 0] if a.ndim == 3 else a
            b = b[..., 0] if b.ndim == 3 else b
            pa, va = pad_to(a, args.height, args.width)
            pb, _ = pad_to(b, args.height, args.width)
            cb.append(pa); ct.append(pb); valid.append(va)
        cb = np.stack(cb)[..., None]
        ct = np.stack(ct)[..., None]
        valid = np.stack(valid)[..., None]

        fake = c_gen(cb, training=False).numpy()

        real_bg = ct <= EXACT                       # 真實 CT 的背景（含 padding）
        pad = ~valid
        mask_bg = real_bg & valid                   # 原始影像內被 mask 的背景
        body = ~real_bg

        vals['fake_pad'].append(fake[pad])
        vals['fake_maskbg'].append(fake[mask_bg])
        vals['fake_body'].append(fake[body])
        vals['real_bg'].append(ct[real_bg])
        vals['cbct_bg'].append(cb[cb <= EXACT])

        # 生成的 CT，背景強制改成 -1
        fake_fixed = np.where(real_bg, -1.0, fake).astype(np.float32)

        for key, img in (('real', ct), ('fake', fake), ('fake_fixed', fake_fixed)):
            logits = c_disc(img, training=False).numpy()
            probs[key].append(tf.sigmoid(logits).numpy().reshape(len(chunk), -1).mean(axis=1))

    cat = {k: np.concatenate(v) for k, v in vals.items()}
    pr = {k: np.concatenate(v) for k, v in probs.items()}

    print("=" * 100)
    print("1. 背景數值（歸一化空間，-1 = 背景）")
    print("=" * 100)
    describe('真實 CT 的背景', cat['real_bg'])
    describe('輸入 CBCT 的背景', cat['cbct_bg'])
    describe('生成 CT：padding 區域', cat['fake_pad'])
    describe('生成 CT：mask 背景區域', cat['fake_maskbg'])
    describe('生成 CT：身體內（參考）', cat['fake_body'])

    print()
    print("=" * 100)
    print(f"2. 判別器（CT）認為「是真的」的平均機率（{len(pr['real'])} 張切片）")
    print("=" * 100)
    print(f"  真實 CT                     : {pr['real'].mean():.4f}")
    print(f"  生成 CT                     : {pr['fake'].mean():.4f}")
    print(f"  生成 CT，背景強制改成 -1     : {pr['fake_fixed'].mean():.4f}")

    print()
    print("判讀：")
    bg_exact = np.mean(np.concatenate([cat['fake_pad'], cat['fake_maskbg']]) <= EXACT)
    if bg_exact < 0.5:
        print(f"  - 生成 CT 的背景只有 {bg_exact:.1%} 剛好是 -1，真實 CT 則是 100%：背景數值確實不同。")
    else:
        print(f"  - 生成 CT 的背景有 {bg_exact:.1%} 剛好是 -1，背景數值差異不大。")
    gain = pr['fake_fixed'].mean() - pr['fake'].mean()
    print(f"  - 把背景改成 -1 後，判別器認為是真的機率變化 {gain:+.4f}。")
    if gain > 0.1:
        print("    → 變化明顯：判別器很可能靠背景在分辨真假，假設成立。")
    elif gain > 0.02:
        print("    → 有一些影響：背景是線索之一，但不是全部。")
    else:
        print("    → 幾乎沒變：判別器主要不是靠背景分辨，原因在身體內的影像本身。")


if __name__ == '__main__':
    main()
