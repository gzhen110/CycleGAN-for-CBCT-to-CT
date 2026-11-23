"""
量測 self-attention 分支實際上有多大作用

SelfAttention.py 中：out = LayerNorm(0.002 * attention分支 + x)
這支腳本對 4 個 attention 層（生成器 2 個、判別器 2 個）量測：
  1. 分支大小比例：‖0.002 × 分支‖ / ‖x‖（L2 範數），越接近 0 代表分支越微弱
  2. 對該層輸出的影響：‖LN(分支 + x) − LN(x)‖ / ‖LN(x)‖
  3. softmax 的集中程度：每列 attention 權重的最大值平均（若平均分配，約為 1 / 列數）
並做端對端比較：
  4. 把所有 attention 分支關掉（只保留 LayerNorm(x)），生成器輸出的 sCT 改變多少（HU），
     以及兩者相對真實 CT 的 body MAE

用法：
    python measure_attention.py --model_dir ./models/dataset2/ --data /work/jhz21803/dataset2_npy/
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
from SelfAttention import self_attention, self_attentionDecoder
from check_background import pad_to
import config as cfg

GAMMA = 0.002
HU_PER_UNIT = 300.0
LAYERS = {
    'gen': ['gen_bottleneck_attention', 'gen_decoder_attention'],
    'disc': ['dis_attention_1', 'dis_attention_2'],
}


def branch_parts(layer, x):
    """依照 SelfAttention.py 的 call() 重算一次，回傳 (0.002×分支, softmax 權重)；推論時 dropout 不作用"""
    f, g, h = layer.f(x), layer.g(x), layer.h(x)
    _, height, width, ch = x.shape
    dk = tf.cast(tf.shape(g)[-1], tf.float32)
    s = tf.matmul(layer.hw_flatten(g), layer.hw_flatten(f), transpose_b=True) / tf.math.sqrt(dk)
    beta = tf.nn.softmax(s)
    transpose_a = not isinstance(layer, self_attentionDecoder)   # 編碼器版本有 transpose_a=True
    o = tf.matmul(beta, layer.hw_flatten(h), transpose_a=transpose_a)
    o = layer.reshape(o, height, width, ch)
    o = layer.last_(o)
    return GAMMA * o, beta


def norm(t):
    return tf.sqrt(tf.reduce_sum(tf.square(t), axis=[1, 2, 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--ckpt', default='epoch_200.ckpt')
    ap.add_argument('--data', required=True)
    ap.add_argument('--patients', nargs='*', default=None, help='預設：每個中心取測試名單的前 2 位')
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--height', type=int, default=304)
    ap.add_argument('--width', type=int, default=304)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    if args.patients:
        patients = args.patients
    else:
        patients = []
        for c in 'ABC':
            patients += [p for p in cfg.TEST_PATIENTS if p[2] == c][:2]

    # ---------- 載入整個 GAN（含判別器）----------
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

    # 取得每個 attention 層的輸入（建立子模型：網路輸入 → 該層輸入）
    probes = {}
    for net_name, net in (('gen', c_gen), ('disc', c_disc)):
        for lname in LAYERS[net_name]:
            layer = net.get_layer(lname)
            probes[lname] = (net, layer, tf.keras.Model(net.input, layer.input))

    dirA, dirB = os.path.join(args.data, 'trainA'), os.path.join(args.data, 'trainB')
    names = [f for f in sorted(os.listdir(dirA))
             if f.endswith('.npy') and f.split('_')[0] in patients][::args.every]
    if not names:
        sys.exit("找不到這些病人的切片")

    stats = defaultdict(list)
    sct_on, sct_off, cts, bodies = [], [], [], []
    for s in range(0, len(names), args.batch):
        chunk = names[s:s + args.batch]
        cb, ct = [], []
        for n in chunk:
            a = np.load(os.path.join(dirA, n)).astype(np.float32)
            b = np.load(os.path.join(dirB, n)).astype(np.float32)
            a = a[..., 0] if a.ndim == 3 else a
            b = b[..., 0] if b.ndim == 3 else b
            cb.append(pad_to(a, args.height, args.width)[0])
            ct.append(pad_to(b, args.height, args.width)[0])
        cb = np.stack(cb)[..., None]
        ct = np.stack(ct)[..., None]

        # 1~3：逐層量測（判別器量測時輸入真實 CT）
        for lname, (net, layer, sub) in probes.items():
            inp = cb if net is c_gen else ct
            x = sub(inp, training=False)
            branch, beta = branch_parts(layer, x)
            ln_with = layer.layernorm(branch + x)
            ln_without = layer.layernorm(x)
            stats[lname + '|ratio'].append((norm(branch) / norm(x)).numpy())
            stats[lname + '|effect'].append((norm(ln_with - ln_without) / norm(ln_without)).numpy())
            stats[lname + '|peak'].append(tf.reduce_mean(tf.reduce_max(beta, axis=-1), axis=-1).numpy())
            stats[lname + '|rows'] = int(beta.shape[-1])

        # 4：端對端，有 attention vs 關掉 attention
        sct_on.append(c_gen(cb, training=False).numpy())
        cts.append(ct)

    # 關掉所有 attention 分支：只保留 LayerNorm(x)
    orig_enc, orig_dec = self_attention.call, self_attentionDecoder.call
    self_attention.call = lambda self, x: self.layernorm(x)
    self_attentionDecoder.call = lambda self, x: self.layernorm(x)
    try:
        for s in range(0, len(names), args.batch):
            chunk = names[s:s + args.batch]
            cb = []
            for n in chunk:
                a = np.load(os.path.join(dirA, n)).astype(np.float32)
                a = a[..., 0] if a.ndim == 3 else a
                cb.append(pad_to(a, args.height, args.width)[0])
            sct_off.append(c_gen(np.stack(cb)[..., None], training=False).numpy())
    finally:
        self_attention.call, self_attentionDecoder.call = orig_enc, orig_dec

    on, off, ct = np.concatenate(sct_on), np.concatenate(sct_off), np.concatenate(cts)
    body = ct > -0.999

    print("=" * 96)
    print("1~3. 各 attention 層（數值為所有切片的平均）")
    print("=" * 96)
    print(f"  {'層':28s} {'分支/原特徵':>12s} {'對層輸出的影響':>14s} {'softmax 最大權重':>16s} {'(均分時)':>10s}")
    for lname in probes:
        r = np.concatenate(stats[lname + '|ratio']).mean()
        e = np.concatenate(stats[lname + '|effect']).mean()
        p = np.concatenate(stats[lname + '|peak']).mean()
        rows = stats[lname + '|rows']
        print(f"  {lname:28s} {r:12.4%} {e:14.4%} {p:16.3f} {1 / rows:10.4f}")

    print()
    print("=" * 96)
    print(f"4. 端對端：關掉所有 attention 分支後，生成器輸出的改變（{len(on)} 張切片，身體內）")
    print("=" * 96)
    diff = np.abs(on - off)[body] * HU_PER_UNIT
    mae_on = np.abs(on - ct)[body].mean() * HU_PER_UNIT
    mae_off = np.abs(off - ct)[body].mean() * HU_PER_UNIT
    print(f"  sCT 改變量：平均 {diff.mean():.3f} HU | 99% 分位數 {np.percentile(diff, 99):.3f} HU | 最大 {diff.max():.3f} HU")
    print(f"  相對真實 CT 的 body MAE（歸一化空間 ×300，未截斷）：有 attention {mae_on:.3f} HU | 關掉 {mae_off:.3f} HU")

    print()
    print("判讀：")
    max_ratio = max(np.concatenate(stats[l + '|ratio']).mean() for l in probes)
    if max_ratio < 0.01 and diff.mean() < 1.0:
        print(f"  - 所有 attention 分支都不到原特徵的 1%（最大 {max_ratio:.3%}），關掉後 sCT 平均只改變 {diff.mean():.2f} HU：")
        print("    attention 在這個模型中實質上幾乎沒有作用。")
    elif max_ratio < 0.05:
        print(f"  - attention 分支最大約為原特徵的 {max_ratio:.2%}，作用很小，但不是完全沒有；關掉後 sCT 平均改變 {diff.mean():.2f} HU。")
    else:
        print(f"  - attention 分支最大約為原特徵的 {max_ratio:.2%}：網路學會放大了這條分支，attention 有實際作用。")


if __name__ == '__main__':
    main()
