import os
import logging

# 必須在 import tensorflow 之前設定，阻斷 C++ 層級的矩陣與警告傾印
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 靜音 Python 層級的 TensorFlow 警告
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from generator import Generator
from discriminator import Discriminator
from CycleGAN import CycleGan
from losses import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss

# ========================================================
# 設定
# ========================================================
class DummyConfig:
    height = 256
    width = 256
    alpha = 0.1
    model_save_path = './models/'
    ckpt_path = './models/model_weights'
    test_samples_dir = './test_samples'
    beta1 = 0.5
    beta2 = 0.999
    gen_lr = 0.0002
    dis_lr = 0.0002
    epsilon = 1e-7

config = DummyConfig()

mtime = os.path.getmtime(config.ckpt_path + '.index')
# ========================================================
# 讀取 log.txt 取得目前 epoch 和 loss
# ========================================================
current_epoch = '未知'
last_loss_str = ''

if os.path.exists('log.txt'):
    with open('log.txt', 'r') as f:
        lines = f.readlines()
    # 找最後一筆有 "step" 或 "loss" 資訊的行
    for line in reversed(lines):
        if 'DomA_gen_loss' in line or 'Epoch' in line:
            last_loss_str = line.strip()
            break
    # 找目前 epoch 數
    for line in reversed(lines):
        if 'Epoch' in line:
            try:
                current_epoch = line.split('Epoch')[1].split('/')[0].strip()
            except:
                pass
            break

# ========================================================
# 載入權重
# ========================================================
print(f" 此權重檔案最後更新時間：{datetime.datetime.fromtimestamp(mtime)}")
print(f"\n--- 正在從 checkpoints/last.ckpt 載入權重（訓練至 Epoch {current_epoch}）---")

# 1. 保持原有的模型初始化與編譯
c_gen  = Generator(config.height, config.width, config.alpha)
s_gen  = Generator(config.height, config.width, config.alpha)
c_disc = Discriminator(config.height, config.width, config.alpha)
s_disc = Discriminator(config.height, config.width, config.alpha)

gan_model = CycleGan(c_gen, s_gen, c_disc, s_disc)
gan_model.compile(
    c_gen_optimizer  = tf.keras.optimizers.Adam(config.gen_lr, beta_1=config.beta1,
                                                 beta_2=config.beta2, epsilon=config.epsilon),
    s_gen_optimizer  = tf.keras.optimizers.Adam(config.gen_lr, beta_1=config.beta1,
                                                 beta_2=config.beta2, epsilon=config.epsilon),
    c_disc_optimizer = tf.keras.optimizers.Adam(config.dis_lr, beta_1=config.beta1,
                                                 beta_2=config.beta2, epsilon=config.epsilon),
    s_disc_optimizer = tf.keras.optimizers.Adam(config.dis_lr, beta_1=config.beta1,
                                                 beta_2=config.beta2, epsilon=config.epsilon),
    gen_loss_fn      = generator_loss,
    disc_loss_fn     = discriminator_loss,
    cycle_loss_fn    = calc_cycle_loss,
    identity_loss_fn = identity_loss
)

# 2. 插入假資料觸發內部元件變數初始化
dummy_input = tf.zeros([1, config.height, config.width, 1])
_ = c_gen(dummy_input, training=False)
_ = s_gen(dummy_input, training=False)
_ = c_disc(dummy_input, training=False)
_ = s_disc(dummy_input, training=False)
# 先呼叫一次 train_step 讓 optimizer 變數也初始化
dummy_batch = (dummy_input, dummy_input)
_ = gan_model.train_step(dummy_batch)

# 3. 載入權重：將狀態物件賦值給變數 load_status，避免立即被回收而觸發警告傾印
load_status = gan_model.load_weights(config.ckpt_path)
load_status.expect_partial()

# 4. 指定評估用的生成器
Eval_generator = c_gen
print("權重載入成功！\n")

Eval_generator = c_gen

# ========================================================
# 自動配對 sCT 和 realCT
# ========================================================
all_files = os.listdir(config.test_samples_dir)
sct_files = sorted([f for f in all_files if f.endswith('_sCT.npy')])

print(f"共找到 {len(sct_files)} 對測試切片：")
for f in sct_files:
    print(f"  {f}")
print()

# ========================================================
# 歸一化函式
# ========================================================
def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    if img_max - img_min != 0:
        return 2.0 * (img - img_min) / (img_max - img_min) - 1.0
    return img * 0.0

# ========================================================
# 逐一計算指標
# ========================================================
psnr_list = []
ssim_list = []
mae_list  = []
rmse_list = []

print("=" * 60)
print(f"  測試結果（訓練至 Epoch {current_epoch}）")
if last_loss_str:
    print(f"  最後 Loss 記錄：{last_loss_str}")
print("=" * 60)

for sct_fname in sct_files:
    # 自動找對應的 realCT 檔名
    realct_fname = sct_fname.replace('_sCT.npy', '_realCT.npy')
    sct_path    = os.path.join(config.test_samples_dir, sct_fname)
    realct_path = os.path.join(config.test_samples_dir, realct_fname)

    if not os.path.exists(realct_path):
        print(f"找不到對應的 realCT：{realct_fname}，跳過")
        continue

    # 讀取影像
    sct_img    = np.load(sct_path)
    realct_img = np.load(realct_path)

    # 歸一化
    sct_norm    = normalize(sct_img)
    realct_norm = normalize(realct_img)

    # Resize 並送入模型
    sct_4d     = np.expand_dims(np.expand_dims(sct_norm, axis=-1), axis=0)
    test_input = tf.image.resize(sct_4d, [config.height, config.width]).numpy()

    realct_4d      = np.expand_dims(np.expand_dims(realct_norm, axis=-1), axis=0)
    realct_resized = tf.image.resize(realct_4d, [config.height, config.width]).numpy()[0, :, :, 0]

    # 模型推論
    prediction = Eval_generator(test_input, training=False)
    ct_pred    = prediction[0, :, :, 0].numpy()

    # 轉回 [0, 1]
    gt_01   = (realct_resized + 1.0) / 2.0
    pred_01 = (ct_pred + 1.0) / 2.0

    # 計算指標
    p_val = psnr(gt_01, pred_01, data_range=1.0)
    s_val = ssim(gt_01, pred_01, data_range=1.0, channel_axis=None)

    # MAE / RMSE 換算回 HU
    ct_min, ct_max = np.min(realct_img), np.max(realct_img)
    pred_hu = pred_01 * (ct_max - ct_min) + ct_min
    gt_hu   = gt_01   * (ct_max - ct_min) + ct_min

    m_val = np.mean(np.abs(gt_hu - pred_hu))
    r_val = np.sqrt(np.mean((gt_hu - pred_hu) ** 2))

    psnr_list.append(p_val)
    ssim_list.append(s_val)
    mae_list.append(m_val)
    rmse_list.append(r_val)

    # 印出每張切片的個別結果
    slice_name = sct_fname.replace('_sCT.npy', '')
    print(f"\n{slice_name}")
    print(f"   PSNR : {p_val:.2f} dB")
    print(f"   SSIM : {s_val:.4f}")
    print(f"   MAE  : {m_val:.2f} HU")
    print(f"   RMSE : {r_val:.2f} HU")

# ========================================================
# 平均結果
# ========================================================
print("\n" + "=" * 60)
print("  平均指標（所有測試切片）")
print("=" * 60)
print(f"1. PSNR : {np.mean(psnr_list):.2f} dB")
print(f"2. SSIM : {np.mean(ssim_list):.4f}")
print(f"3. MAE  : {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f} HU")
print(f"4. RMSE : {np.mean(rmse_list):.2f} ± {np.std(rmse_list):.2f} HU")
print("=" * 60)