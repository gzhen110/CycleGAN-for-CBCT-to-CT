import os
# 若外部（例如國網中心的 Slurm）已指定 GPU，就不覆寫；本機預設仍用 GPU 1
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 關掉 TensorFlow 自己的 C++ 層級日誌（oneDNN / cuBLAS / NUMA 等）

import sys
import time
import numpy as np
import tensorflow as tf
from DatasetLoader import GANDataGenerator, GANDataGeneratorXY
from Datasetlabels import get_train_files
from losses import *
from generator import Generator
from discriminator import Discriminator
from CycleGAN import CycleGan, LOSS_NAMES

tf.compat.v1.reset_default_graph()

# 接續訓練用的檔名（都存在 model_save_path 底下）
LAST_CKPT_NAME = 'last.ckpt'          # 每個 epoch 結束都覆寫的完整權重（含 optimizer 狀態與 history pool）
LAST_EPOCH_FILE = 'last_epoch.txt'    # 記錄 last.ckpt 是第幾個 epoch 完成時存的
LOSS_CSV_NAME = 'loss_log.csv'        # 每個 epoch 的各項 loss（接續訓練時會接著寫）


def lr_for_epoch(epoch, initial_lr, total_epochs, decay_start_epoch):
    '''
    第 epoch 個 epoch（從 0 開始）要用的 learning rate。
    與原本 LinearDecayLR 的時程完全相同：
    前半段固定，第 decay_start_epoch 個 epoch 結束後開始線性遞減。
    '''
    prev = epoch - 1
    if prev < decay_start_epoch:
        return initial_lr
    lr = initial_lr * (1 - (prev - decay_start_epoch) / (total_epochs - decay_start_epoch))
    return max(lr, 0.0)


# ========================================================
# Learning Rate Linear Decay Callback
# ========================================================
class LinearDecayLR(tf.keras.callbacks.Callback):
    """
    前半段維持固定 lr，後半段線性遞減到 0
    例如 epoch=200，前 100 固定，後 100 線性降到 0

    修改：改在每個 epoch「開始時」直接依 epoch 編號設定 lr，
    這樣接續訓練時不管從第幾個 epoch 開始，lr 都會是正確的值。
    時程與原本（在 epoch 結束時更新下一個 epoch 的 lr）完全相同。
    """
    def __init__(self, initial_lr, total_epochs, decay_start_epoch, optimizers, label=''):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.decay_start_epoch = decay_start_epoch
        self.optimizers = optimizers
        self.label = label
        self._last_lr = None

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = lr_for_epoch(epoch, self.initial_lr, self.total_epochs, self.decay_start_epoch)
        for opt in self.optimizers:
            opt.learning_rate.assign(new_lr)
        if new_lr != self._last_lr:
            # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔
            print(f"\n Epoch {epoch+1}: {self.label} Learning Rate = {new_lr:.7f}", file=sys.stderr, flush=True)
            self._last_lr = new_lr

# ========================================================
# 自訂 Callback - 每隔 N 個 Epoch 存一次檔
# ========================================================
class SaveEveryNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_epochs=5):
        super().__init__()
        self.filepath = filepath
        self.save_epochs = save_epochs

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1  # Keras 的 epoch 從 0 開始算，所以加 1

        # 判斷是否為指定的倍數 (例如 5, 10, 15...)
        if current_epoch % self.save_epochs == 0:
            # 將路徑中的 {epoch:03d} 替換成真實的數字
            save_path = self.filepath.format(epoch=current_epoch)

            # 存取整個 GAN 模型的權重
            self.model.save_weights(save_path)
            # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔
            print(f"\n [儲存點] 已成功儲存第 {current_epoch} 世代的模型權重至：{save_path}",
                  file=sys.stderr, flush=True)

# ========================================================
# 自訂 Callback - 每個 Epoch 都覆寫一份 last.ckpt（接續訓練用）
# ========================================================
class SaveLastCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.ckpt_path = os.path.join(save_dir, LAST_CKPT_NAME)
        self.epoch_file = os.path.join(save_dir, LAST_EPOCH_FILE)

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.ckpt_path)
        # 先寫暫存檔再改名，避免寫到一半被中斷造成檔案損毀
        tmp = self.epoch_file + '.tmp'
        with open(tmp, 'w') as f:
            f.write(str(epoch + 1))
        os.replace(tmp, self.epoch_file)

# ========================================================
# 自訂 Callback - 印出目前訓練到第幾個 epoch、耗時多久（給 progress 檔用）
# ========================================================
class ProgressLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        epoch_secs = now - self.epoch_start
        total_secs = now - self.train_start

        def fmt(secs):
            secs = int(secs)
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        current_epoch = epoch + 1
        total_epochs = self.params.get('epochs', '?')
        print(
            f"[進度] Epoch {current_epoch}/{total_epochs} 完成 | "
            f"本 epoch 耗時: {fmt(epoch_secs)} | "
            f"本次執行累計耗時: {fmt(total_secs)}",
            file=sys.stderr, flush=True
        )

# ========================================================
# 自訂 Callback - 每個 Epoch 結束印一行 loss summary，並寫入 CSV
# ========================================================
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, csv_path=None):
        super().__init__()
        self.csv_path = csv_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        def to_scalar(val):
            if val is None:
                return float('nan')
            v = np.mean(val)
            return v.item() if hasattr(v, 'item') else float(v)
        vals = {name: to_scalar(logs.get(name)) for name in LOSS_NAMES}

        # 乾淨的 loss 資訊 -> 維持在 stdout（loss 檔）
        # 前四項維持原本格式；後面接上分項 loss
        print(f"Epoch {epoch+1:03d} | " + " | ".join(f"{n}: {vals[n]:.4f}" for n in LOSS_NAMES),
              flush=True)

        # 同時寫入 CSV（方便之後畫 loss curve；接續訓練時會接在後面）
        if self.csv_path:
            new_file = not os.path.exists(self.csv_path)
            with open(self.csv_path, 'a') as f:
                if new_file:
                    f.write('epoch,' + ','.join(LOSS_NAMES) + '\n')
                f.write(f"{epoch+1}," + ','.join(f"{vals[n]:.6f}" for n in LOSS_NAMES) + '\n')

# ========================================================
# 主訓練函式
# ========================================================
def train(config):
    os.makedirs(config.model_save_path, exist_ok=True)
    last_ckpt = os.path.join(config.model_save_path, LAST_CKPT_NAME)
    last_epoch_file = os.path.join(config.model_save_path, LAST_EPOCH_FILE)
    loss_csv = os.path.join(config.model_save_path, LOSS_CSV_NAME)

    # ========================================================
    # 決定訓練起點：
    #   1. model_save_path 裡有 last.ckpt → 接續訓練（優先，即使有給 --init_ckpt）
    #   2. 有給 --init_ckpt → 從預訓練權重開始微調，epoch 從 0 開始
    #   3. 都沒有 → 全新訓練
    # ========================================================
    initial_epoch = 0
    load_path, load_mode = None, None
    if os.path.exists(last_ckpt + '.index') and os.path.exists(last_epoch_file):
        with open(last_epoch_file) as f:
            initial_epoch = int(f.read().strip())
        load_path, load_mode = last_ckpt, 'resume'
        print(f"偵測到接續訓練存檔：{last_ckpt}（已完成 {initial_epoch} 個 epoch），"
              f"將從第 {initial_epoch + 1} 個 epoch 接續訓練", file=sys.stderr)
    elif config.init_ckpt:
        if not os.path.exists(config.init_ckpt + '.index'):
            sys.exit(f"找不到預訓練權重：{config.init_ckpt}.index")
        load_path, load_mode = config.init_ckpt, 'finetune'
        print(f"微調模式：從預訓練權重 {config.init_ckpt} 開始，epoch 從 0 開始", file=sys.stderr)
    else:
        print("未偵測到接續訓練存檔，將從 Epoch 0 開始全新訓練！", file=sys.stderr)

    if initial_epoch >= config.epoch:
        print(f"已完成全部 {config.epoch} 個 epoch，不需要再訓練。", file=sys.stderr)

    # 固定亂數種子，讓不同資料集的實驗可以公平比較
    # （接續訓練時加上 initial_epoch，避免重複使用第一段的打亂順序）
    # 注意：必須在建立 data generator 之前設定，因為建立時就會打亂一次
    tf.random.set_seed(config.seed + initial_epoch)
    np.random.seed(config.seed + initial_epoch)

    # 依 CLI 的 -d/--dataset 建立檔案清單（不再使用 config.basedir 寫死的路徑）
    domainA, domainB = get_train_files(config.dataset)

    # 修正：使用 GANDataGeneratorXY 讓 CBCT 和 CT 成對輸入
    # paired=config.paired：由 CLI 的 --paired 參數控制 paired / unpaired 訓練策略
    train_gen = GANDataGeneratorXY(
        domainA,                # CBCT
        domainB,                # CT
        config.dataset,
        config.dataset,
        config.batch,
        dim=(config.height, config.width),
        paired=config.paired
    )

    c_generator     = Generator(config.height, config.width, config.alpha)
    s_generator     = Generator(config.height, config.width, config.alpha)
    c_discriminator = Discriminator(config.height, config.width, config.alpha)
    s_discriminator = Discriminator(config.height, config.width, config.alpha)

    c_generator_optimizer     = tf.keras.optimizers.Adam(config.gen_lr, beta_1=config.beta1,
                                                          beta_2=config.beta2, epsilon=config.epsilon)
    s_generator_optimizer     = tf.keras.optimizers.Adam(config.gen_lr, beta_1=config.beta1,
                                                          beta_2=config.beta2, epsilon=config.epsilon)
    c_discriminator_optimizer = tf.keras.optimizers.Adam(config.dis_lr, beta_1=config.beta1,
                                                          beta_2=config.beta2, epsilon=config.epsilon)
    s_discriminator_optimizer = tf.keras.optimizers.Adam(config.dis_lr, beta_1=config.beta1,
                                                          beta_2=config.beta2, epsilon=config.epsilon)

    gan_model = CycleGan(
        c_generator, s_generator, c_discriminator,
        s_discriminator,
        pool_size=config.pool_size
    )
    print(f"History pool：{'關閉' if config.pool_size == 0 else f'{config.pool_size} 張'}", file=sys.stderr)

    gan_model.compile(
        c_gen_optimizer   = c_generator_optimizer,
        s_gen_optimizer   = s_generator_optimizer,
        c_disc_optimizer  = c_discriminator_optimizer,
        s_disc_optimizer  = s_discriminator_optimizer,
        gen_loss_fn       = generator_loss,
        disc_loss_fn      = discriminator_loss,
        cycle_loss_fn     = calc_cycle_loss,
        identity_loss_fn  = identity_loss
    )

    # ========================================================
    # 載入權重（接續訓練 / 微調）
    # 先建立所有變數（含 optimizer），再載入，才能完整還原 optimizer 狀態與 history pool
    # ========================================================
    if load_path:
        gan_model.build_for_restore()
        print(f"\n 正在載入權重 {load_path} ...", file=sys.stderr)
        status = gan_model.load_weights(load_path)
        if load_mode == 'resume':
            # 接續訓練：存檔與目前模型結構完全相同，任何對不上的地方都直接報錯
            status.assert_existing_objects_matched()
        else:
            # 微調：預訓練權重可能來自不同設定（例如沒有 history pool），允許部分載入
            status.expect_partial()
        print("權重載入成功！\n", file=sys.stderr)

    # ========================================================
    # 自訂存檔：每隔 10 個 Epoch 存一次（保留，方便之後挑選）
    # ========================================================
    # 檔名加上 {epoch:03d}，這樣存出來就會是 epoch_010.ckpt, epoch_020.ckpt...
    checkpoint_path = os.path.join(config.model_save_path, 'epoch_{epoch:03d}.ckpt')

    cp_callback = SaveEveryNEpochs(
        filepath=checkpoint_path,
        save_epochs=10
    )
    last_callback = SaveLastCheckpoint(config.model_save_path)

    # ========================================================
    # LinearDecayLR：前半段固定 lr，後半段線性遞減到 0
    # 修正：G 和 D 分開控制，各自用正確的 initial_lr
    # 原本的 bug：全部用 gen_lr，導致 decay 後 D 的 lr 被錯誤覆寫成 gen_lr
    # ========================================================
    decay_start = config.epoch // 2
    lr_decay_callback_gen = LinearDecayLR(
        initial_lr=config.gen_lr,       # Generator 用 gen_lr
        total_epochs=config.epoch,
        decay_start_epoch=decay_start,
        optimizers=[
            c_generator_optimizer,
            s_generator_optimizer
        ],
        label='Generator'
    )
    lr_decay_callback_dis = LinearDecayLR(
        initial_lr=config.dis_lr,       # Discriminator 用 dis_lr（gen_lr 的 1/10）
        total_epochs=config.epoch,
        decay_start_epoch=decay_start,
        optimizers=[
            c_discriminator_optimizer,
            s_discriminator_optimizer
        ],
        label='Discriminator'
    )

    # ========================================================
    # EpochLogger：每個 Epoch 結束印一行 loss，並寫入 loss_log.csv
    # ========================================================
    epoch_logger = EpochLogger(csv_path=loss_csv)
    progress_logger = ProgressLogger()

    # ========================================================
    # 開始訓練（verbose=0 關閉 per-batch 進度條）
    # workers：多執行緒同時讀取資料（只影響速度，不影響訓練品質）
    # ========================================================
    if initial_epoch < config.epoch:
        gan_model.fit(
            train_gen,
            epochs=config.epoch,
            initial_epoch=initial_epoch,
            callbacks=[lr_decay_callback_gen, lr_decay_callback_dis,
                       cp_callback, last_callback, epoch_logger, progress_logger],
            verbose=0,
            workers=config.workers,
            use_multiprocessing=False,
            max_queue_size=max(10, config.workers * 2),
        )

    # 訓練完後儲存 c_generator 權重
    c_generator.save_weights(os.path.join(config.model_save_path, 'model_weights'))
    # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔
    print(f"--- 權重已成功儲存至 {os.path.join(config.model_save_path, 'model_weights')} ---", file=sys.stderr)

    return c_generator
