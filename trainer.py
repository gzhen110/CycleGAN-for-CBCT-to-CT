import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
from DatasetLoader import GANDataGenerator, GANDataGeneratorXY
from Datasetlabels import *
from losses import *
from generator import Generator
from discriminator import Discriminator
from CycleGAN import CycleGan

tf.compat.v1.reset_default_graph()

# ========================================================
# Learning Rate Linear Decay Callback
# ========================================================
class LinearDecayLR(tf.keras.callbacks.Callback):
    """
    前半段維持固定 lr，後半段線性遞減到 0
    例如 epoch=200，前 100 固定，後 100 線性降到 0
    """
    def __init__(self, initial_lr, total_epochs, decay_start_epoch, optimizers):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.decay_start_epoch = decay_start_epoch
        self.optimizers = optimizers

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.decay_start_epoch:
            new_lr = self.initial_lr * (1 - (epoch - self.decay_start_epoch) /
                                        (self.total_epochs - self.decay_start_epoch))
            new_lr = max(new_lr, 0.0)
            for opt in self.optimizers:
                opt.learning_rate.assign(new_lr)
            print(f"\n Epoch {epoch+1}: Learning Rate 更新為 {new_lr:.6f}", flush=True)

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
            print(f"\n [儲存點] 已成功儲存第 {current_epoch} 世代的模型權重至：{save_path}", flush=True)

# ========================================================
# 自訂 Callback - 每個 Epoch 結束只印一行 loss summary
# ========================================================
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        import numpy as np
        logs = logs or {}
        def to_scalar(val):
            if val is None:
                return float('nan')
            v = np.mean(val)
            return v.item() if hasattr(v, 'item') else float(v)
        print(
            f"Epoch {epoch+1:03d} | "
            f"DomA_gen_loss: {to_scalar(logs.get('DomA_gen_loss')):.4f} | "
            f"DomB_gen_loss: {to_scalar(logs.get('DomB_gen_loss')):.4f} | "
            f"DomA_disc_loss: {to_scalar(logs.get('DomA_disc_loss')):.4f} | "
            f"DomB_disc_loss: {to_scalar(logs.get('DomB_disc_loss')):.4f}",
            flush=True
        )

# ========================================================
# 主訓練函式
# ========================================================
def train(config, resume_ckpt=None):
    # 修正：使用 GANDataGeneratorXY 讓 CBCT 和 CT 成對輸入
    train_gen = GANDataGeneratorXY(
        domainA,                # CBCT
        domainB,                # CT
        config.dataset,
        config.dataset,
        config.batch,
        dim=(config.height, config.width)
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
        s_discriminator
    )

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
    # 自訂存檔：每隔 10 個 Epoch 存一次
    # ========================================================
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 檔名加上 {epoch:03d}，這樣存出來就會是 epoch_005.ckpt, epoch_010.ckpt...
    checkpoint_path = os.path.join(config.model_save_path, 'epoch_{epoch:03d}.ckpt')

    cp_callback = SaveEveryNEpochs(
        filepath=checkpoint_path,
        save_epochs=10  
    )

    # ========================================================
    # LinearDecayLR：前半段固定 lr，後半段線性遞減到 0
    # ========================================================
    decay_start = config.epoch // 2
    lr_decay_callback = LinearDecayLR(
        initial_lr=config.gen_lr,
        total_epochs=config.epoch,
        decay_start_epoch=decay_start,
        optimizers=[
            c_generator_optimizer,
            s_generator_optimizer,
            c_discriminator_optimizer,
            s_discriminator_optimizer
        ]
    )

    # ========================================================
    # EpochLogger：每個 Epoch 結束只印一行 loss
    # ========================================================
    epoch_logger = EpochLogger()

    # ========================================================
    # 斷點續傳：若有舊權重則載入
    # ========================================================
    if resume_ckpt:
        print(f"\n 正在將斷點權重 {resume_ckpt} 載入至 CycleGAN 模型中...")
        try:
            gan_model.load_weights(resume_ckpt)
            print("斷點權重載入成功！開始接續訓練。\n")
        except Exception as e:
            print(f"載入權重失敗：{e}。系統將從頭開始全新訓練。\n")

    # ========================================================
    # 開始訓練（verbose=0 關閉 per-batch 進度條）
    # ========================================================
    gan_model.fit(
        train_gen,
        epochs=config.epoch,
        callbacks=[cp_callback, lr_decay_callback, epoch_logger],
        verbose=0
    )

    # 訓練完後儲存 c_generator 權重
    c_generator.save_weights(os.path.join(config.model_save_path, 'model_weights'))
    print(f"--- 權重已成功儲存至 {config.model_save_path}model_weights ---")

    return c_generator
