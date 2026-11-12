from tensorflow import keras
import tensorflow as tf

# 每個 epoch 要記錄的 loss（epoch 結束時為整個 epoch 的平均值）
LOSS_NAMES = [
    "DomA_gen_loss",   # A 域（CBCT）生成器總 loss = 對抗 + cycle + identity
    "DomB_gen_loss",   # B 域（CT）生成器總 loss
    "DomA_disc_loss",  # A 域判別器 loss
    "DomB_disc_loss",  # B 域判別器 loss
    "DomA_adv_loss",   # A 域生成器的對抗 loss（騙過判別器的程度）
    "DomB_adv_loss",   # B 域生成器的對抗 loss
    "cycle_loss",      # cycle consistency loss（A→B→A 與 B→A→B 的總和，已乘 lambda）
    "DomA_id_loss",    # A 域 identity loss（已乘權重）
    "DomB_id_loss",    # B 域 identity loss（已乘權重）
]


class CycleGan(keras.Model):
    def __init__(
        self,
        c_generator,
        s_generator,
        c_discriminator,
        s_discriminator,
        lambda_cycle=10,
        pool_size=50,
    ):
        super(CycleGan, self).__init__()
        self.c_gen = c_generator
        self.s_gen = s_generator


        self.c_disc = c_discriminator
        self.s_disc = s_discriminator
        self.lambda_cycle = lambda_cycle

        # ==============================================================
        # History pool（CycleGAN 原論文的 image buffer）
        # 判別器判斷假影像時，有 50% 機率改看「過去生成過的假影像」，
        # 避免判別器只針對生成器當下的狀態學習，讓訓練更穩定。
        # pool 以 tf.Variable 儲存，會隨 checkpoint 一起存檔，接續訓練時可還原。
        # pool_size=0 代表關閉（行為與原本完全相同）
        # ==============================================================
        self.pool_size = int(pool_size)
        if self.pool_size > 0:
            _, h, w, ch = self.c_gen.input_shape
            pool_shape = (self.pool_size, h, w, ch)
            # fake_s（生成的 CBCT）與 fake_c（生成的 CT）各一個 pool
            self.pool_s = tf.Variable(tf.zeros(pool_shape), trainable=False, name="pool_s")
            self.pool_c = tf.Variable(tf.zeros(pool_shape), trainable=False, name="pool_c")
            self.pool_s_count = tf.Variable(0, trainable=False, dtype=tf.int32, name="pool_s_count")
            self.pool_c_count = tf.Variable(0, trainable=False, dtype=tf.int32, name="pool_c_count")

        # 每個 loss 一個 Mean metric：Keras 會在每個 epoch 開始時自動歸零，
        # epoch 結束時得到的是「整個 epoch 的平均」，而不是最後一個 batch 的值
        self.loss_trackers = {name: keras.metrics.Mean(name=name) for name in LOSS_NAMES}

    @property
    def metrics(self):
        return list(self.loss_trackers.values())

    def compile(
        self,
        c_gen_optimizer,
        s_gen_optimizer,
        c_disc_optimizer,
        s_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.c_gen_optimizer = c_gen_optimizer
        self.s_gen_optimizer = s_gen_optimizer
        self.c_disc_optimizer = c_disc_optimizer
        self.s_disc_optimizer = s_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def _query_pool(self, pool, count, images):
        '''
        History pool 查詢（與 CycleGAN 官方 ImagePool 相同邏輯，逐張處理）：
          - pool 未滿：把新影像存進 pool，回傳新影像
          - pool 已滿：50% 機率回傳 pool 中隨機一張舊影像，並用新影像取代它；
                       50% 機率直接回傳新影像
        '''
        images = tf.stop_gradient(images)
        n = tf.shape(images)[0]

        def body(i, out):
            img = images[i]

            def fill():
                # pool 未滿：存入第 count 格
                c = count.read_value()
                with tf.control_dependencies([pool[c].assign(img)]):
                    with tf.control_dependencies([count.assign_add(1)]):
                        return tf.identity(img)

            def swap():
                # 回傳隨機一張舊影像，並用新影像取代它（先讀再寫）
                idx = tf.random.uniform([], 0, self.pool_size, dtype=tf.int32)
                old = tf.identity(pool[idx])
                with tf.control_dependencies([old]):
                    upd = pool[idx].assign(img)
                with tf.control_dependencies([upd]):
                    return tf.identity(old)

            def keep():
                return tf.identity(img)

            def full():
                return tf.cond(tf.random.uniform([]) > 0.5, swap, keep)

            res = tf.cond(count.read_value() < self.pool_size, fill, full)
            return i + 1, out.write(i, res)

        # parallel_iterations=1：逐張依序處理，確保 pool 的讀寫順序正確
        _, out = tf.while_loop(
            lambda i, _: i < n, body,
            [tf.constant(0), tf.TensorArray(tf.float32, size=n)],
            parallel_iterations=1,
        )
        return tf.ensure_shape(out.stack(), images.shape)

    def build_for_restore(self):
        '''
        先建立所有模型與 optimizer 的變數，讓 load_weights 能立即、完整地還原
        （包含 optimizer 狀態與 learning rate），接續訓練時使用。
        不會更新任何權重。
        '''
        dummy = tf.zeros((1,) + tuple(self.c_gen.input_shape[1:]))
        for m in (self.c_gen, self.s_gen, self.c_disc, self.s_disc):
            m(dummy, training=False)
        self.c_gen_optimizer.build(self.c_gen.trainable_variables)
        self.s_gen_optimizer.build(self.s_gen.trainable_variables)
        self.c_disc_optimizer.build(self.c_disc.trainable_variables)
        self.s_disc_optimizer.build(self.s_disc.trainable_variables)

    def train_step(self, batch_data):
        real_s, real_c = batch_data

        # 修正（TF 2.11+ / Keras 2.15）：fit() 讀取 Sequence 時，
        # 輸入張量的空間維度會變成 None（shape=(None, None, None, 1)），
        # 而 SelfAttention 的 Reshape 需要靜態的 H、W，因此在這裡明確指定形狀
        expected_shape = self.c_gen.input_shape   # (None, H, W, 1)
        real_s = tf.ensure_shape(real_s, expected_shape)
        real_c = tf.ensure_shape(real_c, expected_shape)

        with tf.GradientTape(persistent=True) as tape:

            fake_s = self.s_gen(real_c, training=True)
            cycled_c = self.c_gen(fake_s, training=True)

            fake_c = self.c_gen(real_s, training=True)
            cycled_s = self.s_gen(fake_c, training=True)

            # generating itself
            same_s = self.s_gen(real_s, training=True)
            same_c = self.c_gen(real_c, training=True)

            # discriminator used to check, inputing real images
            disc_real_s = self.s_disc(real_s, training=True)
            disc_real_c = self.c_disc(real_c, training=True)

            # discriminator used to check, inputing fake images
            # （生成器的對抗 loss 一律使用「當下」生成的假影像）
            disc_fake_s = self.s_disc(fake_s, training=True)
            disc_fake_c = self.c_disc(fake_c, training=True)

            # 判別器 loss 使用的假影像：開啟 history pool 時從 pool 取出
            if self.pool_size > 0:
                pooled_fake_s = self._query_pool(self.pool_s, self.pool_s_count, fake_s)
                pooled_fake_c = self._query_pool(self.pool_c, self.pool_c_count, fake_c)
                disc_pool_fake_s = self.s_disc(pooled_fake_s, training=True)
                disc_pool_fake_c = self.c_disc(pooled_fake_c, training=True)
            else:
                disc_pool_fake_s = disc_fake_s
                disc_pool_fake_c = disc_fake_c

            # evaluates generator loss
            s_gen_loss = self.gen_loss_fn(disc_fake_s)
            c_gen_loss = self.gen_loss_fn(disc_fake_c)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_s,
                                                  cycled_s,
                                                  self.lambda_cycle) + self.cycle_loss_fn(real_c,
                                                                     cycled_c,
                                                                     self.lambda_cycle)

            s_id_loss = self.identity_loss_fn(real_s, same_s, self.lambda_cycle)
            c_id_loss = self.identity_loss_fn(real_c, same_c, self.lambda_cycle)

            # evaluates total generator loss
            total_s_gen_loss = s_gen_loss + total_cycle_loss + s_id_loss
            total_c_gen_loss = c_gen_loss + total_cycle_loss + c_id_loss

            # evaluates discriminator loss
            s_disc_loss = self.disc_loss_fn(disc_real_s, disc_pool_fake_s)
            c_disc_loss = self.disc_loss_fn(disc_real_c, disc_pool_fake_c)


        # Calculate the gradients for generator and discriminator
        s_generator_gradients = tape.gradient(total_s_gen_loss,
                                                  self.s_gen.trainable_variables)
        c_generator_gradients = tape.gradient(total_c_gen_loss,
                                                  self.c_gen.trainable_variables)

        s_discriminator_gradients = tape.gradient(s_disc_loss,
                                                  self.s_disc.trainable_variables)
        c_discriminator_gradients = tape.gradient(c_disc_loss,
                                                  self.c_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.s_gen_optimizer.apply_gradients(zip(s_generator_gradients,
                                                 self.s_gen.trainable_variables))

        self.c_gen_optimizer.apply_gradients(zip(c_generator_gradients,
                                                 self.c_gen.trainable_variables))

        self.s_disc_optimizer.apply_gradients(zip(s_discriminator_gradients,
                                                  self.s_disc.trainable_variables))

        self.c_disc_optimizer.apply_gradients(zip(c_discriminator_gradients,
                                                  self.c_disc.trainable_variables))

        # 更新各項 loss 的 epoch 平均
        values = {
            "DomA_gen_loss": total_s_gen_loss,
            "DomB_gen_loss": total_c_gen_loss,
            "DomA_disc_loss": s_disc_loss,
            "DomB_disc_loss": c_disc_loss,
            "DomA_adv_loss": s_gen_loss,
            "DomB_adv_loss": c_gen_loss,
            "cycle_loss": total_cycle_loss,
            "DomA_id_loss": s_id_loss,
            "DomB_id_loss": c_id_loss,
        }
        for name, v in values.items():
            self.loss_trackers[name].update_state(tf.reduce_mean(v))
        return {name: t.result() for name, t in self.loss_trackers.items()}

    def call(self, x):
        pass
