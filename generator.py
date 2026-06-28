from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf
from AutoEncoderBlocks import downsample, upsample
from SelfAttention import self_attention, self_attentionDecoder

def Generator(HEIGHT, WIDTH, alpha):
    # --- 修改點 1：輸入通道改為 1 (灰階) ---
    inputs = layers.Input(shape=[HEIGHT, WIDTH, 1]) 
    
    # --- 修改點 2：輸出通道改為 1 (生成 CT 灰階圖) ---
    OUTPUT_CHANNELS = 1 

    down_stack = [
        downsample(64, 4, alpha, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4, alpha), # (bs, 64, 64, 128)
        downsample(256, 4, alpha), # (bs, 32, 32, 256)
        downsample(512, 4, alpha), # (bs, 16, 16, 512)
    ]

    up_stack = [
        upsample(512, 4, True), # (bs, 16, 16, 1024)
        upsample(256, 4, True), # (bs, 32, 32, 512)
        upsample(128, 4),       # (bs, 64, 64, 256)
        # 注意：最後一個 upsample 的輸出通道決定了之後 Concatenate 的維度
        upsample(64, 4, True),  # (bs, 128, 128, 128)
    ]

    # initializer = tf.random_normal_initializer(0., 0.02)
    # 最後一層轉置卷積，輸出 1 通道的 CT 影像
    #last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
    #                              strides=2,
    #                              padding='same',
    #                              kernel_initializer=initializer,
    #                              activation='tanh') 
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # --- 修改點 3：根除最後一層的棋盤效應 (Resize-Convolution) ---
    # 刪除原先的 Conv2DTranspose，改用 UpSampling2D + Conv2D
    last = keras.Sequential([
        layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
        layers.Conv2D(OUTPUT_CHANNELS, kernel_size=3, strides=1, padding='same',
                      kernel_initializer=initializer, activation='tanh')
    ], name="final_resize_conv")

    x = inputs

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # --- 修正點 1：使用 name_scope 隔離 Bottleneck Attention 的名字 ---
    with tf.name_scope("bottleneck_attention"):
        attention0 = self_attention(512, name='gen_bottleneck_attention')
        x = attention0(x)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
        
    # --- 修正點 2：使用 name_scope 隔離 Decoder Attention 的名字 ---
    with tf.name_scope("decoder_attention"):
        attention1 = self_attentionDecoder(192, name='gen_decoder_attention')
        x = attention1(x)
        
    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)