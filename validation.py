from tensorflow import keras
import matplotlib.pyplot as plt
from generator import Generator
import os

def testModel(test_input=None, target=None, config=None):
    
    Eval_generator = Generator(config.height, config.width, config.alpha)
    Eval_generator.load_weights(config.model_save_path + 'model_weights')

    prediction = Eval_generator(test_input, training=False)
    
    # 確保資料是 numpy 格式
    p_img = prediction[0].numpy() if hasattr(prediction[0], 'numpy') else prediction[0]
    in_img = test_input[0].numpy() if hasattr(test_input[0], 'numpy') else test_input[0]

    if config.subject == 0:
        t_img = target[0].numpy() if hasattr(target[0], 'numpy') else target[0]
        display_list = [in_img, t_img, p_img]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        
        plt.figure(figsize=(12, 4)) # 調整畫布大小方便觀察
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # 加上 cmap='gray' 並將最後的單通道維度壓平 [256, 256, 1] -> [256, 256]
            plt.imshow((display_list[i][:, :, 0] + 1) / 2.0, cmap='gray')
            plt.axis('off')
    
    if config.subject == 1:
        title = ['Input Image', 'Predicted Image']
        display_list = [in_img, p_img]

        plt.figure(figsize=(8, 4))
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            plt.imshow((display_list[i][:, :, 0] + 1) / 2.0, cmap='gray')
            plt.axis('off')

    # --- 新增的程式碼：調整子圖之間的間距與邊緣 ---
    # wspace 控制圖片之間的水平留白，數值越小靠得越近（可依喜好微調，例如 0.05 或 0.1）
    plt.subplots_adjust(wspace=0.05, left=0.02, right=0.98, bottom=0.02, top=0.9)

    output_path = f"Gan_Output_{config.validate.replace('.npy', '')}.png"
    
    # 修改的程式碼：在 savefig 加上 bbox_inches='tight' 來消除外圍多餘的白色邊框
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    
    print(f"--- 成功！成果圖已儲存至專案目錄下的: {output_path} ---")
    plt.close() # 關閉畫布釋放記憶體
    print(f"--- 成果對比圖已成功儲存為 Gan_Output_{config.validate} ---")