import os # 新增：用來檢查路徑與檔案
import sys
from trainer import train
from validation import testModel
from DatasetLoader import GANDataGenerator, GANDataGeneratorXY
import numpy as np
from Datasetlabels import *
from config import get_parameters

def main(config):
    if config.train:
        '''
           Train the CycleGAN Model
        '''
        # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔
        print("=== 啟動 CycleGAN 訓練模式 ===", file=sys.stderr)

        # 1. 定義我們預期存檔的名稱與路徑 (假設存在 model_save_path 裡)
        # 注意：如果你使用的是 PyTorch Lightning，存檔通常是 .ckpt 副檔名
        ckpt_path = os.path.join(config.model_save_path, 'last.ckpt')

        # 2. 檢查是否有舊的存檔
        # 修正：原本這裡分成兩個獨立的 if（一個檢查 .index，一個檢查原始路徑），
        # 沒有用 else 串起來，導致「偵測到舊存檔」跟「未偵測到舊存檔」曾經同時被印出。
        # 統一改成一個判斷式，TensorFlow 的 checkpoint 用 .index 檔判斷是否存在即可。
        if os.path.exists(ckpt_path + '.index'):
            print(f"偵測到舊存檔：{ckpt_path}", file=sys.stderr)
            print("將從斷點接續訓練！", file=sys.stderr)
            resume_ckpt = ckpt_path
        else:
            print("未偵測到舊存檔，將從 Epoch 0 開始全新訓練！", file=sys.stderr)
            resume_ckpt = None

        # 3. 將存檔路徑傳入 train 函數中
        # (你需要去 trainer.py 裡面，讓 train() 函數能接收這個參數)
        model = train(config, resume_ckpt=resume_ckpt)

        # saving model weights (訓練全部完成後儲存最終權重)
        model.save_weights(config.model_save_path + 'model_weights')

    else:
        '''
           Validate the CycleGAN Model
        '''
        validation_image_path = np.array([config.validate])
        if config.subject==0:
            # dataset-loader used in case of sketch to colorize image
            loader = GANDataGenerator(validation_image_path,
                                        config.dataset,
                                        1,
                                        dim = (config.height, config.width)
                                        )
        else:
            # dataset-loader used in case of gender-bender and glass to no-glass
            loader = GANDataGeneratorXY(validation_image_path,
                                        validation_image_path,
                                        config.dataset,
                                        1,
                                        dim = (config.height, config.width)
                                        )
        source, destination = next(iter(loader))
        testModel(source, destination, config)

if __name__ == '__main__':
    config = get_parameters()
    # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔
    print(config, file=sys.stderr)
    main(config)
