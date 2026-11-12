import os # 新增：用來檢查路徑與檔案
import sys
from trainer import train
from validation import testModel
from DatasetLoader import GANDataGenerator, GANDataGeneratorXY
import numpy as np
from config import get_parameters

def main(config):
    if config.train:
        '''
           Train the CycleGAN Model
        '''
        # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔
        print("=== 啟動 CycleGAN 訓練模式 ===", file=sys.stderr)

        # 接續訓練 / 微調 / 全新訓練的判斷都在 trainer.train() 裡處理：
        #   model_save_path 有 last.ckpt → 自動接續；否則有 --init_ckpt → 微調；都沒有 → 全新訓練
        # 訓練結束時 train() 會儲存 model_weights（c_generator 權重）
        train(config)

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
