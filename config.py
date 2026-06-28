import argparse

# 這裡設定你切好的 .npy 檔案存放的根目錄
basedir = '/home/qjanet/data/medical_npy/'

# ==============================================================================
# 新增：資料集病人切分 (Dataset Patient Splits)
# ==============================================================================
TRAIN_PATIENTS = [
    '2BA001', '2BA002', '2BA003', '2BA004', '2BA005', '2BA006', '2BA008', '2BA009', 
    '2BA010', '2BA011', '2BA012', '2BA013', '2BA023', '2BA024', '2BA025', '2BA027', 
    '2BA029', '2BA030', '2BA031', '2BA032', '2BA036', '2BA039', '2BA040', '2BA041', 
    '2BA045', '2BA047', '2BA051', '2BA052', '2BA053', '2BA055', '2BA058', '2BA059', 
    '2BA061', '2BA064', '2BA066', '2BA067', '2BA068', '2BA069', '2BA071', '2BA073', 
    '2BA074', '2BA076', '2BA080', '2BA081', '2BA083', '2BA085', '2BA090', '2BA092', 
    '2BA094', '2BB002', '2BB003', '2BB005', '2BB006', '2BB008', '2BB016', '2BB026', 
    '2BB030', '2BB031', '2BB033', '2BB039', '2BB041', '2BB043', '2BB044', '2BB048', 
    '2BB050', '2BB051', '2BB052', '2BB059', '2BB062', '2BB066', '2BB071', '2BB072', 
    '2BB073', '2BB076', '2BB079', '2BB082', '2BB085', '2BB090', '2BB091', '2BB095', 
    '2BB098', '2BB099', '2BB100', '2BB102', '2BB111', '2BB145', '2BB152', '2BB171', 
    '2BB173', '2BB177', '2BB182', '2BB184', '2BB189', '2BB200', '2BB205', '2BC001', 
    '2BC002', '2BC006', '2BC010', '2BC011', '2BC012', '2BC013', '2BC015', '2BC017', 
    '2BC018', '2BC019', '2BC022', '2BC023', '2BC024', '2BC026', '2BC030', '2BC033', 
    '2BC035', '2BC038', '2BC040', '2BC043', '2BC044', '2BC045', '2BC046', '2BC047', 
    '2BC048', '2BC049', '2BC050', '2BC051', '2BC055', '2BC057', '2BC059', '2BC060', 
    '2BC061', '2BC064', '2BC066', '2BC068', '2BC071', '2BC072', '2BC073', '2BC074', 
    '2BC075', '2BC076', '2BC079', '2BC082', '2BC085', '2BC086', '2BC088', '2BC089', 
    '2BC090'
]

VAL_PATIENTS = [
    '2BA037', '2BA044', '2BA088', '2BB034', '2BB083', '2BB096', '2BB198',
    '2BC016', '2BC027', '2BC037', '2BC056', '2BC081'
]

TEST_PATIENTS = [
    '2BA007', '2BA014', '2BA028', '2BA035', '2BA049', '2BA056', '2BA063', '2BA077', 
    '2BB007', '2BB011', '2BB017', '2BB028', '2BB049', '2BB075', '2BB109', '2BB151', 
    '2BB175', '2BB179', '2BC007', '2BC014', '2BC021', '2BC042', '2BC070', '2BC084'
]
# ==============================================================================

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()
    # 資料集根目錄
    parser.add_argument('-d', '--dataset', dest='dataset', default=basedir, 
                        help="Path to the dataset directory.")
    # 模型儲存路徑
    parser.add_argument('-m', '--model_save_path', type=str, default='./models/',
                        help="Path to Saved model directory.")
    
    # 影像設定
    # 醫學影像通常是 256x256 或 512x512
    parser.add_argument('--height', dest='height', default=256, type=int,
                        help="Height of the image")
    parser.add_argument('--width', dest='width', default=256, type=int,
                        help="Width of the image")
    
    # --- 關鍵修改：通道數 ---
    # 因為 CBCT/CT 是灰階圖，所以通道數 (n_channels) 必須設為 1
    parser.add_argument('--channels', dest='channels', default=1, type=int,
                        help="Number of channels (1 for grayscale medical images)")

    # 訓練相關
    parser.add_argument('-t','--train', type=str2bool, default=True, help="True when train the model")
    parser.add_argument('-v','--validate', default=None, type=str, help="Path for validation")

    # 超參數 (Hyperparameters)
    # Batch size 對於醫學影像建議先從 1 開始，避免記憶體溢位
    parser.add_argument('-b', '--batch', dest='batch', type=int, default=1)
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-b1', '--beta1', type=float, default=0.5) # CycleGAN 常用 0.5
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-g_lr', '--gen_lr', type=float, default=0.0002) # 常用 Learning Rate
    parser.add_argument('-d_lr', '--dis_lr', type=float, default=0.00002) # 讓 Discriminator 學習速度只有 Generator 的 1/10，給 Generator 更多喘息空間。
    
    # 權重參數 (CycleGAN 核心)
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument('--lambda_id', type=float, default=5.0, help="Identity loss weight")
    parser.add_argument('--epsilon', type=float, default=1e-7, help="Adam optimizer epsilon")
    parser.add_argument('--epoch', type=int, default=100, help="Number of training epochs")

    # 補回主體設定，避免 trainer.py 報錯（預設設為 0）
    parser.add_argument('-s', '--subject', dest='subject', type=int, default=0)

    return parser.parse_args()