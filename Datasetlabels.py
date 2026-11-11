import os
import sys
import config  # 引入你設定好的 config.py


def get_filtered_files(dataset_root, subfolder, target_patients):
    """
    這個函式會去目標資料夾抓檔案，並且只保留在 target_patients 名單內的病人切片
    回傳的路徑是相對於 dataset_root 的（例如 'trainA/2BA001_050.npy'）
    """
    folder = os.path.join(dataset_root, subfolder)
    files = []

    # 遍歷資料夾內的所有檔案
    for f in sorted(os.listdir(folder)):
        if f.endswith('.npy'):
            # 抓出檔名前面的病人 ID (例如 "2BA001")
            patient_id = f.split('_')[0]

            # 判斷有沒有在訓練名單中
            if patient_id in target_patients:
                files.append(os.path.join(subfolder, f))

    return files


def get_train_files(dataset_root):
    """
    修正：原本在 import 時就用 config.basedir（寫死的路徑）建立檔案清單，
    導致 CLI 的 -d/--dataset 只影響「讀檔路徑」而不影響「檔案清單」，
    換資料集時清單仍來自舊資料夾。現在改成由 trainer 依 config.dataset 呼叫。
    """
    domainA = get_filtered_files(dataset_root, 'trainA', config.TRAIN_PATIENTS)
    domainB = get_filtered_files(dataset_root, 'trainB', config.TRAIN_PATIENTS)

    # 安全檢查：確保 A 和 B 數量一致
    assert len(domainA) == len(domainB), \
        f"trainA ({len(domainA)}) 與 trainB ({len(domainB)}) 數量不一致！請檢查資料集：{dataset_root}"
    assert len(domainA) > 0, f"在 {dataset_root} 找不到任何訓練切片！"

    # 雜訊資訊 -> stderr（進度/紀錄檔），不進 loss 檔
    print("=" * 50, file=sys.stderr)
    print(f"資料集：{dataset_root}", file=sys.stderr)
    print(f"共篩選出 {len(domainA)} 對切片進行訓練。", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    return domainA, domainB
