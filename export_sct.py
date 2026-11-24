"""
把模型產生的 sCT 匯出成醫學影像檔（NIfTI 及／或 DICOM），下載後用電腦上的醫學影像程式檢視

做法：
    1. 讀 evaluate_tf.py --save_npy 存下的 <病人>_sCT.npy（HU）
    2. 依切片檔名（<病人>_<z>.npy）把每張 sCT 放回原始 3D 體積的第 z 張
    3. 套用原始 ct.nii.gz 的方向與像素間距（affine），所以在程式裡比例、方向都正確
    4. 同時把原始 CBCT、CT 也轉成 HU 一起輸出，三者可以用相同 window 對照

輸出（每位病人一個資料夾）：
    nifti：<病人>_sCT.nii.gz / _CT.nii.gz / _CBCT.nii.gz   → ITK-SNAP、3D Slicer、MRIcroGL…
    dicom：sCT/ CT/ CBCT/ 三個 series 資料夾（同一個 study）→ RadiAnt、MicroDicom、Horos、OsiriX…

用法：
    python export_sct.py \
        --npy_dir ./results/dataset2/npy_exports \
        --data /work/jhz21803/dataset2_npy \
        --src /work/jhz21803/<Dataset 2 的 3D 資料夾> \
        --out ./results/dataset2/export \
        --patients 2BA028 2BA035 --format both
"""
import os
import sys
import glob
import argparse
import datetime

import numpy as np
import nibabel as nib

HU_MIN, HU_MAX, BACKGROUND_HU, BODY_THRESHOLD = -150.0, 150.0, -1000.0, -0.5


def unnormalize_to_hu(arr):
    """與 evaluate_tf.py / evaluate3D.py 相同"""
    out = np.full_like(arr, BACKGROUND_HU, dtype=np.float32)
    body = arr > BODY_THRESHOLD
    out[body] = np.clip(arr[body] * 300.0 - 150.0, HU_MIN, HU_MAX)
    return out


def slice_indices(data_dir, pid):
    """切片檔名 <病人>_<z>.npy 中的 z（排序方式與 evaluate_tf.py 相同）"""
    names = sorted(f for f in os.listdir(os.path.join(data_dir, 'trainB'))
                   if f.endswith('.npy') and f.split('_')[0] == pid)
    return [int(os.path.splitext(n)[0].split('_')[1]) for n in names]


def save_nifti(vol, ref, path):
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    img = nib.Nifti1Image(vol.astype(np.float32), ref.affine, hdr)
    img.set_qform(ref.affine, code=1); img.set_sform(ref.affine, code=1)
    nib.save(img, path)


def save_dicom(vol, affine, folder, pid, desc, series_no, study_uid, frame_uid):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage

    os.makedirs(folder, exist_ok=True)
    lps = np.diag([-1, -1, 1])                       # NIfTI 為 RAS，DICOM 為 LPS
    col_dir = lps @ affine[:3, 0]; sx = np.linalg.norm(col_dir)   # 沿 i（影像的「欄」方向）
    row_dir = lps @ affine[:3, 1]; sy = np.linalg.norm(row_dir)   # 沿 j（影像的「列」方向）
    sz = np.linalg.norm(affine[:3, 2])
    series_uid = generate_uid()
    now = datetime.datetime.now()
    data = np.clip(np.rint(vol), -32768, 32767).astype(np.int16)

    for k in range(vol.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.Modality = 'CT'
        ds.PatientName = ds.PatientID = pid
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.StudyDate = ds.SeriesDate = now.strftime('%Y%m%d')
        ds.StudyTime = ds.SeriesTime = now.strftime('%H%M%S')
        ds.StudyDescription = 'CBCT-to-CT CycleGAN'
        ds.SeriesDescription = desc
        ds.SeriesNumber, ds.InstanceNumber = series_no, k + 1
        ds.ImageOrientationPatient = [float(v) for v in np.r_[col_dir / sx, row_dir / sy]]
        ds.ImagePositionPatient = [float(v) for v in lps @ (affine @ [0, 0, k, 1])[:3]]
        ds.SliceThickness = float(sz)
        ds.PixelSpacing = [float(sy), float(sx)]      # [列與列的間距, 欄與欄的間距]
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = 0, 1, 'HU'
        ds.WindowCenter, ds.WindowWidth = 0, 300                       # 預設 window：-150 ~ 150 HU
        ds.PixelData = np.ascontiguousarray(data[:, :, k].T).tobytes()  # (i, j) → (列 = j, 欄 = i)
        ds.save_as(os.path.join(folder, f'{desc}_{k + 1:04d}.dcm'), enforce_file_format=True)


def main():
    ap = argparse.ArgumentParser(description='把 sCT 匯出成 NIfTI / DICOM')
    ap.add_argument('--npy_dir', required=True, help='evaluate_tf.py --save_npy 的輸出資料夾（npy_exports）')
    ap.add_argument('--data', required=True, help='切片資料夾（內含 trainA/ trainB/），用來找每張切片原本的 z')
    ap.add_argument('--src', required=True, help='原始 3D 資料集根目錄（底下是 <病人>/cbct.nii.gz、ct.nii.gz）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--patients', nargs='*', default=None, help='預設為 npy_dir 中所有病人')
    ap.add_argument('--format', choices=['nifti', 'dicom', 'both'], default='nifti')
    args = ap.parse_args()

    found = sorted(os.path.basename(f)[:-len('_sCT.npy')] for f in glob.glob(os.path.join(args.npy_dir, '*_sCT.npy')))
    if not found:
        sys.exit(f"在 {args.npy_dir} 找不到 *_sCT.npy。請先用 evaluate_tf.py 加 --save_npy 跑一次（不要加 --resize_eval）。")
    if args.format != 'nifti':
        try:
            import pydicom  # noqa: F401
        except ImportError:
            sys.exit("輸出 DICOM 需要 pydicom：pip install pydicom")

    for pid in args.patients or found:
        if pid not in found:
            print(f"⚠️ 找不到 {pid}_sCT.npy，略過"); continue
        ref_ct = nib.load(os.path.join(args.src, pid, 'ct.nii.gz'))
        shape = ref_ct.shape[:3]
        sct = np.load(os.path.join(args.npy_dir, f'{pid}_sCT.npy'))
        zs = slice_indices(args.data, pid)
        if len(zs) != len(sct) or sct.shape[1:] != shape[:2]:
            print(f"⚠️ {pid}：sCT {sct.shape} 與原始體積 {shape}／{len(zs)} 張切片對不起來"
                  f"（是否用了 --resize_eval？），略過"); continue

        vol_sct = np.full(shape, BACKGROUND_HU, np.float32)
        for k, z in enumerate(zs):
            vol_sct[:, :, z] = sct[k]
        vol_ct = unnormalize_to_hu(np.asanyarray(ref_ct.dataobj).astype(np.float32))
        vol_cbct = unnormalize_to_hu(np.asanyarray(
            nib.load(os.path.join(args.src, pid, 'cbct.nii.gz')).dataobj).astype(np.float32))
        vols = [('sCT', vol_sct), ('CT', vol_ct), ('CBCT', vol_cbct)]

        d = os.path.join(args.out, pid); os.makedirs(d, exist_ok=True)
        if args.format in ('nifti', 'both'):
            for name, v in vols:
                save_nifti(v, ref_ct, os.path.join(d, f'{pid}_{name}.nii.gz'))
        if args.format in ('dicom', 'both'):
            from pydicom.uid import generate_uid
            study_uid, frame_uid = generate_uid(), generate_uid()
            for no, (name, v) in enumerate(vols, 1):
                save_dicom(v, ref_ct.affine, os.path.join(d, 'dicom', name), pid, name, no, study_uid, frame_uid)
        zooms = tuple(round(float(x), 3) for x in ref_ct.header.get_zooms()[:3])
        print(f"{pid}：體積 {shape}，間距 {zooms} mm，{len(zs)} 張有 sCT → {d}")

    print(f"完成。下載 {args.out} 後用醫學影像程式開啟，window 建議 center 0 / width 300（-150 ~ 150 HU）。")


if __name__ == '__main__':
    main()
