"""
사전 전처리: Moving Object Mask 생성
======================================
lidarseg LiDAR label을 이미지 평면으로 투영하여
픽셀별 moving/static 2D mask를 생성한다.

실행:
    python precompute_moving_masks.py \
        --dataset_root /VoxFormer/dataset/semantickitti \
        --seqs 00 01 02 03 04 05 06 07 09 10 \
        --workers 4
"""

import os, glob, argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

MOVING_IDS = {252, 253, 254, 255, 257, 258, 259}
IMG_H, IMG_W = 370, 1220   # VoxFormer 입력 해상도

TRAIN_SEQS = ['00','01','02','03','04','05','06','07','09','10']
VAL_SEQS   = ['08']


def read_calib(calib_path):
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                calib[k.strip()] = np.array(list(map(float, v.strip().split())))
    P2 = calib['P2'].reshape(3, 4)
    Tr = np.eye(4)
    Tr[:3, :] = calib['Tr'].reshape(3, 4)
    return P2, Tr


def project_moving_mask(velodyne_path, label_path, P2, Tr,
                         H=IMG_H, W=IMG_W):
    """
    LiDAR 포인트 클라우드 → 이미지 평면 moving mask
    Returns: (H, W) int8
        1  = moving object
        0  = static
       -1  = no LiDAR point (unknown)
    """
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(label_path,    dtype=np.uint32) & 0xFFFF

    # 카메라 앞 포인트만
    xyz = points[:, :3]
    front = xyz[:, 0] > 0
    xyz   = xyz[front]
    sem   = labels[front]

    # LiDAR → 카메라
    xyz_hom = np.hstack([xyz, np.ones((len(xyz), 1))])
    xyz_cam = (Tr @ xyz_hom.T).T[:, :3]
    cf = xyz_cam[:, 2] > 0
    xyz_cam = xyz_cam[cf]
    sem     = sem[cf]

    # 이미지 투영
    xyz_hom2 = np.hstack([xyz_cam, np.ones((len(xyz_cam), 1))])
    uv = (P2 @ xyz_hom2.T).T
    u  = (uv[:, 0] / uv[:, 2]).astype(int)
    v  = (uv[:, 1] / uv[:, 2]).astype(int)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (xyz_cam[:, 2] < 80)

    mask = np.full((H, W), -1, dtype=np.int8)
    for idx in np.where(valid)[0]:
        vi, ui = v[idx], u[idx]
        label  = int(sem[idx])
        if mask[vi, ui] == -1:
            mask[vi, ui] = 1 if label in MOVING_IDS else 0
        elif mask[vi, ui] == 0 and label in MOVING_IDS:
            mask[vi, ui] = 1   # moving이 우선

    return mask


def process_frame(args):
    velodyne_path, label_path, calib_path, out_path = args
    try:
        if os.path.exists(out_path):
            return out_path, 'skipped'
        P2, Tr = read_calib(calib_path)
        mask   = project_moving_mask(velodyne_path, label_path, P2, Tr)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mask)
        return out_path, 'ok'
    except Exception as e:
        return out_path, f'error: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/VoxFormer/dataset/semantickitti')
    parser.add_argument('--seqs', nargs='+', default=TRAIN_SEQS + VAL_SEQS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    print("="*55)
    print("Moving Mask 전처리")
    print("="*55)

    tasks = []
    for seq in args.seqs:
        velodyne_dir = os.path.join(
            args.dataset_root, 'sequences', seq, 'velodyne'
        )
        # lidarseg label 경로 (두 가지 패턴)
        label_patterns = [
            os.path.join(args.dataset_root, 'lidarseg', seq, 'labels', '*.label'),
            os.path.join(args.dataset_root, 'sequences', seq, 'labels', '*.label'),
        ]
        label_files = []
        for pat in label_patterns:
            label_files = sorted(glob.glob(pat))
            if label_files: break

        calib_path = os.path.join(
            args.dataset_root, 'sequences', seq, 'calib.txt'
        )
        out_dir = os.path.join(
            args.dataset_root, 'moving_masks', seq
        )

        if not label_files:
            print(f"[WARN] Seq {seq}: label 없음, 스킵")
            continue

        print(f"  Seq {seq}: {len(label_files)}개")
        for lp in label_files:
            fid = os.path.splitext(os.path.basename(lp))[0]
            vp  = os.path.join(velodyne_dir, fid + '.bin')
            op  = os.path.join(out_dir, fid + '.npy')
            if os.path.exists(vp):
                tasks.append((vp, lp, calib_path, op))

    print(f"\n총 {len(tasks)}개 프레임 처리 시작...")
    ok = skip = err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_frame, t): t for t in tasks}
        pbar = tqdm(total=len(futures), unit='frame')
        for f in as_completed(futures):
            _, status = f.result()
            if status == 'ok':     ok   += 1
            elif status == 'skipped': skip += 1
            else:                  err  += 1
            pbar.update(1)
        pbar.close()

    print(f"\n완료: ok={ok}, skipped={skip}, errors={err}")
    print(f"저장 위치: {args.dataset_root}/moving_masks/")


if __name__ == '__main__':
    main()