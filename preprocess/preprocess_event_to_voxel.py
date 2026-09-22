"""
작업 1: SemanticKITTI-E Raw Events → Voxel Grid 전처리 스크립트
=============================================================
전체 시퀀스의 raw event npy 파일을 voxel grid로 변환하여 저장한다.

실행:
    python preprocess_to_voxel.py

    # 특정 시퀀스만 처리:
    python preprocess_to_voxel.py --seqs 00 01 04

    # 병렬 처리 (CPU 코어 수 지정):
    python preprocess_to_voxel.py --workers 8
"""

import os
import glob
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ============================================================
# 경로 설정 (수정 필요)
# ============================================================
EVENT_ROOT  = "./data/events"         # raw events npy 루트
OUTPUT_ROOT = "./data/events_voxel"   # 변환된 voxel grid 저장 위치

# SemanticKITTI 학습/검증 시퀀스
TRAIN_SEQS = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQS   = ['08']
ALL_SEQS   = TRAIN_SEQS + VAL_SEQS

# Voxel grid 파라미터
NUM_BINS = 5      # 시간 빈 수 (EvSSC와 동일하게 맞춤)
IMG_H    = 352    # SemanticKITTI-E 해상도 (확인된 y max+1)
IMG_W    = 1216   # SemanticKITTI-E 해상도 (확인된 x max+1)


# ============================================================
# 핵심 변환 함수
# ============================================================
def raw_events_to_voxel_grid(events: np.ndarray,
                              num_bins: int = NUM_BINS,
                              H: int = IMG_H,
                              W: int = IMG_W) -> np.ndarray:
    """
    Raw Events (N, 4) → Voxel Grid (B, H, W)

    Zhu et al. 2019 "Unsupervised Event-based Optical Flow"의
    선형 보간 방법을 사용한다. 각 이벤트를 인접한 두 bin에
    타임스탬프 거리에 비례하여 분배한다.

    Args:
        events:   (N, 4) int64/float array  [x, y, t, p]
        num_bins: 시간 빈 수
        H, W:     출력 해상도

    Returns:
        voxel_grid: (num_bins, H, W) float32
    """
    voxel_grid = np.zeros((num_bins, H, W), dtype=np.float32)

    if len(events) == 0:
        return voxel_grid

    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    t = events[:, 2].astype(np.float64)
    p = events[:, 3].astype(np.float32)

    # 유효 픽셀 범위 필터링
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]

    if len(x) == 0:
        return voxel_grid

    # 타임스탬프 → [0, num_bins-1] 정규화
    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        t_norm = np.zeros_like(t)
    else:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1)

    # 극성을 +1 / -1로 통일
    pol = np.where(p > 0, 1.0, -1.0).astype(np.float32)

    # 선형 보간: 각 이벤트를 floor/ceil bin에 가중치로 분배
    t_floor = t_norm.astype(np.int32)
    t_ceil  = t_floor + 1
    w_ceil  = (t_norm - t_floor).astype(np.float32)
    w_floor = 1.0 - w_ceil

    # Floor bin 누적
    mask_f = t_floor < num_bins
    np.add.at(voxel_grid,
               (t_floor[mask_f], y[mask_f], x[mask_f]),
               pol[mask_f] * w_floor[mask_f])

    # Ceil bin 누적
    mask_c = t_ceil < num_bins
    np.add.at(voxel_grid,
               (t_ceil[mask_c], y[mask_c], x[mask_c]),
               pol[mask_c] * w_ceil[mask_c])

    return voxel_grid


def normalize_voxel(voxel: np.ndarray) -> np.ndarray:
    """
    Voxel grid를 [-1, 1] 범위로 정규화한다.
    Non-zero 픽셀들의 표준편차 기반 정규화 (robust normalization).
    """
    nonzero = voxel[voxel != 0]
    if len(nonzero) == 0:
        return voxel
    std = nonzero.std()
    if std < 1e-8:
        return voxel
    voxel = voxel / (3.0 * std)   # 3-sigma clipping
    voxel = np.clip(voxel, -1.0, 1.0)
    return voxel


# ============================================================
# 단일 파일 처리 함수 (병렬 처리용)
# ============================================================
def process_single_file(args):
    """
    단일 raw event npy → voxel grid npy 변환.
    (병렬 처리를 위해 인자를 튜플로 받음)
    """
    src_path, dst_path, num_bins, H, W, normalize = args

    try:
        # 이미 변환된 파일은 스킵 (재처리 방지)
        if os.path.exists(dst_path):
            return dst_path, "skipped", 0

        # Raw events 로드
        events = np.load(src_path)

        # 포맷 검증
        if events.ndim != 2 or events.shape[1] != 4:
            return src_path, "error", f"예상치 못한 shape: {events.shape}"

        # Voxel grid 변환
        voxel = raw_events_to_voxel_grid(events, num_bins=num_bins, H=H, W=W)

        # 정규화 (선택)
        if normalize:
            voxel = normalize_voxel(voxel)

        # 저장
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        np.save(dst_path, voxel)

        nonzero_ratio = (voxel != 0).mean() * 100
        return dst_path, "ok", nonzero_ratio

    except Exception as e:
        return src_path, "error", str(e)


# ============================================================
# 시퀀스 처리 함수
# ============================================================
def get_file_pairs(event_root, output_root, sequences):
    """
    변환할 (src_path, dst_path) 쌍 목록을 생성한다.
    디렉토리 구조를 자동으로 탐지한다.
    """
    pairs = []

    for seq in sequences:
        # 가능한 입력 경로 패턴
        patterns = [
            os.path.join(event_root, "sequences", seq, "events", "*.npy"),
            os.path.join(event_root,  seq, "image_0", "*.npy"),
            os.path.join(event_root, seq, "image_0","*.npy"),
        ]

        src_files = []
        for pat in patterns:
            src_files = sorted(glob.glob(pat))
            if src_files:
                # 입력 디렉토리 구조 파악
                src_dir = os.path.dirname(src_files[0])
                break

        if not src_files:
            print(f"[WARN] 시퀀스 {seq}: npy 파일 없음, 스킵")
            continue

        print(f"  시퀀스 {seq}: {len(src_files)}개 파일 발견")

        # 출력 경로 설정 (동일한 구조로 저장)
        dst_dir = os.path.join(output_root,  seq, "image_0")

        for src in src_files:
            fname = os.path.basename(src)
            dst = os.path.join(dst_dir, fname)
            pairs.append((src, dst))

    return pairs


# ============================================================
# 전처리 통계 출력
# ============================================================
def print_stats(results, elapsed):
    ok_count      = sum(1 for _, status, _ in results if status == "ok")
    skip_count    = sum(1 for _, status, _ in results if status == "skipped")
    error_count   = sum(1 for _, status, _ in results if status == "error")
    total         = len(results)

    ok_ratios = [info for _, status, info in results
                 if status == "ok" and isinstance(info, (int, float))]

    print("\n" + "=" * 60)
    print("전처리 완료 통계")
    print("=" * 60)
    print(f"전체 파일:    {total:,}")
    print(f"변환 완료:    {ok_count:,}")
    print(f"스킵 (기존):  {skip_count:,}")
    print(f"오류:         {error_count:,}")
    print(f"소요 시간:    {elapsed:.1f}초")
    if ok_count > 0:
        print(f"처리 속도:    {ok_count / elapsed:.1f} 파일/초")
    if ok_ratios:
        print(f"평균 non-zero 비율: {np.mean(ok_ratios):.2f}%")

    # 오류 파일 목록 출력
    errors = [(p, info) for p, status, info in results if status == "error"]
    if errors:
        print(f"\n오류 파일 목록:")
        for p, info in errors[:10]:
            print(f"  {p}: {info}")


# ============================================================
# 변환 결과 샘플 검증
# ============================================================
def verify_conversion(output_root, sequences, num_samples=3):
    """
    변환된 voxel grid 파일 몇 개를 샘플링하여 검증한다.
    """
    print("\n" + "=" * 60)
    print("변환 결과 검증")
    print("=" * 60)

    for seq in sequences[:2]:  # 처음 2개 시퀀스만 확인
        dst_dir = os.path.join(output_root,  seq, "image_0")
        files = sorted(glob.glob(os.path.join(dst_dir, "*.npy")))
        if not files:
            continue

        print(f"\n시퀀스 {seq}: {len(files)}개 voxel 파일")
        sample_files = files[:num_samples]

        for f in sample_files:
            v = np.load(f)
            print(f"  {os.path.basename(f)}: "
                  f"shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min():.3f}, max={v.max():.3f}, "
                  f"non-zero={( v != 0).mean()*100:.1f}%")


# ============================================================
# 메인 실행
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='SemanticKITTI-E 전처리 스크립트')
    parser.add_argument('--event_root',  default=EVENT_ROOT)
    parser.add_argument('--output_root', default=OUTPUT_ROOT)
    parser.add_argument('--seqs',        nargs='+', default=ALL_SEQS,
                        help='처리할 시퀀스 번호 (예: 00 01 08)')
    parser.add_argument('--num_bins',    type=int, default=NUM_BINS)
    parser.add_argument('--height',      type=int, default=IMG_H)
    parser.add_argument('--width',       type=int, default=IMG_W)
    parser.add_argument('--workers',     type=int, default=4,
                        help='병렬 처리 프로세스 수')
    parser.add_argument('--normalize',   action='store_true', default=False,
                        help='Voxel grid 정규화 여부')
    parser.add_argument('--verify_only', action='store_true', default=False,
                        help='변환 없이 결과 검증만 수행')
    args = parser.parse_args()

    print("=" * 60)
    print("SemanticKITTI-E 전처리: Raw Events → Voxel Grid")
    print("=" * 60)
    print(f"입력 경로:  {args.event_root}")
    print(f"출력 경로:  {args.output_root}")
    print(f"시퀀스:     {args.seqs}")
    print(f"Bins:       {args.num_bins}")
    print(f"해상도:     {args.height} x {args.width}")
    print(f"Workers:    {args.workers}")
    print(f"정규화:     {args.normalize}")

    if args.verify_only:
        verify_conversion(args.output_root, args.seqs)
        return

    # 파일 쌍 수집
    print("\n파일 목록 수집 중...")
    pairs = get_file_pairs(args.event_root, args.output_root, args.seqs)
    print(f"총 {len(pairs):,}개 파일 처리 예정")

    if len(pairs) == 0:
        print("[ERROR] 처리할 파일이 없습니다. 경로를 확인하세요.")
        return

    # 처리 인자 패키징
    task_args = [
        (src, dst, args.num_bins, args.height, args.width, args.normalize)
        for src, dst in pairs
    ]

    # 병렬 처리
    print(f"\n변환 시작 ({args.workers}개 프로세스)...")
    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, a): a for a in task_args}
        pbar = tqdm(total=len(futures), desc="변환 중", unit="파일")

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # 오류 발생 시 즉시 출력
            if result[1] == "error":
                tqdm.write(f"[ERROR] {result[0]}: {result[2]}")

            pbar.update(1)
        pbar.close()

    elapsed = time.time() - start
    print_stats(results, elapsed)

    # 변환 결과 검증
    verify_conversion(args.output_root, args.seqs)

    print(f"\n✅ 전처리 완료!")
    print(f"출력 위치: {args.output_root}")


if __name__ == "__main__":
    main()