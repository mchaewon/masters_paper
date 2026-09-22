"""
Event Voxel SUM / DIFF 시각화
==============================
사용법:
    python visualize_event_sum_diff.py \
        --event_path /data/knuvi/moon/SSC/semantickitti/events/00/image_0/000252.npy \
        --out_dir ./event_vis \
        --num_bins 5

    # 또는 디렉토리 전체
    python visualize_event_sum_diff.py \
        --event_dir /data/knuvi/moon/SSC/semantickitti/events/00/image_0 \
        --out_dir ./event_vis \
        --max_files 10
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


def load_event_voxel(path, num_bins=5, H=352, W=1216):
    """
    .npy 파일 로드 → (T, H, W) float32 voxel
    지원 형식:
      (T, H, W): 이미 voxel grid
      (N, 4)   : raw events [x, y, t, p]
    """
    data = np.load(path)

    if data.ndim == 3:
        # 이미 voxel grid
        return data.astype(np.float32)

    elif data.ndim == 2 and data.shape[1] == 4:
        # raw events → voxel 변환
        voxel = np.zeros((num_bins, H, W), dtype=np.float32)
        x = data[:, 0].astype(np.int32)
        y = data[:, 1].astype(np.int32)
        t = data[:, 2].astype(np.float64)
        p = data[:, 3].astype(np.float32)

        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x, y, t, p = x[ok], y[ok], t[ok], p[ok]

        if len(t) == 0:
            return voxel

        tn  = (t - t.min()) / (t.max() - t.min() + 1e-8) * (num_bins - 1)
        pol = np.where(p > 0, 1., -1.).astype(np.float32)
        tf  = tn.astype(np.int32)
        tc  = tf + 1
        wc  = (tn - tf).astype(np.float32)
        wf  = 1. - wc

        mf = tf < num_bins
        mc = tc < num_bins
        if mf.sum():
            np.add.at(voxel, (tf[mf], y[mf], x[mf]), pol[mf] * wf[mf])
        if mc.sum():
            np.add.at(voxel, (tc[mc], y[mc], x[mc]), pol[mc] * wc[mc])
        return voxel

    else:
        raise ValueError(f"Unsupported shape: {data.shape}")


def visualize(event_path, out_dir, num_bins=5, H=352, W=1216):
    """단일 파일 시각화 → PNG 저장"""
    os.makedirs(out_dir, exist_ok=True)

    # 로드
    voxel = load_event_voxel(event_path, num_bins, H, W)  # (T, H, W)
    T = voxel.shape[0]

    # ── 계산 ────────────────────────────────────────────────
    ev_sum  = voxel.sum(axis=0)                        # (H, W)
    diffs   = voxel[1:] - voxel[:-1]                   # (T-1, H, W)
    ev_diff = np.abs(diffs).sum(axis=0)                # (H, W) motion magnitude

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor='#111120')
    fname = os.path.splitext(os.path.basename(event_path))[0]
    fig.suptitle(f'Event Temporal Decomposition  |  {fname}',
                 color='white', fontsize=13, fontweight='bold', y=0.97)

    nrows = 3
    gs = gridspec.GridSpec(nrows, T,
                           figure=fig,
                           hspace=0.30, wspace=0.05,
                           left=0.05, right=0.98,
                           top=0.92, bottom=0.04)

    # ── Row 0: individual bins ───────────────────────────────
    for t in range(T):
        ax = fig.add_subplot(gs[0, t])
        vmax = max(np.abs(voxel[t]).max(), 1e-3) * 0.8
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ax.imshow(voxel[t], cmap='RdBu_r', norm=norm,
                  aspect='auto', interpolation='nearest')
        ax.set_title(f'Bin {t}', color='#aaaacc', fontsize=9, pad=2)
        ax.axis('off')

    fig.text(0.01, 0.78, 'Bins', color='#aaaacc',
             fontsize=9, va='center', ha='center',
             rotation=90, fontweight='bold')

    # ── Row 1: SUM ──────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, :])
    vmax_s = max(np.abs(ev_sum).max(), 1e-3) * 0.85
    norm_s = TwoSlopeNorm(vmin=-vmax_s, vcenter=0, vmax=vmax_s)
    im_s   = ax_sum.imshow(ev_sum, cmap='RdBu_r', norm=norm_s,
                           aspect='auto', interpolation='nearest')
    ax_sum.set_title(
        'SUM  (E_ego)  =  ego-motion depth signal  '
        '[red=positive, blue=negative events]',
        color='#7ecfcf', fontsize=10, pad=4, loc='left'
    )
    ax_sum.axis('off')
    cb = plt.colorbar(im_s, ax=ax_sum, fraction=0.008, pad=0.005)
    cb.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)

    fig.text(0.01, 0.50, 'SUM', color='#7ecfcf',
             fontsize=9, va='center', ha='center',
             fontweight='bold')

    # ── Row 2: DIFF magnitude ───────────────────────────────
    ax_diff = fig.add_subplot(gs[2, :])
    im_d    = ax_diff.imshow(ev_diff, cmap='hot',
                             aspect='auto', interpolation='nearest')
    ax_diff.set_title(
        'DIFF magnitude  (E_obj)  =  object motion signal  '
        '[bright = large inter-bin change = moving object boundary]',
        color='#ffaa77', fontsize=10, pad=4, loc='left'
    )
    ax_diff.axis('off')
    cb2 = plt.colorbar(im_d, ax=ax_diff, fraction=0.008, pad=0.005)
    cb2.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)

    fig.text(0.01, 0.20, 'DIFF\nMAG', color='#ffaa77',
             fontsize=9, va='center', ha='center',
             fontweight='bold')

    # ── 저장 ────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f'{fname}_sum_diff.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#111120')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_path', type=str, default=None,
                        help='단일 .npy 파일 경로')
    parser.add_argument('--event_dir',  type=str, default=None,
                        help='디렉토리 경로 (모든 .npy 처리)')
    parser.add_argument('--out_dir',    type=str, default='./event_vis')
    parser.add_argument('--num_bins',   type=int, default=5)
    parser.add_argument('--H',          type=int, default=352)
    parser.add_argument('--W',          type=int, default=1216)
    parser.add_argument('--max_files',  type=int, default=5,
                        help='디렉토리 처리 시 최대 파일 수')
    parser.add_argument('--stride',     type=int, default=1,
                        help='디렉토리 처리 시 파일 간격 (e.g. 10이면 10장마다 1장)')
    args = parser.parse_args()

    if args.event_path:
        visualize(args.event_path, args.out_dir, args.num_bins, args.H, args.W)

    elif args.event_dir:
        files = sorted([
            os.path.join(args.event_dir, f)
            for f in os.listdir(args.event_dir)
            if f.endswith('.npy')
        ])
        files = files[::args.stride][:args.max_files]
        print(f'Processing {len(files)} files...')
        for f in files:
            try:
                visualize(f, args.out_dir, args.num_bins, args.H, args.W)
            except Exception as e:
                print(f'  ERROR {f}: {e}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()