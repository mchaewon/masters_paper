"""
SSC 결과 시각화 스크립트
========================
pkl로 저장된 test 결과를 BEV, 3D point cloud, 클래스별 IoU 그래프로 시각화.

실행:
    python visualize_ssc_results.py \
        --results results/exp1_scratch/test_results.pkl \
        --exp_name exp1_scratch \
        --output_dir ./vis_results
"""

import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

# SemanticKITTI 클래스 정의
CLASS_NAMES = [
    "empty", "car", "bicycle", "motorcycle", "truck",
    "other-vehicle", "person", "bicyclist", "motorcyclist", "road",
    "parking", "sidewalk", "other-ground", "building", "fence",
    "vegetation", "trunk", "terrain", "pole", "traffic-sign"
]

# 클래스별 색상 (RGB, 0~1)
CLASS_COLORS = np.array([
    [0,   0,   0  ],  # empty        (black)
    [100, 150, 245],  # car          (blue)
    [59,  130, 246],  # bicycle      (light blue)
    [245, 230, 100],  # motorcycle   (yellow)
    [138, 43,  226],  # truck        (purple)
    [200, 100, 100],  # other-vehicle(pink)
    [220, 20,  60 ],  # person       (red)
    [255, 127, 80 ],  # bicyclist    (coral)
    [255, 140, 0  ],  # motorcyclist (orange)
    [128, 64,  255],  # road         (violet)
    [75,  0,   75 ],  # parking      (dark purple)
    [75,  0,   175],  # sidewalk     (indigo)
    [128, 128, 0  ],  # other-ground (olive)
    [175, 0,   75 ],  # building     (dark red)
    [75,  75,  75 ],  # fence        (gray)
    [0,   175, 0  ],  # vegetation   (green)
    [135, 60,  0  ],  # trunk        (brown)
    [150, 240, 80 ],  # terrain      (lime)
    [255, 240, 150],  # pole         (light yellow)
    [255, 0,   0  ],  # traffic-sign (bright red)
], dtype=np.float32) / 255.0


def load_results(pkl_path):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    print(f"로드 완료: {len(results)}개 샘플")
    print(f"결과 형식: {type(results[0])}")
    if isinstance(results[0], dict):
        print(f"키: {list(results[0].keys())}")
    return results


def compute_iou_per_class(results, n_classes=20, ignore_label=255):
    """전체 결과에서 클래스별 IoU 계산"""
    total_tp = np.zeros(n_classes)
    total_fp = np.zeros(n_classes)
    total_fn = np.zeros(n_classes)

    for r in results:
        if isinstance(r, dict):
            pred = r['y_pred'].flatten()
            true = r['y_true'].flatten()
        else:
            continue

        mask = true != ignore_label
        pred = pred[mask]
        true = true[mask]

        for c in range(n_classes):
            tp = ((pred == c) & (true == c)).sum()
            fp = ((pred == c) & (true != c)).sum()
            fn = ((pred != c) & (true == c)).sum()
            total_tp[c] += tp
            total_fp[c] += fp
            total_fn[c] += fn

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    return iou


def plot_class_iou_comparison(iou_dict, output_dir, filename="class_iou_comparison.png"):
    """
    여러 실험의 클래스별 IoU를 한 그래프에 비교.
    iou_dict: {'exp_name': iou_array, ...}
    """
    n_classes = len(CLASS_NAMES)
    x = np.arange(n_classes)
    n_exps = len(iou_dict)
    width = 0.8 / n_exps

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    fig.suptitle('클래스별 IoU 비교', fontsize=16, fontweight='bold')

    colors = ['steelblue', 'tomato', 'seagreen', 'orange', 'purple']

    # 상단: 전체 클래스 막대그래프
    ax = axes[0]
    for i, (name, iou) in enumerate(iou_dict.items()):
        offset = (i - n_exps/2 + 0.5) * width
        bars = ax.bar(x + offset, iou * 100, width,
                      label=f"{name} (mIoU={iou[1:].mean()*100:.2f}%)",
                      color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('IoU (%)')
    ax.set_title('전체 클래스별 IoU')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # 하단: static vs dynamic 그룹 비교
    static_cls  = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # road~traffic-sign
    dynamic_cls = [1, 2, 3, 4, 5, 6, 7, 8]                      # car~motorcyclist

    ax2 = axes[1]
    group_labels = ['Static\n(평균)', 'Dynamic\n(평균)'] + \
                   [CLASS_NAMES[c] for c in dynamic_cls]
    group_x = np.arange(len(group_labels))

    for i, (name, iou) in enumerate(iou_dict.items()):
        vals = [
            iou[static_cls].mean() * 100,
            iou[dynamic_cls].mean() * 100,
        ] + [iou[c] * 100 for c in dynamic_cls]
        offset = (i - n_exps/2 + 0.5) * width
        ax2.bar(group_x + offset, vals, width,
                label=name, color=colors[i % len(colors)], alpha=0.8)

    ax2.set_xticks(group_x)
    ax2.set_xticklabels(group_labels, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('IoU (%)')
    ax2.set_title('Static/Dynamic 그룹 및 동적 클래스 세부 비교')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[저장] {save_path}")
    plt.show()


def visualize_bev(result, output_dir, sample_idx=0, exp_name=""):
    """
    단일 샘플의 BEV(Bird's Eye View) 시각화.
    X-Y 평면에서 Z축을 따라 max pooling.
    """
    if isinstance(result, dict):
        pred = result['y_pred']  # (X, Y, Z)
        true = result['y_true']  # (X, Y, Z)
    else:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'{exp_name} - Sample {sample_idx} BEV 시각화', fontsize=13)

    for ax, data, title in zip(axes,
                                [true, pred],
                                ['Ground Truth', f'Prediction ({exp_name})']):
        # Z축 max pooling (가장 높은 semantic label 선택, empty 제외)
        bev = np.zeros(data.shape[:2], dtype=np.int32)
        for z in range(data.shape[2] - 1, -1, -1):
            layer = data[:, :, z]
            mask = (layer != 0) & (layer != 255)
            bev[mask] = layer[mask]

        # 컬러맵으로 변환
        H, W = bev.shape
        img = np.ones((H, W, 3), dtype=np.float32)
        for c in range(len(CLASS_NAMES)):
            img[bev == c] = CLASS_COLORS[c]
        img[bev == 255] = [0.5, 0.5, 0.5]  # ignore → gray

        ax.imshow(img, origin='lower')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Y (lateral)')
        ax.set_ylabel('X (forward)')
        ax.axis('off')

    # 범례
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(1, len(CLASS_NAMES))]
    fig.legend(handles=patches, loc='lower center', ncol=10,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))

    save_path = os.path.join(output_dir, f"{exp_name}_bev_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[저장] {save_path}")
    plt.show()


def print_summary_table(iou_dict):
    """수치 요약 테이블 출력"""
    static_cls  = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    dynamic_cls = [1, 2, 3, 4, 5, 6, 7, 8]

    print("\n" + "="*70)
    print(f"{'실험':<20} {'mIoU':>8} {'Static':>8} {'Dynamic':>8} {'IoU(geo)':>10}")
    print("="*70)
    for name, iou in iou_dict.items():
        miou    = iou[1:].mean() * 100   # empty 제외
        static  = iou[static_cls].mean() * 100
        dynamic = iou[dynamic_cls].mean() * 100
        print(f"  {name:<18} {miou:>7.2f}% {static:>7.2f}% {dynamic:>7.2f}%")
    print("="*70)

    print("\n클래스별 상세 (IoU %):")
    header = f"{'클래스':<15}" + "".join(f"{n:>12}" for n in iou_dict.keys())
    print(header)
    print("-" * len(header))
    for c in range(1, len(CLASS_NAMES)):
        row = f"  {CLASS_NAMES[c]:<13}"
        for iou in iou_dict.values():
            row += f"{iou[c]*100:>12.2f}"
        print(row)


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+',
                        help='pkl 결과 파일 경로들 (공백으로 구분)')
    parser.add_argument('--exp_names', nargs='+',
                        help='실험 이름들 (결과 파일과 순서 맞춤)')
    parser.add_argument('--output_dir', default='./vis_results')
    parser.add_argument('--n_bev_samples', type=int, default=5,
                        help='BEV 시각화할 샘플 수')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.results:
        print("사용법: python visualize_ssc_results.py \\")
        print("    --results results/exp1/test_results.pkl results/exp2/test_results.pkl \\")
        print("    --exp_names exp1_scratch exp2_scratch \\")
        print("    --output_dir ./vis_results")
        return

    exp_names = args.exp_names if args.exp_names else \
                [f"exp{i+1}" for i in range(len(args.results))]

    # 결과 로드 및 IoU 계산
    iou_dict = {}
    all_results = {}
    for path, name in zip(args.results, exp_names):
        print(f"\n로드 중: {name} ({path})")
        results = load_results(path)
        iou = compute_iou_per_class(results)
        iou_dict[name] = iou
        all_results[name] = results
        print(f"  mIoU (empty 제외): {iou[1:].mean()*100:.2f}%")

    # 수치 요약 출력
    print_summary_table(iou_dict)

    # 클래스별 IoU 비교 그래프
    plot_class_iou_comparison(iou_dict, args.output_dir)

    # BEV 시각화 (첫 번째 실험 기준으로 샘플 선택)
    first_name = exp_names[0]
    for i in range(min(args.n_bev_samples, len(all_results[first_name]))):
        for name, results in all_results.items():
            if i < len(results):
                visualize_bev(results[i], args.output_dir,
                              sample_idx=i, exp_name=name)

    print(f"\n✅ 시각화 완료: {args.output_dir}")


if __name__ == "__main__":
    main()