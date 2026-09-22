"""
개선 2: Dense Moving Mask 생성 (v2)
sparse LiDAR mask → dilation + event density 교차로 dense mask 생성

실행:
    python precompute_moving_masks_v2.py \
        --dataset_root /VoxFormer/dataset/semantickitti \
        --seqs 00 01 02 03 04 05 06 07 09 10 08 \
        --workers 4
"""

import os, glob, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import binary_dilation
from tqdm import tqdm

MOVING_IDS   = {252, 253, 254, 255, 257, 258, 259}
IMG_H, IMG_W = 370, 1220
DILATION_PX  = 5
EV_PERCENTILE = 40


def read_calib(path):
    calib = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                calib[k.strip()] = np.array(list(map(float, v.strip().split())))
    P2 = calib['P2'].reshape(3, 4)
    Tr = np.eye(4); Tr[:3, :] = calib['Tr'].reshape(3, 4)
    return P2, Tr


def project_sparse(velodyne, label, P2, Tr, H=IMG_H, W=IMG_W):
    pts = np.fromfile(velodyne, dtype=np.float32).reshape(-1, 4)
    sem = np.fromfile(label, dtype=np.uint32) & 0xFFFF
    xyz = pts[:, :3]; front = xyz[:, 0] > 0
    xyz, sem = xyz[front], sem[front]
    hom = np.hstack([xyz, np.ones((len(xyz), 1))])
    cam = (Tr @ hom.T).T[:, :3]
    cf  = cam[:, 2] > 0; cam, sem = cam[cf], sem[cf]
    hom2 = np.hstack([cam, np.ones((len(cam), 1))])
    uv = (P2 @ hom2.T).T
    u  = (uv[:, 0] / uv[:, 2]).astype(int)
    v  = (uv[:, 1] / uv[:, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (cam[:, 2] < 80)
    mask = np.full((H, W), -1, dtype=np.int8)
    for i in np.where(ok)[0]:
        lbl = int(sem[i])
        if mask[v[i], u[i]] == -1:
            mask[v[i], u[i]] = 1 if lbl in MOVING_IDS else 0
        elif mask[v[i], u[i]] == 0 and lbl in MOVING_IDS:
            mask[v[i], u[i]] = 1
    return mask


def load_ev_density(ev_root, seq, fid, H=IMG_H, W=IMG_W,
                    ev_H=352, ev_W=1216):
    for p in [
        os.path.join(ev_root, "sequences", seq, f"{fid}.npy"),
        os.path.join(ev_root, seq, "image_0", f"{fid}.npy"),
    ]:
        if os.path.exists(p):
            d = np.load(p)
            if d.ndim == 3:
                density = np.abs(d).sum(0).astype(np.float32)
            elif d.ndim == 2 and d.shape[1] == 4:
                x = d[:,0].astype(int); y = d[:,1].astype(int)
                ok = (x>=0)&(x<ev_W)&(y>=0)&(y<ev_H)
                density = np.zeros((ev_H, ev_W), dtype=np.float32)
                np.add.at(density, (y[ok], x[ok]), 1.)
            else:
                return None
            if density.shape != (H, W):
                from PIL import Image as PILImage
                density = np.array(
                    PILImage.fromarray(density).resize((W, H), PILImage.BILINEAR)
                )
            return density
    return None


def make_dense(sparse, ev_density, dilation=DILATION_PX, pct=EV_PERCENTILE):
    moving = sparse == 1
    struct  = np.ones((2*dilation+1, 2*dilation+1), dtype=bool)
    dilated = binary_dilation(moving, structure=struct)
    if ev_density is not None:
        nz = ev_density[ev_density > 0]
        ev_mask = ev_density > np.percentile(nz, pct) if len(nz) else np.zeros_like(moving)
    else:
        ev_mask = np.zeros_like(moving)
    dense = np.zeros_like(sparse, dtype=np.int8)
    dense[sparse == 0] = 0
    dense[dilated & ev_mask & (sparse != 0)] = 1
    dense[moving] = 1
    return dense


def proc(args):
    vp, lp, cp, ev_root, seq, fid, out = args
    try:
        if os.path.exists(out):
            return 'skipped'
        P2, Tr = read_calib(cp)
        sp = project_sparse(vp, lp, P2, Tr)
        ev = load_ev_density(ev_root, seq, fid)
        dm = make_dense(sp, ev)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, dm)
        return 'ok'
    except Exception as e:
        return f'err:{e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', default='/VoxFormer/dataset/semantickitti')
    ap.add_argument('--seqs', nargs='+',
                    default=['00','01','02','03','04','05','06','07','09','10','08'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    ev_root = os.path.join(args.dataset_root, 'events')
    tasks = []
    for seq in args.seqs:
        vd = os.path.join(args.dataset_root, 'sequences', seq, 'velodyne')
        lbs = []
        for pat in [
            os.path.join(args.dataset_root, 'lidarseg', seq, 'labels', '*.label'),
            os.path.join(args.dataset_root, 'sequences', seq, 'labels', '*.label'),
        ]:
            lbs = sorted(glob.glob(pat))
            if lbs: break
        if not lbs:
            print(f"[SKIP] seq {seq}"); continue
        cp  = os.path.join(args.dataset_root, 'sequences', seq, 'calib.txt')
        out = os.path.join(args.dataset_root, 'moving_masks_v2', seq)
        for lp in lbs:
            fid = os.path.splitext(os.path.basename(lp))[0]
            vp  = os.path.join(vd, fid + '.bin')
            op  = os.path.join(out, fid + '.npy')
            if os.path.exists(vp):
                tasks.append((vp, lp, cp, ev_root, seq, fid, op))

    print(f"총 {len(tasks)}개 처리...")
    ok = sk = er = 0
    with ProcessPoolExecutor(args.workers) as ex:
        pbar = tqdm(total=len(tasks))
        for r in as_completed([ex.submit(proc, t) for t in tasks]):
            s = r.result()
            if s == 'ok': ok += 1
            elif s == 'skipped': sk += 1
            else: er += 1
            pbar.update(1)
        pbar.close()
    print(f"완료: ok={ok}, skipped={sk}, errors={er}")


if __name__ == '__main__':
    main()