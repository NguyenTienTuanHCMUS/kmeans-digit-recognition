"""
Tinh chỉnh cách 2 (K-means học đặc trưng) mà KHÔNG nhìn vào tập test.

60.000 ảnh train được chia làm hai: 50.000 ảnh để học, 10.000 ảnh làm tập kiểm định (validation).
Mỗi cấu hình (kích thước mảnh, số nét K, lưới gộp) được học trên 50.000 ảnh rồi chấm trên tập
kiểm định, với nhiều mức điều chuẩn C của hồi quy logistic. Tập test chỉ được dùng một lần
ở cuối, bằng train_features.py với cấu hình tốt nhất.

Chạy:  python tune_features.py
       python tune_features.py --configs 6:400:2,6:400:3 --C 0.1,1
       python tune_features.py --gpu      (chạy trên GPU NVIDIA, cần CuPy)
       (mỗi cấu hình viết dạng kích_thước_mảnh:K:lưới_gộp)
"""
import argparse
import contextlib
import io
import os
import time
import warnings

import numpy as np

from train_features import accuracy, encode, fit_classifier, fit_zca, normalize, random_patches, to_gpu
from train_kmeans import DATA_DIR, kmeans, load_images, load_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="6:400:2,6:400:3")
    ap.add_argument("--C", default="0.03,0.1,0.3,1")
    ap.add_argument("--patches", type=int, default=100_000)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*(converge|max_iter).*")

    X = load_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte"))
    y = load_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"))
    perm = np.random.default_rng(123).permutation(len(X))
    tr, va = perm[:50_000], perm[50_000:]
    Cs = [float(c) for c in args.C.split(",")]

    print(f"{'mảnh':>4} {'K':>4} {'lưới':>4} {'số đặc trưng':>12} | " + " | ".join(f"C={c:<5g}" for c in Cs) + " | thời gian")
    for cfg in args.configs.split(","):
        p, K, pool = (int(v) for v in cfg.split(":"))
        t0 = time.time()
        rng = np.random.default_rng(0)
        patches = normalize(random_patches(X[tr], args.patches, rng, p=p))
        zca_mean, zca_W = fit_zca(patches)
        with contextlib.redirect_stdout(io.StringIO()):
            centroids, _ = kmeans((patches - zca_mean) @ zca_W, K, max_iters=100, seed=0)
        F = encode(X, centroids, zca_mean, zca_W, p=p, pool=pool, gpu=args.gpu)
        mu, sd = F[tr].mean(axis=0), F[tr].std(axis=0) + 1e-6
        F = (F - mu) / sd
        accs = []
        F_tr = to_gpu(F[tr]) if args.gpu else F[tr]   # chép lên GPU một lần, dùng cho mọi C
        for C in Cs:
            coef, intercept = fit_classifier(F_tr, y[tr], C=C, gpu=args.gpu)
            accs.append(accuracy(F[va], y[va], coef, intercept))
        del F_tr
        print(f"{p:>4} {K:>4} {pool:>4} {F.shape[1]:>12} | " + " | ".join(f"{a * 100:6.2f}%" for a in accs)
              + f" | {time.time() - t0:.0f}s", flush=True)
        del F


if __name__ == "__main__":
    main()
