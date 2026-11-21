"""
Cách 2: dùng K-means để HỌC ĐẶC TRƯNG (theo Coates, Lee & Ng, 2011), rồi phân loại.

Khác với train_kmeans.py (K-means trên cả ảnh, mỗi centroid là một mẫu chữ số),
ở đây K-means chạy trên các MẢNH NHỎ 6x6, nên mỗi centroid là một NÉT CƠ BẢN.

Các bước:
  1. Cắt ngẫu nhiên các mảnh 6x6 từ ảnh train, chuẩn hóa độ sáng/độ tương phản, làm trắng (ZCA).
  2. K-means trên các mảnh -> "từ điển nét" gồm K centroid.
  3. Mô tả mỗi ảnh: trượt cửa sổ 6x6 qua 23x23 vị trí, đo mức giống từng nét
     (mã hóa "tam giác": max(0, trung bình khoảng cách - khoảng cách tới nét k)),
     rồi cộng dồn theo 4 góc phần tư -> vector 4K số.
  4. Huấn luyện hồi quy logistic trên các vector đó (đây là bước dùng nhãn).

Chạy:  python train_features.py            (K = 200)
       python train_features.py --k 400 --pool 3 --C 0.03
       python train_features.py --k 400 --gpu  (bước 3 và 4 chạy trên GPU NVIDIA, cần CuPy)
Bước 4 trên CPU cần scikit-learn:  pip install scikit-learn
"""
import argparse
import os
import time

import numpy as np

from train_kmeans import DATA_DIR, array_module, kmeans, load_images, load_labels, squared_distances

HERE = os.path.dirname(os.path.abspath(__file__))
P = 6                       # kích thước mảnh
G = 28 - P + 1              # 23 vị trí mỗi chiều
NORM_EPS = 10 / 255 ** 2    # giống Coates & Ng (+10 trên thang 0..255)
ZCA_EPS = 0.1


def get_xp(gpu):
    """numpy để chạy trên CPU, cupy để chạy trên GPU."""
    if not gpu:
        return np
    import cupy
    return cupy


def to_numpy(a):
    return a.get() if hasattr(a, "get") else a


def normalize(patches):
    """Trừ độ sáng trung bình, chia độ lệch chuẩn của từng mảnh."""
    xp = array_module(patches)
    p = patches - patches.mean(axis=1, keepdims=True)
    return p / xp.sqrt(p.var(axis=1, keepdims=True) + NORM_EPS)


def fit_zca(patches):
    mean = patches.mean(axis=0)
    cov = np.cov(patches - mean, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(1.0 / np.sqrt(vals + ZCA_EPS)) @ vecs.T
    return mean.astype(np.float32), W.astype(np.float32)


def random_patches(X, n, rng, p=P):
    """Cắt n mảnh p x p ngẫu nhiên, bỏ các mảnh gần như trống (nền đen) để từ điển học nét."""
    g = 28 - p + 1
    out = []
    while sum(len(o) for o in out) < n:
        idx = rng.integers(len(X), size=n)
        r, c = rng.integers(g, size=n), rng.integers(g, size=n)
        imgs = X[idx].reshape(-1, 28, 28)
        ps = np.stack([imgs[i, r[i]:r[i] + p, c[i]:c[i] + p].reshape(-1) for i in range(n)])
        out.append(ps[ps.var(axis=1) > 0.01])
    return np.concatenate(out)[:n]


def pool_edges(g, cells):
    """Ranh giới các ô gộp khi chia g vị trí thành `cells` phần (giống np.array_split).
    Ví dụ g = 23, cells = 2 -> [0, 12, 23]; cells = 3 -> [0, 8, 16, 23]."""
    return np.concatenate([[0], np.cumsum([len(a) for a in np.array_split(np.arange(g), cells)])])


def patch_index(p):
    """Chỉ số pixel của mọi mảnh p x p trong ảnh 28x28 phẳng, dạng (số vị trí, p*p).
    Vị trí đi theo hàng rồi cột, trong mỗi mảnh cũng theo hàng rồi cột."""
    g = 28 - p + 1
    pos = (np.arange(g)[:, None] * 28 + np.arange(g)[None, :]).reshape(-1)
    off = (np.arange(p)[:, None] * 28 + np.arange(p)[None, :]).reshape(-1)
    return pos[:, None] + off[None, :]


def encode(X, centroids, zca_mean, zca_W, batch=None, p=P, pool=2, gpu=False):
    """Biến mỗi ảnh thành vector pool*pool*K số: mức xuất hiện của từng nét trong từng ô gộp.
    Mặc định p = 6, pool = 2 (4 góc phần tư). gpu=True thì tính trên GPU bằng CuPy."""
    xp = get_xp(gpu)
    batch = batch or (1000 if gpu else 500)
    K, g = len(centroids), 28 - p + 1
    e = pool_edges(g, pool)
    idx = xp.asarray(patch_index(p))
    C, zm, zW = xp.asarray(centroids), xp.asarray(zca_mean), xp.asarray(zca_W)
    feats = np.empty((len(X), pool * pool * K), dtype=np.float32)
    for s in range(0, len(X), batch):
        imgs = xp.asarray(X[s:s + batch].reshape(-1, 28 * 28))
        B = len(imgs)
        patches = imgs[:, idx].reshape(B * g * g, p * p)
        patches = (normalize(patches) - zm) @ zW
        z = xp.sqrt(squared_distances(patches, C))
        f = xp.maximum(0.0, z.mean(axis=1, keepdims=True) - z).reshape(B, g, g, K)
        feats[s:s + B] = to_numpy(xp.concatenate([f[:, e[i]:e[i + 1], e[j]:e[j + 1]].sum(axis=(1, 2))
                                                  for i in range(pool) for j in range(pool)], axis=1))
    return feats


def fit_classifier(F, y, C=1.0, gpu=False, max_iter=1000):
    """Hồi quy logistic nhiều lớp, cực tiểu  C * tổng log-loss + ½ ||W||²  (hệ số chặn không bị phạt).
    Trên CPU dùng scikit-learn. Trên GPU tự tính hàm mất mát và đạo hàm bằng CuPy, rồi để
    L-BFGS của SciPy (cùng thuật toán scikit-learn dùng) tìm cực tiểu.
    Trả về (coef 10 x d, intercept 10)."""
    if not gpu:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=C, max_iter=max_iter).fit(F, y)
        return clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)

    import cupy as cp
    from scipy.optimize import minimize
    Fg = to_gpu(F)
    n, d = Fg.shape
    Y = cp.eye(10, dtype=cp.float32)[cp.asarray(y)]

    def loss_and_grad(theta):
        t = cp.asarray(theta, dtype=cp.float32)
        W, b = t[:d * 10].reshape(d, 10), t[d * 10:]
        Z = Fg @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        E = cp.exp(Z)
        S = E.sum(axis=1, keepdims=True)
        nll = (cp.log(S[:, 0]) - (Z * Y).sum(axis=1)).astype(cp.float64).sum()
        loss = C * float(nll) + 0.5 * float((W.astype(cp.float64) ** 2).sum())
        G = E / S - Y
        grad = cp.concatenate([(C * (Fg.T @ G) + W).ravel(), C * G.sum(axis=0)])
        return loss, cp.asnumpy(grad).astype(np.float64)

    res = minimize(loss_and_grad, np.zeros(d * 10 + 10), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "gtol": 1e-4})
    W, b = res.x[:d * 10].reshape(d, 10), res.x[d * 10:]
    del Fg, Y
    cp.get_default_memory_pool().free_all_blocks()
    return W.T.astype(np.float32), b.astype(np.float32)


def to_gpu(F, chunk=5000):
    """Chép ma trận đặc trưng lên GPU theo từng khúc (tránh xin một vùng RAM ghim quá lớn).
    Nếu F đã nằm trên GPU thì giữ nguyên."""
    import cupy as cp
    if isinstance(F, cp.ndarray):
        return F
    out = cp.empty(F.shape, dtype=cp.float32)
    for s in range(0, len(F), chunk):
        out[s:s + chunk] = cp.asarray(np.ascontiguousarray(F[s:s + chunk], dtype=np.float32))
    return out


def accuracy(F, y, coef, intercept):
    return float(((F @ coef.T + intercept).argmax(axis=1) == y).mean())


def save_dictionary_image(centroids, zca_W, path, cols=20, scale=8):
    """Vẽ từ điển nét ra ảnh PNG để xem K-means đã học được những nét gì."""
    from PIL import Image
    K = len(centroids)
    P = int(round(centroids.shape[1] ** 0.5))
    rows = (K + cols - 1) // cols
    tile = P * scale + 4
    sheet = Image.new("L", (cols * tile, rows * tile), 255)
    for k, c in enumerate(centroids):
        v = c - c.min()
        v = (v / (v.max() + 1e-8) * 255).astype(np.uint8).reshape(P, P)
        img = Image.fromarray(v).resize((P * scale, P * scale), Image.NEAREST)
        sheet.paste(img, ((k % cols) * tile + 2, (k // cols) * tile + 2))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description="K-means học đặc trưng trên MNIST")
    ap.add_argument("--k", type=int, default=200, help="số nét trong từ điển")
    ap.add_argument("--patch", type=int, default=P, help="kích thước mảnh (mặc định 6)")
    ap.add_argument("--pool", type=int, default=2, help="lưới gộp pool x pool (mặc định 2: 4 góc phần tư)")
    ap.add_argument("--C", type=float, default=1.0, help="mức điều chuẩn của hồi quy logistic (nhỏ = mạnh hơn)")
    ap.add_argument("--patches", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="tên file mô hình (mặc định features_k<K>.npz)")
    ap.add_argument("--gpu", action="store_true", help="chạy bước 3 và 4 trên GPU (cần CuPy)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    p, pool = args.patch, args.pool

    X_train = load_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte"))
    y_train = load_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"))
    X_test = load_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"))
    y_test = load_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"))

    t0 = time.time()
    patches = normalize(random_patches(X_train, args.patches, rng, p=p))
    zca_mean, zca_W = fit_zca(patches)
    patches = (patches - zca_mean) @ zca_W
    print(f"1. Đã cắt và làm trắng {len(patches):,} mảnh {p}x{p}")

    print(f"2. K-means trên các mảnh, K = {args.k}")
    centroids, _ = kmeans(patches, args.k, max_iters=100, seed=args.seed)
    t1 = time.time()

    print(f"3. Mô tả ảnh bằng từ điển nét (lưới gộp {pool}x{pool}) ...")
    F_train = encode(X_train, centroids, zca_mean, zca_W, p=p, pool=pool, gpu=args.gpu)
    F_test = encode(X_test, centroids, zca_mean, zca_W, p=p, pool=pool, gpu=args.gpu)
    mu, sd = F_train.mean(axis=0), F_train.std(axis=0) + 1e-6
    F_train, F_test = (F_train - mu) / sd, (F_test - mu) / sd
    t2 = time.time()
    print(f"   mỗi ảnh -> vector {F_train.shape[1]} số")

    print(f"4. Huấn luyện hồi quy logistic (C = {args.C:g}, {'GPU' if args.gpu else 'CPU'}) ...")
    coef, intercept = fit_classifier(F_train, y_train, C=args.C, gpu=args.gpu)
    t3 = time.time()

    train_acc = accuracy(F_train, y_train, coef, intercept)
    test_acc = accuracy(F_test, y_test, coef, intercept)
    print(f"\nThời gian: học từ điển {t1 - t0:.0f}s, mô tả ảnh {t2 - t1:.0f}s, phân loại {t3 - t2:.0f}s")
    print(f"Độ chính xác train: {train_acc * 100:.2f}%")
    print(f"Độ chính xác test : {test_acc * 100:.2f}%")

    name = args.out or f"features_k{args.k}.npz"
    np.savez_compressed(os.path.join(HERE, name),
                        centroids=centroids, zca_mean=zca_mean, zca_W=zca_W, feat_mean=mu, feat_std=sd,
                        coef=coef, intercept=intercept,
                        test_acc=test_acc, patch=p, pool=pool)
    img = os.path.join(HERE, os.path.splitext(name)[0].replace("features", "dictionary") + ".png")
    save_dictionary_image(centroids, zca_W, img)
    print(f"Đã lưu mô hình {name} và ảnh từ điển {os.path.basename(img)}")


if __name__ == "__main__":
    main()
