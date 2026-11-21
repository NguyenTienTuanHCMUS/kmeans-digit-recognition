"""
Thí nghiệm: dùng K-means để HỌC ĐẶC TRƯNG (theo Coates & Ng, 2011), rồi phân loại.

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
       python train_features.py --k 50
Cần thêm scikit-learn cho bước 4:  pip install scikit-learn
"""
import argparse
import os
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from train_kmeans import DATA_DIR, kmeans, load_images, load_labels, squared_distances

HERE = os.path.dirname(os.path.abspath(__file__))
P = 6                       # kích thước mảnh
G = 28 - P + 1              # 23 vị trí mỗi chiều
NORM_EPS = 10 / 255 ** 2    # giống Coates & Ng (+10 trên thang 0..255)
ZCA_EPS = 0.1


def normalize(patches):
    """Trừ độ sáng trung bình, chia độ lệch chuẩn của từng mảnh."""
    p = patches - patches.mean(axis=1, keepdims=True)
    return p / np.sqrt(p.var(axis=1, keepdims=True) + NORM_EPS)


def fit_zca(patches):
    mean = patches.mean(axis=0)
    cov = np.cov(patches - mean, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(1.0 / np.sqrt(vals + ZCA_EPS)) @ vecs.T
    return mean.astype(np.float32), W.astype(np.float32)


def random_patches(X, n, rng):
    """Cắt n mảnh ngẫu nhiên, bỏ các mảnh gần như trống (nền đen) để từ điển học nét."""
    out = []
    while sum(len(o) for o in out) < n:
        idx = rng.integers(len(X), size=n)
        r, c = rng.integers(G, size=n), rng.integers(G, size=n)
        imgs = X[idx].reshape(-1, 28, 28)
        p = np.stack([imgs[i, r[i]:r[i] + P, c[i]:c[i] + P].reshape(-1) for i in range(n)])
        out.append(p[p.var(axis=1) > 0.01])
    return np.concatenate(out)[:n]


def encode(X, centroids, zca_mean, zca_W, batch=500):
    """Biến mỗi ảnh thành vector 4K số (mức xuất hiện của từng nét trong 4 góc phần tư)."""
    K = len(centroids)
    feats = np.empty((len(X), 4 * K), dtype=np.float32)
    h = (G + 1) // 2  # 12: chia 23 hàng/cột thành 12 + 11
    for s in range(0, len(X), batch):
        imgs = X[s:s + batch].reshape(-1, 28, 28)
        B = len(imgs)
        patches = sliding_window_view(imgs, (P, P), axis=(1, 2)).reshape(B * G * G, P * P)
        patches = (normalize(patches) - zca_mean) @ zca_W
        z = np.sqrt(squared_distances(patches, centroids))
        f = np.maximum(0.0, z.mean(axis=1, keepdims=True) - z).reshape(B, G, G, K)
        feats[s:s + B] = np.concatenate([
            f[:, :h, :h].sum(axis=(1, 2)), f[:, :h, h:].sum(axis=(1, 2)),
            f[:, h:, :h].sum(axis=(1, 2)), f[:, h:, h:].sum(axis=(1, 2)),
        ], axis=1)
    return feats


def save_dictionary_image(centroids, zca_W, path, cols=20, scale=8):
    """Vẽ từ điển nét ra ảnh PNG để xem K-means đã học được những nét gì."""
    from PIL import Image
    K = len(centroids)
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
    ap.add_argument("--patches", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    X_train = load_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte"))
    y_train = load_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"))
    X_test = load_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"))
    y_test = load_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"))

    t0 = time.time()
    patches = normalize(random_patches(X_train, args.patches, rng))
    zca_mean, zca_W = fit_zca(patches)
    patches = (patches - zca_mean) @ zca_W
    print(f"1. Đã cắt và làm trắng {len(patches):,} mảnh {P}x{P}")

    print(f"2. K-means trên các mảnh, K = {args.k}")
    centroids, _ = kmeans(patches, args.k, max_iters=100, seed=args.seed)
    t1 = time.time()

    print("3. Mô tả ảnh bằng từ điển nét ...")
    F_train = encode(X_train, centroids, zca_mean, zca_W)
    F_test = encode(X_test, centroids, zca_mean, zca_W)
    mu, sd = F_train.mean(axis=0), F_train.std(axis=0) + 1e-6
    F_train, F_test = (F_train - mu) / sd, (F_test - mu) / sd
    t2 = time.time()
    print(f"   mỗi ảnh -> vector {F_train.shape[1]} số")

    print("4. Huấn luyện hồi quy logistic ...")
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(F_train, y_train)
    t3 = time.time()

    train_acc = clf.score(F_train, y_train)
    test_acc = clf.score(F_test, y_test)
    print(f"\nThời gian: học từ điển {t1 - t0:.0f}s, mô tả ảnh {t2 - t1:.0f}s, phân loại {t3 - t2:.0f}s")
    print(f"Độ chính xác train: {train_acc * 100:.2f}%")
    print(f"Độ chính xác test : {test_acc * 100:.2f}%")

    np.savez_compressed(os.path.join(HERE, f"features_k{args.k}.npz"),
                        centroids=centroids, zca_mean=zca_mean, zca_W=zca_W, feat_mean=mu, feat_std=sd,
                        coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
                        test_acc=test_acc)
    img = os.path.join(HERE, f"dictionary_k{args.k}.png")
    save_dictionary_image(centroids, zca_W, img)
    print(f"Đã lưu mô hình features_k{args.k}.npz và ảnh từ điển {os.path.basename(img)}")


if __name__ == "__main__":
    main()
