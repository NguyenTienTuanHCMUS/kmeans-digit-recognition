"""
Chương trình 1: Học chữ số viết tay MNIST bằng K-means clustering.

Các bước:
  1. Đọc dữ liệu MNIST (định dạng IDX) trong thư mục MNIST/raw.
  2. Chạy K-means (tự cài đặt bằng numpy, khởi tạo k-means++) trên ảnh train
     -> K-means KHÔNG dùng nhãn, chỉ gom các ảnh giống nhau thành cụm.
  3. Gán nhãn cho từng cluster: lấy nhãn xuất hiện nhiều nhất trong cụm (majority vote).
  4. Đánh giá độ chính xác trên tập test (gán mỗi ảnh vào centroid gần nhất).
  5. Lưu centroid + nhãn của cluster vào file clusters.npz.

Chạy:  python train_kmeans.py            (mặc định K = 200)
       python train_kmeans.py --k 400    (nhiều cluster hơn -> chính xác hơn)
"""
import argparse
import gzip
import os
import time

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MNIST", "raw")


# ----------------------------------------------------------------------------
# Đọc dữ liệu MNIST
# ----------------------------------------------------------------------------
def open_idx(path):
    """Mở file IDX; nếu không có bản giải nén thì đọc thẳng file .gz."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rb")
    return open(path, "rb")


def load_images(path):
    with open_idx(path) as f:
        magic, n, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        assert magic == 2051, "Sai định dạng file ảnh IDX"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0  # pixel trong [0, 1]


def load_labels(path):
    with open_idx(path) as f:
        magic, n = np.frombuffer(f.read(8), dtype=">i4")
        assert magic == 2049, "Sai định dạng file nhãn IDX"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


# ----------------------------------------------------------------------------
# K-means
# ----------------------------------------------------------------------------
def squared_distances(X, C, x_sq=None):
    """Khoảng cách Euclid bình phương giữa mọi điểm X (n,d) và mọi centroid C (k,d).
    Dùng công thức ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 để tính nhanh bằng nhân ma trận."""
    if x_sq is None:
        x_sq = (X ** 2).sum(axis=1)
    c_sq = (C ** 2).sum(axis=1)
    d = x_sq[:, None] - 2.0 * (X @ C.T) + c_sq[None, :]
    return np.maximum(d, 0.0)


def kmeans_plus_plus(X, k, rng, x_sq):
    """Khởi tạo k-means++: chọn centroid tiếp theo với xác suất tỉ lệ với
    khoảng cách bình phương tới centroid gần nhất đã chọn."""
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
    centroids[0] = X[rng.integers(n)]
    closest = squared_distances(X, centroids[:1], x_sq)[:, 0]
    for i in range(1, k):
        probs = closest / closest.sum()
        centroids[i] = X[rng.choice(n, p=probs)]
        closest = np.minimum(closest, squared_distances(X, centroids[i:i + 1], x_sq)[:, 0])
    return centroids


def kmeans(X, k, max_iters=100, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    x_sq = (X ** 2).sum(axis=1)

    print(f"Khởi tạo {k} centroid bằng k-means++ ...")
    C = kmeans_plus_plus(X, k, rng, x_sq)

    prev_inertia = None
    for it in range(1, max_iters + 1):
        # Bước E: gán mỗi điểm vào centroid gần nhất
        D = squared_distances(X, C, x_sq)
        assign = D.argmin(axis=1)
        inertia = D[np.arange(len(X)), assign].sum()

        # Bước M: cập nhật centroid = trung bình các điểm trong cụm
        counts = np.bincount(assign, minlength=k)
        sums = np.zeros_like(C)
        np.add.at(sums, assign, X)
        nonempty = counts > 0
        C[nonempty] = sums[nonempty] / counts[nonempty, None]

        # Cụm rỗng -> đặt lại vào điểm đang xa centroid của nó nhất
        empty = np.flatnonzero(~nonempty)
        if len(empty):
            far = np.argsort(D[np.arange(len(X)), assign])[::-1][:len(empty)]
            C[empty] = X[far]

        change = None if prev_inertia is None else (prev_inertia - inertia) / prev_inertia
        print(f"  vòng {it:3d}: inertia = {inertia:,.1f}"
              + ("" if change is None else f"  (giảm {change * 100:.3f}%)"))
        if change is not None and 0 <= change < tol:
            print("Hội tụ.")
            break
        prev_inertia = inertia

    assign = squared_distances(X, C, x_sq).argmin(axis=1)
    return C, assign


# ----------------------------------------------------------------------------
# Gán nhãn cho cluster và đánh giá
# ----------------------------------------------------------------------------
def label_clusters(assign, y, k):
    """Mỗi cluster nhận nhãn của chữ số xuất hiện nhiều nhất trong nó."""
    counts = np.zeros((k, 10), dtype=np.int64)  # counts[cụm, chữ số]
    np.add.at(counts, (assign, y), 1)
    labels = counts.argmax(axis=1)
    purity = counts.max(axis=1) / np.maximum(counts.sum(axis=1), 1)
    return labels, purity, counts


def predict(X, centroids, cluster_labels):
    return cluster_labels[squared_distances(X, centroids).argmin(axis=1)]


def main():
    ap = argparse.ArgumentParser(description="Học MNIST bằng K-means")
    ap.add_argument("--k", type=int, default=200, help="số cluster (mặc định 200)")
    ap.add_argument("--iters", type=int, default=100, help="số vòng lặp tối đa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="clusters.npz", help="file lưu kết quả")
    args = ap.parse_args()

    print("Đang đọc dữ liệu MNIST ...")
    X_train = load_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte"))
    y_train = load_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"))
    X_test = load_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"))
    y_test = load_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"))
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    t0 = time.time()
    centroids, assign = kmeans(X_train, args.k, args.iters, seed=args.seed)
    print(f"K-means xong sau {time.time() - t0:.1f}s")

    labels, purity, counts = label_clusters(assign, y_train, args.k)
    print("\nSố cluster được gán cho mỗi chữ số:")
    for d in range(10):
        print(f"  chữ số {d}: {np.sum(labels == d)} cluster")
    print(f"Độ thuần trung bình của cluster: {purity.mean() * 100:.1f}%")

    train_acc = (labels[assign] == y_train).mean()
    test_acc = (predict(X_test, centroids, labels) == y_test).mean()
    print(f"\nĐộ chính xác train: {train_acc * 100:.2f}%")
    print(f"Độ chính xác test : {test_acc * 100:.2f}%")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    np.savez_compressed(out, centroids=centroids, labels=labels,
                        purity=purity, sizes=counts.sum(axis=1), test_acc=test_acc)
    print(f"\nĐã lưu {args.k} cluster vào {out}")


if __name__ == "__main__":
    main()
