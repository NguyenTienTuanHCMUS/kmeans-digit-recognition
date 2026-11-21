"""
Chương trình 2: Vẽ chữ số trên bảng 28x28 và nhận diện bằng đặc trưng học từ K-means.

Mô hình (features_k400.npz) do train_features.py tạo ra:
  - K-means đã học một "từ điển" gồm các nét cơ bản 6x6,
  - mỗi hình vẽ được mô tả bằng mức xuất hiện của từng nét ở 4 góc phần tư,
  - hồi quy logistic đọc mô tả đó và cho xác suất của từng chữ số.

Chạy:  python digit_app.py
       python digit_app.py --model features_k200.npz

Cách dùng: giữ chuột trái để vẽ, chuột phải để tẩy, nút "Xóa" để vẽ lại.
Kết quả được cập nhật mỗi khi nhả chuột.
"""
import argparse
import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image

from numpy.lib.stride_tricks import sliding_window_view

from train_features import P, encode, normalize
from train_kmeans import squared_distances

HERE = os.path.dirname(os.path.abspath(__file__))

GRID = 28          # bảng vẽ 28x28 ô, đúng kích thước ảnh MNIST
CELL = 16          # mỗi ô hiển thị 16x16 pixel trên màn hình
BRUSH = 1.5        # bán kính cọ (tính theo ô)
PREVIEW = 140      # cạnh (pixel) của các ảnh xem trước


def preprocess(grid):
    """Chuẩn hóa hình vẽ giống cách MNIST được tạo:
    cắt sát nét vẽ, thu nhỏ vào khung 20x20, rồi đặt sao cho trọng tâm nằm giữa ảnh 28x28."""
    ys, xs = np.nonzero(grid > 0.05)
    if len(ys) == 0:
        return None
    crop = grid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    small = np.asarray(img, dtype=np.float32) / 255.0

    out = np.zeros((GRID, GRID), dtype=np.float32)
    top, left = (GRID - nh) // 2, (GRID - nw) // 2
    out[top:top + nh, left:left + nw] = small

    # dịch để trọng tâm (center of mass) về giữa ảnh
    total = out.sum()
    rows, cols = np.indices(out.shape)
    cy, cx = (rows * out).sum() / total, (cols * out).sum() / total
    dy = int(np.clip(round(GRID / 2 - cy), -top, GRID - top - nh))
    dx = int(np.clip(round(GRID / 2 - cx), -left, GRID - left - nw))
    return np.roll(out, (dy, dx), axis=(0, 1))


class FeatureModel:
    """Từ điển nét K-means + hồi quy logistic, đọc từ file .npz của train_features.py."""

    def __init__(self, path):
        d = np.load(path)
        self.centroids = d["centroids"]
        self.zca_mean, self.zca_W = d["zca_mean"], d["zca_W"]
        self.mu, self.sd = d["feat_mean"], d["feat_std"]
        self.coef, self.intercept = d["coef"], d["intercept"]
        self.test_acc = float(d["test_acc"])
        self.K = len(self.centroids)

    def predict(self, img):
        """Trả về (xác suất 10 chữ số, số vị trí có mực mà mỗi nét là nét gần nhất)."""
        raw = encode(img.reshape(1, -1), self.centroids, self.zca_mean, self.zca_W)[0]
        logits = self.coef @ ((raw - self.mu) / self.sd) + self.intercept
        p = np.exp(logits - logits.max())
        return p / p.sum(), self.stroke_counts(img)

    def stroke_counts(self, img):
        """Chỉ để hiển thị: ở mỗi mảnh 6x6 có mực, nét nào là nét gần nhất."""
        patches = sliding_window_view(img.reshape(28, 28), (P, P)).reshape(-1, P * P)
        patches = patches[patches.var(axis=1) > 0.01]
        if len(patches) == 0:
            return np.zeros(self.K, dtype=int)
        white = (normalize(patches) - self.zca_mean) @ self.zca_W
        nearest = squared_distances(white, self.centroids).argmin(axis=1)
        return np.bincount(nearest, minlength=self.K)

    def stroke_image(self, k):
        """Ảnh 6x6 của nét k, co giãn về 0..1 để hiển thị."""
        c = self.centroids[k]
        return ((c - c.min()) / (c.max() - c.min() + 1e-8)).reshape(P, P)


class DigitApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.grid = np.zeros((GRID, GRID), dtype=np.float32)
        self.last = None

        root.title(f"Nhận diện chữ số: K-means học {model.K} nét cơ bản "
                   f"({model.test_acc * 100:.2f}% trên tập test)")
        root.resizable(False, False)

        # ---- Bảng vẽ ----
        left = tk.Frame(root, padx=12, pady=12)
        left.pack(side=tk.LEFT)
        tk.Label(left, text="Bảng vẽ 28 x 28", font=("Segoe UI", 12, "bold")).pack()
        size = GRID * CELL
        self.canvas = tk.Canvas(left, width=size, height=size, bg="black",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.pack(pady=6)
        self.rects = [[self.canvas.create_rectangle(c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL,
                                                    fill="#000000", outline="#1a1a1a")
                       for c in range(GRID)] for r in range(GRID)]
        btns = tk.Frame(left)
        btns.pack(fill=tk.X)
        tk.Button(btns, text="Xóa", width=12, command=self.clear).pack(side=tk.LEFT)
        tk.Label(btns, text="Chuột trái: vẽ   Chuột phải: tẩy", fg="#666").pack(side=tk.RIGHT)

        for ev in ("<Button-1>", "<B1-Motion>"):
            self.canvas.bind(ev, lambda e: self.paint(e, erase=False))
        for ev in ("<Button-3>", "<B3-Motion>"):
            self.canvas.bind(ev, lambda e: self.paint(e, erase=True))
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonRelease-3>", self.on_release)

        # ---- Kết quả ----
        right = tk.Frame(root, padx=12, pady=12)
        right.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(right, text="Dự đoán", font=("Segoe UI", 12, "bold")).pack()
        self.result = tk.Label(right, text="?", font=("Segoe UI", 72, "bold"), fg="#1f6feb", width=3)
        self.result.pack()
        self.confidence = tk.Label(right, text="", font=("Segoe UI", 10), fg="#555")
        self.confidence.pack()

        previews = tk.Frame(right)
        previews.pack(pady=6)
        self.prev_input = self._make_preview(previews, "Ảnh sau chuẩn hóa", 0, GRID)
        self.prev_stroke = self._make_preview(previews, "Nét được dùng nhiều nhất", 1, P)

        tk.Label(right, text="Xác suất từng chữ số", font=("Segoe UI", 10, "bold")).pack(pady=(10, 2))
        self.prob_list = tk.Label(right, text="", font=("Consolas", 10), justify=tk.LEFT, anchor="w")
        self.prob_list.pack(fill=tk.X)

    def _make_preview(self, parent, title, col, n):
        frame = tk.Frame(parent, padx=4)
        frame.grid(row=0, column=col)
        tk.Label(frame, text=title, font=("Segoe UI", 9)).pack()
        cv = tk.Canvas(frame, width=PREVIEW, height=PREVIEW, bg="black", highlightthickness=0)
        cv.pack()
        s = PREVIEW / n
        cells = [[cv.create_rectangle(c * s, r * s, (c + 1) * s, (r + 1) * s, fill="#000000", width=0)
                  for c in range(n)] for r in range(n)]
        return cv, cells, n

    @staticmethod
    def _show(preview, img):
        cv, cells, n = preview
        for r in range(n):
            for c in range(n):
                v = int(np.clip(img[r, c], 0, 1) * 255) if img is not None else 0
                cv.itemconfig(cells[r][c], fill=f"#{v:02x}{v:02x}{v:02x}")

    # ---- Vẽ ----
    def paint(self, event, erase):
        pos = (event.y / CELL, event.x / CELL)
        # nội suy giữa 2 lần di chuột để nét vẽ liền mạch
        start = self.last if self.last is not None else pos
        steps = max(1, int(np.hypot(pos[0] - start[0], pos[1] - start[1]) / 0.25))
        changed = set()
        for t in np.linspace(0, 1, steps + 1):
            y = start[0] + (pos[0] - start[0]) * t
            x = start[1] + (pos[1] - start[1]) * t
            for r in range(int(y - BRUSH), int(y + BRUSH) + 1):
                for c in range(int(x - BRUSH), int(x + BRUSH) + 1):
                    if not (0 <= r < GRID and 0 <= c < GRID):
                        continue
                    d = np.hypot(r + 0.5 - y, c + 0.5 - x)
                    v = np.clip((BRUSH - d) / (BRUSH - 0.5), 0, 1)  # tâm đậm, rìa nhạt
                    if v <= 0:
                        continue
                    new = max(0.0, self.grid[r, c] - v) if erase else max(self.grid[r, c], v)
                    if new != self.grid[r, c]:
                        self.grid[r, c] = new
                        changed.add((r, c))
        self.last = pos
        for r, c in changed:
            g = int(self.grid[r, c] * 255)
            self.canvas.itemconfig(self.rects[r][c], fill=f"#{g:02x}{g:02x}{g:02x}")

    def on_release(self, _event):
        self.last = None
        self.predict()

    def clear(self):
        self.grid[:] = 0
        for row in self.rects:
            for rect in row:
                self.canvas.itemconfig(rect, fill="#000000")
        self.result.config(text="?")
        self.confidence.config(text="")
        self.prob_list.config(text="")
        self._show(self.prev_input, None)
        self._show(self.prev_stroke, None)

    # ---- Nhận diện ----
    def predict(self):
        img = preprocess(self.grid)
        if img is None:
            self.clear()
            return
        probs, strokes = self.model.predict(img)
        best = int(probs.argmax())

        self.result.config(text=str(best))
        self.confidence.config(text=f"Chắc chắn {probs[best] * 100:.1f}%")
        self._show(self.prev_input, img)
        self._show(self.prev_stroke, self.model.stroke_image(int(strokes.argmax())))
        lines = [f"số {d}  {'█' * int(round(probs[d] * 20)):<20} {probs[d] * 100:5.1f}%" for d in range(10)]
        self.prob_list.config(text="\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Vẽ và nhận diện chữ số")
    ap.add_argument("--model", default="features_k400.npz", help="file mô hình do train_features.py tạo")
    args = ap.parse_args()
    path = os.path.join(HERE, args.model)

    root = tk.Tk()
    if not os.path.exists(path):
        root.withdraw()
        messagebox.showerror("Thiếu file", f"Không tìm thấy {args.model}.\nHãy chạy train_features.py trước.")
        return
    DigitApp(root, FeatureModel(path))
    root.mainloop()


if __name__ == "__main__":
    main()
