"""
Chương trình 2: Vẽ chữ số trên bảng 28x28 và nhận diện bằng các cluster K-means.

Cần chạy train_kmeans.py trước để tạo file clusters.npz.
Chạy:  python digit_app.py

Cách dùng: giữ chuột trái để vẽ, chuột phải để tẩy, nút "Xóa" để vẽ lại.
Kết quả được cập nhật mỗi khi nhả chuột.
"""
import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTER_FILE = os.path.join(HERE, "clusters.npz")

GRID = 28          # bảng vẽ 28x28 ô, đúng kích thước ảnh MNIST
CELL = 16          # mỗi ô hiển thị 16x16 pixel trên màn hình
BRUSH = 1.5        # bán kính cọ (tính theo ô)
PREVIEW_CELL = 5   # kích thước ô của các ảnh xem trước nhỏ


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


class DigitApp:
    def __init__(self, root, centroids, labels, sizes):
        self.root = root
        self.centroids = centroids
        self.c_sq = (centroids ** 2).sum(axis=1)
        self.labels = labels
        self.sizes = sizes
        self.grid = np.zeros((GRID, GRID), dtype=np.float32)
        self.last = None

        root.title(f"Nhận diện chữ số bằng K-means ({len(labels)} cluster)")
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

        previews = tk.Frame(right)
        previews.pack(pady=6)
        self.prev_input = self._make_preview(previews, "Ảnh sau chuẩn hóa", 0)
        self.prev_centroid = self._make_preview(previews, "Centroid gần nhất", 1)

        tk.Label(right, text="5 cluster gần nhất", font=("Segoe UI", 10, "bold")).pack(pady=(10, 2))
        self.top_list = tk.Label(right, text="", font=("Consolas", 10), justify=tk.LEFT, anchor="w")
        self.top_list.pack(fill=tk.X)

    def _make_preview(self, parent, title, col):
        frame = tk.Frame(parent, padx=4)
        frame.grid(row=0, column=col)
        tk.Label(frame, text=title, font=("Segoe UI", 9)).pack()
        s = GRID * PREVIEW_CELL
        cv = tk.Canvas(frame, width=s, height=s, bg="black", highlightthickness=0)
        cv.pack()
        cells = [[cv.create_rectangle(c * PREVIEW_CELL, r * PREVIEW_CELL, (c + 1) * PREVIEW_CELL,
                                      (r + 1) * PREVIEW_CELL, fill="#000000", width=0)
                  for c in range(GRID)] for r in range(GRID)]
        return cv, cells

    @staticmethod
    def _show(preview, img):
        cv, cells = preview
        for r in range(GRID):
            for c in range(GRID):
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
        self.top_list.config(text="")
        self._show(self.prev_input, None)
        self._show(self.prev_centroid, None)

    # ---- Nhận diện: tìm centroid gần nhất ----
    def predict(self):
        img = preprocess(self.grid)
        if img is None:
            self.clear()
            return
        x = img.reshape(-1)
        dist = np.sqrt(np.maximum((x ** 2).sum() - 2 * self.centroids @ x + self.c_sq, 0))
        order = np.argsort(dist)
        best = order[0]

        self.result.config(text=str(self.labels[best]))
        self._show(self.prev_input, img)
        self._show(self.prev_centroid, self.centroids[best].reshape(GRID, GRID))
        lines = [f"cluster {i:3d} -> số {self.labels[i]}  (k/c {dist[i]:5.2f}, {self.sizes[i]} ảnh)"
                 for i in order[:5]]
        self.top_list.config(text="\n".join(lines))


def main():
    root = tk.Tk()
    if not os.path.exists(CLUSTER_FILE):
        root.withdraw()
        messagebox.showerror("Thiếu file", "Không tìm thấy clusters.npz.\nHãy chạy train_kmeans.py trước.")
        return
    data = np.load(CLUSTER_FILE)
    DigitApp(root, data["centroids"].astype(np.float32), data["labels"], data["sizes"])
    root.mainloop()


if __name__ == "__main__":
    main()
