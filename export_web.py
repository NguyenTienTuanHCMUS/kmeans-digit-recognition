"""
Chuyển clusters.npz sang web/clusters.js để phiên bản trình duyệt (web/index.html) dùng được.
Centroid được lượng tử hóa về 0..255 (1 byte/pixel) và mã hóa base64 cho gọn.

Chạy sau train_kmeans.py:  python export_web.py
"""
import base64
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

data = np.load(os.path.join(HERE, "clusters.npz"))
centroids = np.clip(np.round(data["centroids"] * 255), 0, 255).astype(np.uint8)

payload = {
    "k": int(centroids.shape[0]),
    "labels": data["labels"].astype(int).tolist(),
    "sizes": data["sizes"].astype(int).tolist(),
    "purity": np.round(data["purity"], 4).tolist(),
    "testAcc": float(data["test_acc"]),
    "centroids": base64.b64encode(centroids.tobytes()).decode("ascii"),
}

os.makedirs(os.path.join(HERE, "web"), exist_ok=True)
out = os.path.join(HERE, "web", "clusters.js")
with open(out, "w", encoding="utf-8") as f:
    f.write("window.CLUSTERS = " + json.dumps(payload) + ";\n")
print(f"Đã ghi {payload['k']} cluster vào {out} ({os.path.getsize(out) / 1024:.0f} KB)")
