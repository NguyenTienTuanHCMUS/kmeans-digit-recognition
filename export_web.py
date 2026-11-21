"""
Chuyển mô hình đặc trưng (features_k400.npz) sang web/model.js để trang web (web/index.html) dùng được.
Các mảng được giữ ở dạng float32 và mã hóa base64.

Chạy sau train_features.py:  python export_web.py
                             python export_web.py --model features_k200.npz
"""
import argparse
import base64
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="features_k400.npz")
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, args.model))
    payload = {
        "k": int(d["centroids"].shape[0]),
        "patch": int(round(d["centroids"].shape[1] ** 0.5)),
        "pool": int(d["pool"]) if "pool" in d.files else 2,   # lưới gộp pool x pool
        "testAcc": float(d["test_acc"]),
        "centroids": b64(d["centroids"]),   # K x 36: các nét cơ bản (đã làm trắng)
        "zcaMean": b64(d["zca_mean"]),      # 36
        "zcaW": b64(d["zca_W"]),            # 36 x 36
        "featMean": b64(d["feat_mean"]),    # 4K
        "featStd": b64(d["feat_std"]),      # 4K
        "coef": b64(d["coef"]),             # 10 x 4K: hồi quy logistic
        "intercept": b64(d["intercept"]),   # 10
    }

    os.makedirs(os.path.join(HERE, "web"), exist_ok=True)
    out = os.path.join(HERE, "web", "model.js")
    with open(out, "w", encoding="utf-8") as f:
        f.write("window.MODEL = " + json.dumps(payload) + ";\n")
    print(f"Đã ghi mô hình {payload['k']} nét vào {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
