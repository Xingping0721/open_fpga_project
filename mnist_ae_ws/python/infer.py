"""
MNIST AE, infer
- date : 10/10/2026
- author : Kohei Yoshii

test データの先頭 N_SHOW 枚を復元し、
上段: 入力 / 下段: 復元結果 を並べた画像を results/reconstruction.png に保存する。
(matplotlib を使わず、標準ライブラリの zlib でPNGを書き出す)
"""

import os
import struct
import zlib

import numpy as np

from dataset import load_mnist
from model import load_params, forward, encode, mse_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "weights", "mnist_ae_weights.npz")
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "reconstruction.png")

N_SHOW = 10
SCALE = 4  # 28x28 だと小さいので拡大して保存する


def save_gray_png(path, img):
    """img: (H, W) の uint8 配列をグレースケールPNGとして保存する。"""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))  # 各行の先頭にフィルタ種別0

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


Ws, bs = load_params(WEIGHTS_PATH)
_, _, X_test, y_test = load_mnist()

X = X_test[:N_SHOW]
x_hat, _ = forward(X, Ws, bs)
z = encode(X, Ws, bs)

np.set_printoptions(precision=3, suppress=True)
print(f"test_mse (全10000枚) = {mse_loss(forward(X_test, Ws, bs)[0], X_test):.6f}")
print(f"\n--- test[0] (label={y_test[0]}) の潜在変数 z (16次元) ---")
print(z[0])

# 上段に入力、下段に復元を並べる
top = np.hstack([x.reshape(28, 28) for x in X])
bottom = np.hstack([x.reshape(28, 28) for x in x_hat])
grid = np.vstack([top, bottom])
grid = np.clip(grid, 0.0, 1.0)
grid = np.kron(grid, np.ones((SCALE, SCALE)))  # 最近傍で拡大
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
save_gray_png(OUT_PATH, (grid * 255).round().astype(np.uint8))
print(f"\n復元画像を {OUT_PATH} に保存しました (上段: 入力 / 下段: 復元)")
