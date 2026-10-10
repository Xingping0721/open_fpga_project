"""
MNIST用 784-64-16-64-784 全結合AE, train
- date : 10/10/2026
- author : Kohei Yoshii

simple_ae_ws との違い
- データが 60000 x 784 と大きいので、全データ一括の勾配降下ではなくミニバッチ学習にする
- 層が深くなり単純なSGDだと収束が遅いので、Adam で更新する
- 長く回したときに最後の誤差を詰められるよう、学習率は cosine で LR -> LR*0.01 まで下げる

usage : python train.py --epochs 200 --batch-size 64 --lr 1e-3
"""

import argparse
import os

import numpy as np

from dataset import load_mnist
from model import (
    LAYER_DIMS, init_params, forward, backward, mse_loss, save_params, count_params,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "weights", "mnist_ae_weights.npz")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

# ---------------------------------------------------------
# 1. データの用意 (train 60000枚のうち 5000枚を validation に回す)
# ---------------------------------------------------------
X_all, _, X_test, _ = load_mnist()
X_train, X_val = X_all[:55000], X_all[55000:]
print(f"train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
print(f"layers={LAYER_DIMS}  params={count_params()}")

# ---------------------------------------------------------
# 2. パラメータとAdamの状態
# ---------------------------------------------------------
Ws, bs = init_params(seed=0)
params = Ws + bs  # Adamでまとめて更新するために1つのリストにする
m = [np.zeros_like(p) for p in params]  # 勾配の1次モーメント
v = [np.zeros_like(p) for p in params]  # 勾配の2次モーメント

LR = args.lr
LR_MIN = LR * 0.01
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

rng = np.random.default_rng(0)
step = 0

# ---------------------------------------------------------
# 3. 学習ループ (ミニバッチ + Adam)
# ---------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    # cosine で学習率を下げる (epoch 1 で LR, 最終 epoch で LR_MIN)
    lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + np.cos(np.pi * (epoch - 1) / max(EPOCHS - 1, 1)))
    perm = rng.permutation(len(X_train))
    for start in range(0, len(X_train), BATCH_SIZE):
        X = X_train[perm[start:start + BATCH_SIZE]]
        x_hat, cache = forward(X, Ws, bs)
        dWs, dbs = backward(x_hat, cache, X, Ws)

        step += 1
        for p, g, mi, vi in zip(params, dWs + dbs, m, v):
            mi *= BETA1
            mi += (1 - BETA1) * g
            vi *= BETA2
            vi += (1 - BETA2) * g * g
            m_hat = mi / (1 - BETA1 ** step)
            v_hat = vi / (1 - BETA2 ** step)
            p -= lr * m_hat / (np.sqrt(v_hat) + EPS)  # in-place更新なので Ws, bs にも反映される

    train_loss = mse_loss(forward(X_train, Ws, bs)[0], X_train)
    val_loss = mse_loss(forward(X_val, Ws, bs)[0], X_val)
    print(f"epoch {epoch:3d}  lr={lr:.2e}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

# ---------------------------------------------------------
# 4. 最終評価 (学習に一度も使っていない test データ)
# ---------------------------------------------------------
test_loss = mse_loss(forward(X_test, Ws, bs)[0], X_test)
print()
print(f"[最終] test_mse = {test_loss:.6f}")

# ---------------------------------------------------------
# 5. 重みの保存
# ---------------------------------------------------------
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
save_params(WEIGHTS_PATH, Ws, bs)
print(f"\n重みを {WEIGHTS_PATH} に保存しました")
