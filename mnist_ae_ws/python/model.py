"""
MNIST用 全結合AE のモデル定義 (NumPyのみ)
- date : 10/10/2026
- author : Kohei Yoshii

- architecture : 784 -> 64 -> 16 -> 64 -> 784
- activation   : 全層 Leaky ReLU (simple_ae_ws と同じ。FPGAで比較と乗算だけで実装できる)
- input        : 28x28 の手書き数字を 784次元にflattenして [0, 1] に正規化したもの
- z            : 潜在変数 (16次元)
"""

import numpy as np

# 層ごとのユニット数。先頭が入力、真ん中が潜在変数、末尾が出力。
LAYER_DIMS = [784, 64, 16, 64, 784]
LATENT_INDEX = 2  # LAYER_DIMS の中で潜在変数にあたる位置

LEAKY_ALPHA = 0.01  # Leaky ReLUの負の領域の傾き (export時のHWと必ず同じ値にすること)


def leaky_relu(x):
    return np.where(x > 0, x, LEAKY_ALPHA * x)


def leaky_relu_grad(x):
    return np.where(x > 0, 1.0, LEAKY_ALPHA).astype(x.dtype)


# He initialization definition
def he_init(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


def init_params(layer_dims=LAYER_DIMS, seed=0):
    """各層の重み W[l] (fan_in, fan_out) とバイアス b[l] (fan_out,) のリストを返す。"""
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
        Ws.append(he_init(fan_in, fan_out, rng))
        bs.append(np.zeros(fan_out))
    return Ws, bs


def forward(X, Ws, bs):
    """
    順伝播。X: (N, 784)
    返り値: x_hat (N, 784), cache (逆伝播用の各層の入力と活性化前の値)
    """
    a = X
    cache = []  # [(層への入力 a, 活性化前 h), ...]
    for W, b in zip(Ws, bs):
        h = a @ W + b
        cache.append((a, h))
        a = leaky_relu(h)
    return a, cache


def encode(X, Ws, bs):
    """潜在変数 z (N, 16) だけを返す。"""
    a = X
    for W, b in zip(Ws[:LATENT_INDEX], bs[:LATENT_INDEX]):
        a = leaky_relu(a @ W + b)
    return a


def decode(z, Ws, bs):
    """潜在変数 z から復元画像 (N, 784) を返す。"""
    a = z
    for W, b in zip(Ws[LATENT_INDEX:], bs[LATENT_INDEX:]):
        a = leaky_relu(a @ W + b)
    return a


def mse_loss(x_hat, X):
    return np.mean((x_hat - X) ** 2)


def backward(x_hat, cache, X, Ws):
    """誤差逆伝播。各層の (dW, db) のリストを返す。"""
    N, D = X.shape
    # dL/dx_hat  (MSE: L = mean((x_hat - X)^2))
    d_a = 2.0 * (x_hat - X) / (N * D)

    dWs = [None] * len(Ws)
    dbs = [None] * len(Ws)
    for l in reversed(range(len(Ws))):
        a_in, h = cache[l]
        d_h = d_a * leaky_relu_grad(h)
        dWs[l] = a_in.T @ d_h
        dbs[l] = d_h.sum(axis=0)
        d_a = d_h @ Ws[l].T  # 1つ前の層へ伝える勾配
    return dWs, dbs


def save_params(path, Ws, bs):
    """W0, b0, W1, b1, ... の名前で .npz に保存する。"""
    arrays = {}
    for l, (W, b) in enumerate(zip(Ws, bs)):
        arrays[f"W{l}"] = W
        arrays[f"b{l}"] = b
    np.savez(path, layer_dims=np.array(LAYER_DIMS), **arrays)


def load_params(path):
    data = np.load(path)
    n_layers = len(data["layer_dims"]) - 1
    Ws = [data[f"W{l}"] for l in range(n_layers)]
    bs = [data[f"b{l}"] for l in range(n_layers)]
    return Ws, bs


def count_params(layer_dims=LAYER_DIMS):
    return sum(i * o + o for i, o in zip(layer_dims[:-1], layer_dims[1:]))
