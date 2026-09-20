"""
8-4-8 fully connected AE, train
- date : 20/09/2026
- author : Kohei Yoshii

- input : 8 points sampled from sine wave with random phase
- output: 8 points reconstructed from latent variable (4-dim)
- z : latent variable (4-dim)
"""

import numpy as np

np.random.seed(0)

def make_sine_dataset(n_samples, n_points=8):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    phases = np.random.uniform(0, 2 * np.pi, size=n_samples)
    # sin は [-1, 1] なので [0, 1] にスケーリング
    X = np.array([0.5 * (np.sin(t + p) + 1.0) for p in phases])
    return X.astype(np.float64)  # shape: (n_samples, n_points)


N_TRAIN = 900
N_VAL = 100
X_train = make_sine_dataset(N_TRAIN)
X_val = make_sine_dataset(N_VAL)


INPUT_DIM = 8
LATENT_DIM = 4
OUTPUT_DIM = 8

# He initialization definition
def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


W_enc = he_init(INPUT_DIM, LATENT_DIM)   # (8, 4)
b_enc = np.zeros(LATENT_DIM)             # (4,)
W_dec = he_init(LATENT_DIM, OUTPUT_DIM)  # (4, 8)
b_dec = np.zeros(OUTPUT_DIM)             # (8,)


LEAKY_ALPHA = 0.01  # Leaky ReLUの負の領域の傾き


def relu(x):
    return np.where(x > 0, x, LEAKY_ALPHA * x)


def relu_grad(x):
    return np.where(x > 0, 1.0, LEAKY_ALPHA).astype(x.dtype)


def forward(X):
    """順伝播。中間値も返す(逆伝播で使うため)。"""
    z_pre = X @ W_enc + b_enc      # (N, 4)  Encoderの活性化前
    z = relu(z_pre)                # (N, 4)  潜在変数
    x_pre = z @ W_dec + b_dec      # (N, 8)  Decoderの活性化前
    x_hat = relu(x_pre)            # (N, 8)  復元結果
    cache = (X, z_pre, z, x_pre, x_hat)
    return x_hat, cache


def mse_loss(x_hat, X):
    return np.mean((x_hat - X) ** 2)


def backward(cache, X):
    """誤差逆伝播。各パラメータの勾配を返す。"""
    N = X.shape[0]
    _, z_pre, z, x_pre, x_hat = cache

    # dL/dx_hat  (MSE: L = mean((x_hat - X)^2))
    d_xhat = 2.0 * (x_hat - X) / (N * OUTPUT_DIM)

    # Decoder側
    d_xpre = d_xhat * relu_grad(x_pre)          # (N, 8)
    dW_dec = z.T @ d_xpre                       # (4, 8)
    db_dec = d_xpre.sum(axis=0)                 # (8,)
    d_z = d_xpre @ W_dec.T                      # (N, 4)

    # Encoder側
    d_zpre = d_z * relu_grad(z_pre)             # (N, 4)
    dW_enc = X.T @ d_zpre                       # (8, 4)
    db_enc = d_zpre.sum(axis=0)                 # (4,)

    return dW_enc, db_enc, dW_dec, db_dec


# ---------------------------------------------------------
# 3. 学習ループ(単純な勾配降下法)
# ---------------------------------------------------------
LR = 0.1
EPOCHS = 3000

for epoch in range(1, EPOCHS + 1):
    x_hat, cache = forward(X_train)
    loss = mse_loss(x_hat, X_train)

    dW_enc, db_enc, dW_dec, db_dec = backward(cache, X_train)
    W_enc -= LR * dW_enc
    b_enc -= LR * db_enc
    W_dec -= LR * dW_dec
    b_dec -= LR * db_dec

    if epoch % 300 == 0 or epoch == 1:
        val_hat, _ = forward(X_val)
        val_loss = mse_loss(val_hat, X_val)
        print(f"epoch {epoch:5d}  train_mse={loss:.6f}  val_mse={val_loss:.6f}")

# ---------------------------------------------------------
# 4. 最終評価
# ---------------------------------------------------------
train_hat, _ = forward(X_train)
val_hat, _ = forward(X_val)
print()
print(f"[最終] train_mse = {mse_loss(train_hat, X_train):.6f}")
print(f"[最終] val_mse   = {mse_loss(val_hat, X_val):.6f}")
print("train と val の誤差が近ければ過学習していないと判断できる")

# ---------------------------------------------------------
# 5. 重みの保存 (Verilog用のexportスクリプトから読めるように .npz で保存)
# ---------------------------------------------------------
np.savez(
    "weights/ae_weights.npz",
    W_enc=W_enc, b_enc=b_enc,
    W_dec=W_dec, b_dec=b_dec,
)
print("\n重みを weights/ae_weights.npz に保存しました")

# サンプル1個の推論結果をコンソールに出しておく(後でVerilogの結果と照合する用)
sample = X_val[0]
sample_hat, _ = forward(sample.reshape(1, -1))
np.set_printoptions(precision=6, suppress=True)
print("\n--- 照合用サンプル (X_val[0]) ---")
print("input :", sample)
print("output:", sample_hat.flatten())