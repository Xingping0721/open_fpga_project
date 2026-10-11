"""
8-4-8 fully connected AE, train
- date : 20/09/2026
- author : Kohei Yoshii

- input : 8 points sampled from sine wave with random phase
- output: 8 points reconstructed from latent variable (4-dim)
- z : latent variable (4-dim)
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面なしでもPNGを保存できるようにする
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300000)
parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "weights", "ae_weights.npz"),
                    help="重みの保存先 (.npz)")
parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"),
                    help="損失曲線・再構成プロットの保存先")
args = parser.parse_args()

np.random.seed(0)

def make_sine_dataset(n_samples, n_points=8):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    phases = np.random.uniform(0, 2 * np.pi, size=n_samples)
    # sin は [-1, 1] なので [0, 1] にスケーリング
    X = np.array([0.5 * (np.sin(t + p) + 1.0) for p in phases])
    return X.astype(np.float64)  # shape: (n_samples, n_points)


N_TRAIN = 90000
N_VAL = 10000
X_train = make_sine_dataset(N_TRAIN)
X_val = make_sine_dataset(N_VAL)


INPUT_DIM = 8
LATENT_DIM = 4
OUTPUT_DIM = 8

# He initialization definition
def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


W_enc = he_init(INPUT_DIM, LATENT_DIM)   # (8, 4)
b_enc = np.zeros(LATENT_DIM)             # (4,), [0, 0, 0, 0]で初期化
W_dec = he_init(LATENT_DIM, OUTPUT_DIM)  # (4, 8)
b_dec = np.zeros(OUTPUT_DIM)             # (8,)


LEAKY_ALPHA = 0.01  # Leaky ReLUの負の領域の傾き


def relu(x):
    return np.where(x > 0, x, LEAKY_ALPHA * x)


def relu_grad(x):
    return np.where(x > 0, 1.0, LEAKY_ALPHA).astype(x.dtype) ## 微分の代わりに、活性化関数の勾配を返す


def forward(X):
    """順伝播。中間値も返す(逆伝播で使うため)。"""
    z_pre = X @ W_enc + b_enc      # (N, 4)  Encoderの活性化前
    z = relu(z_pre)                # (N, 4)  潜在変数
    x_pre = z @ W_dec + b_dec      # (N, 8)  Decoderの活性化前
    x_hat = relu(x_pre)            # (N, 8)  復元結果
    cache = (X, z_pre, z, x_pre, x_hat)  # cacheは逆伝播で使う中間値をまとめたもの
    return x_hat, cache


def mse_loss(x_hat, X):
    return np.mean((x_hat - X) ** 2)


def backward(cache, X):
    """誤差逆伝播。各パラメータの勾配を返す。"""
    N = X.shape[0]
    _, z_pre, z, x_pre, x_hat = cache ## _, はcacheの中のXを使わないので_で受け取る

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
LR = 0.001
EPOCHS = args.epochs
LOG_EVERY = 300

history = []  # (epoch, train_mse, val_mse) 損失曲線のプロット用

for epoch in range(1, EPOCHS + 1):
    x_hat, cache = forward(X_train)
    loss = mse_loss(x_hat, X_train)

    dW_enc, db_enc, dW_dec, db_dec = backward(cache, X_train)
    W_enc -= LR * dW_enc
    b_enc -= LR * db_enc
    W_dec -= LR * dW_dec
    b_dec -= LR * db_dec

    if epoch % LOG_EVERY == 0 or epoch == 1:    ## test lossを計算してログに出す
        val_hat, _ = forward(X_val)
        val_loss = mse_loss(val_hat, X_val)
        history.append((epoch, loss, val_loss))
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
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(
    args.out,
    W_enc=W_enc, b_enc=b_enc,
    W_dec=W_dec, b_dec=b_dec,
)
print(f"\n重みを {args.out} に保存しました")

# ---------------------------------------------------------
# 6. 損失曲線と再構成結果のプロット保存
# ---------------------------------------------------------
os.makedirs(args.results_dir, exist_ok=True)
hist = np.array(history)
np.savetxt(os.path.join(args.results_dir, "loss_history.csv"), hist,
           delimiter=",", header="epoch,train_mse,val_mse", comments="")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist[:, 0], hist[:, 1], label="train")
ax.plot(hist[:, 0], hist[:, 2], label="val", linestyle="--")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE")
ax.set_title("8-4-8 AE training loss")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(args.results_dir, "loss_curve.png"), dpi=120)
plt.close(fig)

# 検証データの先頭4サンプルについて入力と復元結果を重ねて描く
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
for ax, i in zip(axes.flat, range(4)):
    ax.plot(X_val[i], "o-", label="input")
    ax.plot(val_hat[i], "s--", label="reconstruction")
    ax.set_title(f"X_val[{i}]  mse={np.mean((val_hat[i] - X_val[i]) ** 2):.2e}")
    ax.set_xlabel("sample index")
axes[0, 0].legend()
fig.tight_layout()
fig.savefig(os.path.join(args.results_dir, "reconstruction.png"), dpi=120)
plt.close(fig)
print(f"損失曲線・再構成プロットを {args.results_dir} に保存しました")

# サンプル1個の推論結果をコンソールに出しておく(後でVerilogの結果と照合する用)
sample = X_val[0]
sample_hat, _ = forward(sample.reshape(1, -1))
np.set_printoptions(precision=6, suppress=True)
print("\n--- 照合用サンプル (X_val[0]) ---")
print("input :", sample)
print("output:", sample_hat.flatten())