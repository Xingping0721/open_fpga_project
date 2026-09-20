import numpy as np

# ---------------------------------------------------------
# 1. 重みの読み込み
# ---------------------------------------------------------
data = np.load("weights/ae_weights.npz")
W_enc = data["W_enc"]  # (8, 4)
b_enc = data["b_enc"]  # (4,)
W_dec = data["W_dec"]  # (4, 8)
b_dec = data["b_dec"]  # (8,)

LEAKY_ALPHA = 0.01  # train.pyと必ず同じ値にすること


def leaky_relu(x):
    return np.where(x > 0, x, LEAKY_ALPHA * x)


def forward(x):
    """1サンプル分の推論。x は shape (8,) の1次元配列を想定。"""
    z_pre = x @ W_enc + b_enc      # (4,)  Encoderの活性化前
    z = leaky_relu(z_pre)          # (4,)  潜在変数
    y_pre = z @ W_dec + b_dec      # (8,)  Decoderの活性化前
    y = leaky_relu(y_pre)          # (8,)  復元結果
    return y, z


# ---------------------------------------------------------
# 2. 推論したい入力を用意する
# ---------------------------------------------------------
# 例: 正弦波を8点サンプリングしたベクトル(位相は自由に変えてよい)
t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
phase = 0.7  # ここを変えると別の入力で試せる
x = 0.5 * (np.sin(t + phase) + 1.0)

# ---------------------------------------------------------
# 3. 推論の実行
# ---------------------------------------------------------
y, z = forward(x)

np.set_printoptions(precision=6, suppress=True)
print("input (x) :", x)
print("latent (z):", z)
print("output (y):", y)
print("復元誤差(MSE):", np.mean((y - x) ** 2))