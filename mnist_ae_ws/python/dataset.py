"""
MNIST の読み込み (NumPyと標準ライブラリのみ)
- date : 10/10/2026
- author : Kohei Yoshii

初回実行時に data/ へ IDX形式の .gz をダウンロードし、
以降はローカルのファイルを読む。
"""

import gzip
import os
import urllib.request

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("downloading", BASE_URL + filename)
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def _read_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n, rows, cols = np.frombuffer(data[:16], dtype=">u4")
    assert magic == 2051, f"{path} is not an IDX image file"
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows * cols)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n = np.frombuffer(data[:8], dtype=">u4")
    assert magic == 2049, f"{path} is not an IDX label file"
    return np.frombuffer(data[8:], dtype=np.uint8)


def load_mnist():
    """
    返り値: X_train (60000, 784), y_train (60000,), X_test (10000, 784), y_test (10000,)
    画素値は [0, 1] に正規化済み (float64)
    """
    paths = {k: _download(v) for k, v in FILES.items()}
    X_train = _read_images(paths["train_images"]).astype(np.float64) / 255.0
    X_test = _read_images(paths["test_images"]).astype(np.float64) / 255.0
    y_train = _read_labels(paths["train_labels"])
    y_test = _read_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test
