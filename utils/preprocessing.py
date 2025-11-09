import torch, numpy as np
import os
from torch.utils.data import Dataset

"""PREPROCESSING UTILITIES"""

def make_windows(x, labels, win=128, stride=32, require_ratio=0.0):
    X, y = [], []
    for start in range(0, len(x)-win+1, stride):
        seg = x[start:start+win]
        lab = labels[start:start+win]
        # label window anomalous if >= require_ratio fraction of points are anomalous
        y_lab = 1 if (lab.mean() >= require_ratio) else 0
        X.append(seg)
        y.append(y_lab)
    X = np.asarray(X)[:, :, None]   # shape: (N, window, 1)
    y = np.asarray(y).astype(int)
    return X, y


def scale(X, scaler):
    s = scaler.transform(X.reshape(-1,1)).reshape(X.shape)
    return s

def save_datasets(train_ds, val_ds, test_ds, data_dir="data"):
    """Save train, validation, and test datasets to csv format."""
    os.makedirs(data_dir, exist_ok=True)

    torch.save(train_ds.X, os.path.join(data_dir, "train_dataset.csv"))
    torch.save(val_ds.X,   os.path.join(data_dir, "val_dataset.csv"))
    torch.save(test_ds.X,  os.path.join(data_dir, "test_dataset.csv"))

    print(f"Datasets saved to '{data_dir}' directory.")


def load_datasets(data_dir="data"):
    """Load train, validation, and test datasets from disk."""
    train_X = torch.load(os.path.join(data_dir, "train_dataset.csv"))
    val_X   = torch.load(os.path.join(data_dir, "val_dataset.csv"))
    test_X  = torch.load(os.path.join(data_dir, "test_dataset.csv"))

    train_ds = SeqDataset(train_X.numpy())
    val_ds   = SeqDataset(val_X.numpy())
    test_ds  = SeqDataset(test_X.numpy())

    print(f"Datasets loaded from '{data_dir}' directory.")
    return train_ds, val_ds, test_ds


class SeqDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X.astype(np.float32))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i]