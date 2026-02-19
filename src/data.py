import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset


def load_training_data(data_path):
    """Load U_array/Y_array used by training. Returns lists of torch tensors."""
    mat = scipy.io.loadmat(data_path)
    U_array = mat['U_array'][0]
    Y_array = mat['Y_array'][0]
    U_torch = [torch.tensor(U_array[i], dtype=torch.float32) for i in range(len(U_array))]
    Y_torch = [torch.tensor(Y_array[i], dtype=torch.float32) for i in range(len(Y_array))]
    return U_torch, Y_torch


def load_inference_data(data_path):
    """Load matrixData used by inference. Returns raw numpy arrays."""
    mat = scipy.io.loadmat(data_path)
    CsvData_array = mat['matrixData'][0]
    return CsvData_array


def compute_normalization_stats(U_list, Y_list):
    """Compute per-column min and max from raw training data.

    Returns dict with U_min(3,), U_max(3,), Y_min(4,), Y_max(4,) tensors.
    """
    U_all = torch.cat(U_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)

    stats = {
        'U_min': U_all.min(dim=0).values,
        'U_max': U_all.max(dim=0).values,
        'Y_min': Y_all.min(dim=0).values,
        'Y_max': Y_all.max(dim=0).values,
    }
    return stats


def normalize_list(data_list, data_min, data_max):
    """Apply min-max normalization to [-1, 1]: 2*(x - min)/(max - min) - 1."""
    return [2 * (x - data_min) / (data_max - data_min) - 1 for x in data_list]


def normalize_inference_data(CsvData_array, stats):
    """Normalize raw matrixData and split into U/X/Y components (both raw and normalized).

    matrixData columns: [Gas(0), Decel(1), Steer(2), Ax(3), Ay(4), Vx(5), Yaw(6)]
    X = cols [5,6] = [Vx, Yaw] -> uses Y_min[0:2]/Y_max[0:2]
    Y = cols [3,4] = [Ax, Ay]  -> uses Y_min[2:4]/Y_max[2:4]
    """
    U_min_np = stats['U_min'].numpy()
    U_max_np = stats['U_max'].numpy()
    Y_min_np = stats['Y_min'].numpy()
    Y_max_np = stats['Y_max'].numpy()

    U, X, Y = [], [], []
    U_norm, X_norm, Y_norm = [], [], []

    for matrix in CsvData_array:
        u_raw = matrix[:, [0, 1, 2]]
        x_raw = matrix[:, [5, 6]]
        y_raw = matrix[:, [3, 4]]

        u_normalized = 2 * (u_raw - U_min_np) / (U_max_np - U_min_np) - 1
        x_normalized = 2 * (x_raw - Y_min_np[0:2]) / (Y_max_np[0:2] - Y_min_np[0:2]) - 1
        y_normalized = 2 * (y_raw - Y_min_np[2:4]) / (Y_max_np[2:4] - Y_min_np[2:4]) - 1

        U.append(u_raw)
        X.append(x_raw)
        Y.append(y_raw)
        U_norm.append(u_normalized)
        X_norm.append(x_normalized)
        Y_norm.append(y_normalized)

    return U, X, Y, U_norm, X_norm, Y_norm


def compute_x_zero_scaled(stats):
    """Compute the min-max scaled zero-velocity value."""
    return 2 * (0.0 - stats['Y_min'][0].item()) / (stats['Y_max'][0].item() - stats['Y_min'][0].item()) - 1


class ExperimentDataset(Dataset):
    def __init__(self, U, Y):
        self.U = U
        self.X = [y[:, 0:2] for y in Y]
        self.Y = [y[:, 2:4] for y in Y]

    def __len__(self):
        return len(self.U)

    def __getitem__(self, idx):
        return self.U[idx], self.X[idx], self.Y[idx]
