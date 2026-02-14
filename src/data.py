import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset

from .config import SCALING_FIELDS


def load_scaling_factors(data_path):
    """Load and parse scaling factors from MATLAB file into a clean dict."""
    mat = scipy.io.loadmat(data_path)
    raw_sf = mat['scalingFactors']
    sf = {}
    for name in SCALING_FIELDS:
        sf[name] = {
            'scale': raw_sf[0][0][name]['scale'][0, 0][0, 0],
            'offset': raw_sf[0][0][name]['offset'][0, 0][0, 0],
        }
    return sf


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


def normalize(CsvData_array, sf):
    """Normalize raw data and split into U/X/Y components (both raw and normalized)."""
    col_sf_map = [
        (0, 'Gas___'), (1, 'Decel_m_s2_'), (2, 'Steer_deg_'),
        (3, 'Ax_g_'), (4, 'Ay_g_'), (5, 'Vx_km_h_'), (6, 'Yaw_deg_sec_'),
    ]

    CsvData_norm = []
    for matrix in CsvData_array:
        norm_matrix = matrix.copy()
        for col, key in col_sf_map:
            norm_matrix[:, col] = matrix[:, col] * sf[key]['scale'] + sf[key]['offset']
        CsvData_norm.append(norm_matrix)

    U, X, Y = [], [], []
    U_norm, X_norm, Y_norm = [], [], []

    for matrix, norm_matrix in zip(CsvData_array, CsvData_norm):
        U.append(matrix[:, [0, 1, 2]])
        X.append(matrix[:, [5, 6]])
        Y.append(matrix[:, [3, 4]])
        U_norm.append(norm_matrix[:, [0, 1, 2]])
        X_norm.append(norm_matrix[:, [5, 6]])
        Y_norm.append(norm_matrix[:, [3, 4]])

    return CsvData_norm, U, X, Y, U_norm, X_norm, Y_norm


def compute_x_zero_scaled(sf):
    """Compute the scaled zero-velocity value."""
    return 0.0 * sf['Vx_km_h_']['scale'] + sf['Vx_km_h_']['offset']


class ExperimentDataset(Dataset):
    def __init__(self, U, Y):
        self.U = U
        self.X = [y[:, 0:2] for y in Y]
        self.Y = [y[:, 2:4] for y in Y]

    def __len__(self):
        return len(self.U)

    def __getitem__(self, idx):
        return self.U[idx], self.X[idx], self.Y[idx]
