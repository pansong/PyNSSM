import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .model import CustomNetwork
from .data import load_training_data, load_inference_data, compute_normalization_stats, normalize_inference_data, compute_x_zero_scaled
from .plotting import plot_and_save
from .config import (
    STATE_SIZE, INPUT_SIZE, OUTPUT_SIZE, STATE_LAYERS, OUTPUT_LAYERS,
    DT, DATA_PATH, RESULTS_DIR, EPOCHS,
)


def calculate_errors(actual_list, predicted_list, variable_index):
    actual_var = np.concatenate([seq[:, variable_index] for seq in actual_list])
    predicted_var = np.concatenate([seq[:, variable_index] for seq in predicted_list])
    mse = mean_squared_error(actual_var, predicted_var)
    mae = mean_absolute_error(actual_var, predicted_var)
    return mse, mae


if __name__ == "__main__":
    state_net = CustomNetwork(STATE_LAYERS)
    output_net = CustomNetwork(OUTPUT_LAYERS)

    state_net.load_state_dict(torch.load(Path(RESULTS_DIR) / f'state_net_{EPOCHS}.pth'))
    output_net.load_state_dict(torch.load(Path(RESULTS_DIR) / f'output_net_{EPOCHS}.pth'))

    state_net.eval()
    output_net.eval()

    # Compute normalization stats from training data
    U_train, Y_train = load_training_data(DATA_PATH)
    stats = compute_normalization_stats(U_train, Y_train)
    x_zero_scaled = compute_x_zero_scaled(stats)

    # Load and normalize inference data
    CsvData_array = load_inference_data(DATA_PATH)
    U, X, Y, U_norm, X_norm, Y_norm = normalize_inference_data(CsvData_array, stats)

    U_torch = [torch.tensor(U_norm[i], dtype=torch.float32) for i in range(len(U_norm))]

    # Extract stats for denormalization
    Y_min = stats['Y_min']
    Y_max = stats['Y_max']
    U_min = stats['U_min']
    U_max = stats['U_max']

    with torch.no_grad():
        X_pred_torch = []

        for i in range(len(U_torch)):
            initial_state = torch.tensor(X_norm[i][0,:], dtype=torch.float32).unsqueeze(0)
            states_pred = [initial_state]

            for t in range(1, U_torch[i].shape[0]):
                last_state = states_pred[-1].squeeze(0)
                u_tensor = U_torch[i][t-1].unsqueeze(0)
                state_input = torch.cat((last_state.unsqueeze(0), u_tensor), dim=1)
                dxdt = state_net(state_input)
                next_state_unclamped = last_state + dxdt.squeeze(0) * DT
                next_state = next_state_unclamped.clone()
                next_state[0] = torch.clamp(next_state_unclamped[0], min=x_zero_scaled)
                states_pred.append(next_state.unsqueeze(0))

            states_pred_stacked = torch.cat(states_pred, dim=0)
            X_pred_torch.append(states_pred_stacked)

        X_pred_norm = [x.numpy() for x in X_pred_torch]

        # Denormalize X predictions: raw = (normalized + 1) / 2 * (max - min) + min
        X_pred = []
        for matrix in X_pred_norm:
            denorm_matrix = (matrix + 1) / 2 * (Y_max[0:2] - Y_min[0:2]).numpy() + Y_min[0:2].numpy()
            X_pred.append(denorm_matrix)

        Y_pred_torch = []
        for i in range(len(U_torch)):
            combined_input = torch.cat((X_pred_torch[i], U_torch[i]), dim=1)
            Y_pred_torch.append(output_net(combined_input))

        Y_pred_norm = [y.numpy() for y in Y_pred_torch]

        # Denormalize Y predictions: raw = (normalized + 1) / 2 * (max - min) + min
        Y_pred = []
        for matrix in Y_pred_norm:
            denorm_matrix = (matrix + 1) / 2 * (Y_max[2:4] - Y_min[2:4]).numpy() + Y_min[2:4].numpy()
            Y_pred.append(denorm_matrix)

    for i in range(len(X)):
        plot_and_save(X[i], X_pred[i], Y[i], Y_pred[i], i, RESULTS_DIR, DT)

    variable_indices = {
        'Vx': 0,
        'Yaw': 1,
        'Ax': 0,
        'Ay': 1,
    }

    for var_name, var_index in variable_indices.items():
        if var_name in ['Vx', 'Yaw']:
            mse, mae = calculate_errors(X, X_pred, var_index)
            mse_norm, mae_norm = calculate_errors(X_norm, X_pred_norm, var_index)
        else:
            mse, mae = calculate_errors(Y, Y_pred, var_index)
            mse_norm, mae_norm = calculate_errors(Y_norm, Y_pred_norm, var_index)

        print(f'MSE of {var_name}: {mse}')
        print(f'MAE of {var_name}: {mae}')
        print(f'MSE of {var_name} (scaled): {mse_norm}')
        print(f'MAE of {var_name} (scaled): {mae_norm}')
