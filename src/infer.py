import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .model import CustomNetwork
from .data import load_scaling_factors, load_inference_data, normalize, compute_x_zero_scaled
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

    sf = load_scaling_factors(DATA_PATH)
    x_zero_scaled = compute_x_zero_scaled(sf)

    CsvData_array = load_inference_data(DATA_PATH)
    CsvData_norm, U, X, Y, U_norm, X_norm, Y_norm = normalize(CsvData_array, sf)

    U_torch = [torch.tensor(U_norm[i], dtype=torch.float32) for i in range(len(U_norm))]

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
                next_state_clamped = next_state_unclamped.clone()
                next_state_clamped[0] = torch.clamp(next_state_unclamped[0], min=x_zero_scaled)
                x2_est = ((next_state_clamped[0] - sf['Vx_km_h_']['offset']) / sf['Vx_km_h_']['scale'] / 3.6) / 3.0 * torch.tan(((u_tensor[:, 2].squeeze(0) - sf['Steer_deg_']['offset']) / sf['Steer_deg_']['scale']) / 180.0 * torch.pi / 13.0)/torch.pi*180
                x2_1_scaled = x2_est * 1.2 * sf['Yaw_deg_sec_']['scale'] + sf['Yaw_deg_sec_']['offset']
                x2_2_scaled = x2_est * 0.8 * sf['Yaw_deg_sec_']['scale'] + sf['Yaw_deg_sec_']['offset']
                x2_high_scaled = torch.max(x2_1_scaled, x2_2_scaled)
                x2_low_scaled = torch.min(x2_1_scaled, x2_2_scaled)
                next_state = next_state_clamped.clone()
                next_state[1] = torch.clamp(next_state_clamped[1], min=x2_low_scaled, max=x2_high_scaled)
                states_pred.append(next_state.unsqueeze(0))

            states_pred_stacked = torch.cat(states_pred, dim=0)
            X_pred_torch.append(states_pred_stacked)

        X_pred_norm = [x.numpy() for x in X_pred_torch]

        X_pred = []
        for matrix in X_pred_norm:
            denorm_matrix = matrix.copy()
            denorm_matrix[:, 0] = (matrix[:, 0] - sf['Vx_km_h_']['offset']) / sf['Vx_km_h_']['scale']
            denorm_matrix[:, 1] = (matrix[:, 1] - sf['Yaw_deg_sec_']['offset']) / sf['Yaw_deg_sec_']['scale']
            X_pred.append(denorm_matrix)

        Y_pred_torch = []
        for i in range(len(U_torch)):
            combined_input = torch.cat((X_pred_torch[i], U_torch[i]), dim=1)
            Y_pred_torch.append(output_net(combined_input))

        Y_pred_norm = [y.numpy() for y in Y_pred_torch]

        Y_pred = []
        for matrix in Y_pred_norm:
            denorm_matrix = matrix.copy()
            denorm_matrix[:, 0] = (matrix[:, 0] - sf['Ax_g_']['offset']) / sf['Ax_g_']['scale']
            denorm_matrix[:, 1] = (matrix[:, 1] - sf['Ay_g_']['offset']) / sf['Ay_g_']['scale']
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
