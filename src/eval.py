"""Evaluate all saved checkpoints and plot inference metrics over epochs."""

import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .model import CustomNetwork
from .data import load_training_data, load_inference_data, compute_normalization_stats, normalize_inference_data, compute_x_zero_scaled
from .config import (
    STATE_SIZE, INPUT_SIZE, OUTPUT_SIZE, STATE_LAYERS, OUTPUT_LAYERS,
    DT, DATA_PATH, RESULTS_DIR,
)


def run_inference(state_net, output_net, stats, x_zero_scaled, U_torch, X_norm, Y_norm, X, Y):
    """Run inference and return metrics dict."""
    Y_mean = stats['Y_mean']
    Y_std = stats['Y_std']
    U_mean = stats['U_mean']
    U_std = stats['U_std']

    with torch.no_grad():
        X_pred_torch = []
        for i in range(len(U_torch)):
            initial_state = torch.tensor(X_norm[i][0, :], dtype=torch.float32).unsqueeze(0)
            states_pred = [initial_state]

            for t in range(1, U_torch[i].shape[0]):
                last_state = states_pred[-1].squeeze(0)
                u_tensor = U_torch[i][t-1].unsqueeze(0)
                state_input = torch.cat((last_state.unsqueeze(0), u_tensor), dim=1)
                dxdt = state_net(state_input)
                next_state_unclamped = last_state + dxdt.squeeze(0) * DT
                next_state_clamped = next_state_unclamped.clone()
                next_state_clamped[0] = torch.clamp(next_state_unclamped[0], min=x_zero_scaled)
                # Bicycle model yaw rate constraint
                vx_raw = next_state_clamped[0] * Y_std[0] + Y_mean[0]
                steer_raw = u_tensor[:, 2].squeeze(0) * U_std[2] + U_mean[2]
                x2_est = (vx_raw / 3.6) / 3.0 * torch.tan(steer_raw / 180.0 * torch.pi / 13.0) / torch.pi * 180
                x2_1_scaled = (x2_est * 1.2 - Y_mean[1]) / Y_std[1]
                x2_2_scaled = (x2_est * 0.8 - Y_mean[1]) / Y_std[1]
                x2_high_scaled = torch.max(x2_1_scaled, x2_2_scaled)
                x2_low_scaled = torch.min(x2_1_scaled, x2_2_scaled)
                next_state = next_state_clamped.clone()
                next_state[1] = torch.clamp(next_state_clamped[1], min=x2_low_scaled, max=x2_high_scaled)
                states_pred.append(next_state.unsqueeze(0))

            X_pred_torch.append(torch.cat(states_pred, dim=0))

        X_pred_norm = [x.numpy() for x in X_pred_torch]
        X_pred = []
        for matrix in X_pred_norm:
            X_pred.append(matrix * Y_std[0:2].numpy() + Y_mean[0:2].numpy())

        Y_pred_torch = []
        for i in range(len(U_torch)):
            combined_input = torch.cat((X_pred_torch[i], U_torch[i]), dim=1)
            Y_pred_torch.append(output_net(combined_input))

        Y_pred_norm = [y.numpy() for y in Y_pred_torch]
        Y_pred = []
        for matrix in Y_pred_norm:
            Y_pred.append(matrix * Y_std[2:4].numpy() + Y_mean[2:4].numpy())

    def calc_errors(actual_list, predicted_list, var_index):
        actual = np.concatenate([seq[:, var_index] for seq in actual_list])
        predicted = np.concatenate([seq[:, var_index] for seq in predicted_list])
        return mean_squared_error(actual, predicted), mean_absolute_error(actual, predicted)

    metrics = {}
    for var_name, var_index, actual, predicted, actual_n, predicted_n in [
        ('Vx', 0, X, X_pred, X_norm, X_pred_norm),
        ('Yaw', 1, X, X_pred, X_norm, X_pred_norm),
        ('Ax', 0, Y, Y_pred, Y_norm, Y_pred_norm),
        ('Ay', 1, Y, Y_pred, Y_norm, Y_pred_norm),
    ]:
        mse, mae = calc_errors(actual, predicted, var_index)
        mse_s, mae_s = calc_errors(actual_n, predicted_n, var_index)
        metrics[var_name] = {'mse': mse, 'mae': mae, 'mse_scaled': mse_s, 'mae_scaled': mae_s}

    return metrics


if __name__ == "__main__":
    results_dir = Path(RESULTS_DIR)

    # Find all checkpoint epochs (where both state and output .pth exist)
    state_pths = sorted(results_dir.glob("state_net_*.pth"))
    epochs = []
    for p in state_pths:
        match = re.search(r'state_net_(\d+)\.pth', p.name)
        if match:
            epoch = int(match.group(1))
            if (results_dir / f'output_net_{epoch}.pth').exists():
                epochs.append(epoch)
    epochs.sort()

    print(f"Found {len(epochs)} checkpoints: {epochs[0]} to {epochs[-1]}")

    # Load data once
    U_train, Y_train = load_training_data(DATA_PATH)
    stats = compute_normalization_stats(U_train, Y_train)
    x_zero_scaled = compute_x_zero_scaled(stats)

    CsvData_array = load_inference_data(DATA_PATH)
    U, X, Y, U_norm, X_norm, Y_norm = normalize_inference_data(CsvData_array, stats)
    U_torch = [torch.tensor(U_norm[i], dtype=torch.float32) for i in range(len(U_norm))]

    # Evaluate each checkpoint
    all_metrics = {var: {'mse_scaled': [], 'mae_scaled': []} for var in ['Vx', 'Yaw', 'Ax', 'Ay']}

    for i, epoch in enumerate(epochs):
        print(f"[{i+1}/{len(epochs)}] Evaluating epoch {epoch}...", end=" ", flush=True)

        state_net = CustomNetwork(STATE_LAYERS)
        output_net = CustomNetwork(OUTPUT_LAYERS)
        state_net.load_state_dict(torch.load(results_dir / f'state_net_{epoch}.pth', weights_only=True))
        output_net.load_state_dict(torch.load(results_dir / f'output_net_{epoch}.pth', weights_only=True))
        state_net.eval()
        output_net.eval()

        metrics = run_inference(state_net, output_net, stats, x_zero_scaled, U_torch, X_norm, Y_norm, X, Y)

        for var in all_metrics:
            all_metrics[var]['mse_scaled'].append(metrics[var]['mse_scaled'])
            all_metrics[var]['mae_scaled'].append(metrics[var]['mae_scaled'])

        print(f"MSE(s): Vx={metrics['Vx']['mse_scaled']:.6f}, Yaw={metrics['Yaw']['mse_scaled']:.6f}, "
              f"Ax={metrics['Ax']['mse_scaled']:.6f}, Ay={metrics['Ay']['mse_scaled']:.6f} | "
              f"MAE(s): Vx={metrics['Vx']['mae_scaled']:.6f}, Yaw={metrics['Yaw']['mae_scaled']:.6f}, "
              f"Ax={metrics['Ax']['mae_scaled']:.6f}, Ay={metrics['Ay']['mae_scaled']:.6f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(epochs, all_metrics['Vx']['mse_scaled'], marker='o', label='Vx (Velocity)', linewidth=2, markersize=3)
    ax1.plot(epochs, all_metrics['Ax']['mse_scaled'], marker='s', label='Ax (Long. Accel)', linewidth=2, markersize=3)
    ax1.plot(epochs, all_metrics['Ay']['mse_scaled'], marker='^', label='Ay (Lat. Accel)', color='red', linewidth=3, markersize=3)
    ax1.plot(epochs, all_metrics['Yaw']['mse_scaled'], marker='x', label='Yaw Rate', linestyle='--', markersize=3)
    ax1.set_title('Scaled MSE Trend (Inference)', fontsize=14)
    ax1.set_ylabel('MSE (Scaled)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(epochs, all_metrics['Vx']['mae_scaled'], marker='o', label='Vx (Velocity)', linewidth=2, markersize=3)
    ax2.plot(epochs, all_metrics['Ax']['mae_scaled'], marker='s', label='Ax (Long. Accel)', linewidth=2, markersize=3)
    ax2.plot(epochs, all_metrics['Ay']['mae_scaled'], marker='^', label='Ay (Lat. Accel)', color='red', linewidth=3, markersize=3)
    ax2.plot(epochs, all_metrics['Yaw']['mae_scaled'], marker='x', label='Yaw Rate', linestyle='--', markersize=3)
    ax2.set_title('Scaled MAE Trend (Inference)', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE (Scaled)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(results_dir / 'eval_checkpoints.png', dpi=150)
    plt.show()
    print(f"\nPlot saved to {results_dir / 'eval_checkpoints.png'}")
