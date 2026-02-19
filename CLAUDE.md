# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyNSSM is a neural state-space model for vehicle dynamics prediction built with PyTorch. It predicts vehicle states (velocity, yaw rate) and outputs (longitudinal/lateral acceleration) from control inputs (gas, deceleration, steering) using two coupled neural networks.

## Commands

```bash
# Environment setup (Conda, Python 3.11, PyTorch 2.1.2 + CUDA 12.1)
conda create --name <env_name> --file requirements.txt
conda activate <env_name>

# Train the model (outputs to ./data/results/)
python -m src.train

# Run inference on trained model
python -m src.infer
```

All commands should be run from the project root (`~/PyNSSM`). There is no test suite, linter, or CI/CD configured.

Inference plots require LaTeX. On Ubuntu/Debian: `sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super`.

## Architecture

Two-network state-space model with Euler integration (dt=0.01):

```
Inputs (Gas, Decel, Steer) + State (Vx, Yaw)
    |
[State Network: 5->64x6->2]  -> state derivatives (dx/dt)
    |  Euler integration
[Output Network: 5->64x3->2] -> predicted outputs (Ax, Ay)
```

- **`src/model.py`** -- `CustomNetwork` class (fully-connected, Tanh activations, Xavier init). No final activation on output layer.
- **`src/config.py`** -- All hyperparameters, paths, and constants
- **`src/data.py`** -- Data loading from `data/RawData.mat`, standardization (zero-mean, unit-variance), `ExperimentDataset`
- **`src/plotting.py`** -- All plotting functions (training curves, predictions, inference sequences). Inference plots use LaTeX rendering and Times New Roman at 600 DPI.
- **`src/train.py`** -- Training loop with 80/20 train/val split (Adam optimizer, L1Loss, 32k epochs, full-batch per split). Includes PINN physics losses: velocity non-negativity penalty and bicycle model yaw rate MAE penalty.
- **`src/infer.py`** -- Loads trained weights from final epoch, runs sequential state prediction with Vx >= 0 clamp, computes MSE/MAE metrics

## Key Implementation Details

**Data sources in MATLAB file:** Training uses `U_array`/`Y_array` from `RawData.mat` (raw values); inference uses `matrixData`. Y_array columns: [Vx, Yaw, Ax, Ay]. matrixData columns: inputs [0,1,2] (Gas, Decel, Steer), outputs [3,4] (Ax, Ay), states [5,6] (Vx, Yaw rate). The `ExperimentDataset` splits Y into states (cols 0:2) and outputs (cols 2:4) from the combined Y_array.

**Normalization:** Standardization (zero-mean, unit-variance): `normalized = (raw - mean) / std`. Stats computed from training data via `compute_normalization_stats()`. Denormalization: `raw = normalized * std + mean`. For inference matrixData, X=[cols 5,6] uses `Y_mean[0:2]/Y_std[0:2]`; Y=[cols 3,4] uses `Y_mean[2:4]/Y_std[2:4]`.

**Training mechanism:** State loss is combined with PINN physics losses before backward: `(L_state + LAMBDA_VX * L_vx + LAMBDA_YAW * L_yaw).backward(retain_graph=True)`. `L_vx = mean(ReLU(x_zero_scaled - Vx_pred))` penalizes negative velocity. `L_yaw = mean(|yaw_pred - yaw_est_scaled|)` penalizes deviation from bicycle model yaw rate estimate (WHEELBASE=3.0, STEERING_RATIO=13.0). Output loss backward then propagates gradients through both networks. Separate Adam optimizers step for each network. Validation runs after every epoch.

**Loss plot windowing:** Loss lists are reset every `LOSS_PLOT_INTERVAL` (2000 epochs), so each saved loss plot shows only the most recent window, not cumulative history.

**Inference physical constraints:** Velocity is clamped to >= 0 (scaled zero from `compute_x_zero_scaled`). Yaw rate clamping removed — the PINN-trained network handles yaw rate physics via the bicycle model loss during training.

**Checkpoint naming:** `state_net_{epoch}.pth` and `output_net_{epoch}.pth`. Inference loads from `EPOCHS` in config (currently 32000). Prediction plots saved every 400 epochs, loss curves every 2000, weights every 4000.
