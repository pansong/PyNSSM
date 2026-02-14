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

## Architecture

Two-network state-space model with Euler integration (dt=0.01):

```
Inputs (Gas, Decel, Steer) + State (Vx, Yaw)
    ↓
[State Network: 5→64×6→2]  → state derivatives (dx/dt)
    ↓  Euler integration
[Output Network: 5→64×3→2] → predicted outputs (Ax, Ay)
```

- **`src/model.py`** — `CustomNetwork` class (fully-connected, Tanh activations, Xavier init)
- **`src/config.py`** — All hyperparameters, paths, and constants
- **`src/data.py`** — Data loading from `data/MatlabData.mat`, normalization, scaling factors, `ExperimentDataset`
- **`src/plotting.py`** — All plotting functions (training curves, predictions, inference sequences)
- **`src/train.py`** — Training loop and entrypoint (Adam optimizer, L1Loss, 32k epochs, full-batch)
- **`src/infer.py`** — Loads trained weights from epoch 32k, runs sequential state prediction with physical constraints (Vx ≥ 0, yaw rate bounded by bicycle model), computes MSE/MAE metrics

Data is normalized using scaling factors stored in the MATLAB file. Training checkpoints (weights, loss plots, prediction visualizations) are saved to `data/results/` at regular intervals.
