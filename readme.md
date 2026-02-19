# PyNSSM

A neural state-space model for vehicle dynamics prediction built with PyTorch.

---

> **Looking for the paper version?**
> The code in this repository has been refactored and optimized since the original publication. For the exact code used in the paper, please download the [v1.0-paper Release](https://github.com/pansong/PyNSSM/releases/tag/v1.0-paper) on GitHub.

---

## What's New in v2.0

This version introduces a modular package structure and several optimizations over the original codebase:

- **Modular architecture** — Model, data loading, configuration, and plotting logic are separated into dedicated modules under `src/`.
- **Vectorized inference** — Batch tensor operations replace per-element loops where possible.
- **Combined loss passes** — State and output network gradients are computed in a single backward pass per batch, reducing overhead.
- **Improved readability** — Cleaner separation of concerns, consistent naming, and centralized configuration.

## Project Setup

### Step 1: Setting Up the Conda Environment

A Conda environment with all necessary packages and dependencies is required before running the project. Follow the steps below to create and activate the environment.

#### Creating Environment

1. Open your terminal. For Windows users, Anaconda Prompt or PowerShell is recommended.
2. Navigate to the project root where `requirements.txt` is located:
```bash
cd ~/PyNSSM
```
3. Create the Conda environment, replacing `<env_name>` with your desired name:
```bash
conda create --name <env_name> --file requirements.txt
```

#### Activating Environment

Activate the environment:
```bash
conda activate <env_name>
```

Verify that all packages are installed:
```bash
conda list
```

### Step 1.5: Installing LaTeX Dependencies

The inference plots use LaTeX rendering for axis labels. If you see `RuntimeError: Failed to process string with tex because latex could not be found`, install the following system packages:

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

**Fedora/RHEL:**
```bash
sudo dnf install texlive-latex texlive-dvipng texlive-cm-super
```

**macOS (via Homebrew):**
```bash
brew install --cask mactex
```

Verify LaTeX is available:
```bash
latex --version
```

## File Structure

The project is organized as a Python package under `src/`:

```
PyNSSM/
├── data/
│   ├── RawData.mat              # Training and inference data
│   └── results/                # Output directory for weights, plots, and metrics
├── src/
│   ├── __init__.py
│   ├── config.py               # Hyperparameters, paths, and constants
│   ├── model.py                # Network architecture (CustomNetwork)
│   ├── data.py                 # Data loading, normalization, and dataset classes
│   ├── plotting.py             # All plotting functions (training curves, inference)
│   ├── train.py                # Training loop and entrypoint
│   └── infer.py                # Inference and evaluation entrypoint
├── requirements.txt
├── CLAUDE.md
└── readme.md
```

| Module | Responsibility |
|---|---|
| `config.py` | All hyperparameters (epochs, learning rate, dt), layer sizes, file paths, physics constants, and PINN loss weights |
| `model.py` | `CustomNetwork` class — fully-connected layers with Tanh activations and Xavier initialization |
| `data.py` | Loading MATLAB data, min-max normalization, and the `ExperimentDataset` class |
| `plotting.py` | Training prediction plots, loss curves, and publication-quality inference figures |
| `train.py` | Training loop with Adam optimizer, L1 loss, PINN physics penalties (Vx >= 0, bicycle model yaw rate), periodic checkpointing, and GPU support |
| `infer.py` | Sequential state prediction with Vx >= 0 clamp, MSE/MAE metrics |

## Step 2: Training the Model

Ensure all data and the Python environment are prepared as described above. Training reads data from `./data/RawData.mat` and saves results to `./data/results/`.

### Executing Training Script

From the project root (`~/PyNSSM`):
```bash
python -m src.train
```

### Monitoring Training Progress

- **Console Output:** Real-time updates showing the current epoch, state/output loss, and epoch duration.
- **Results and Outputs:** Loss plots, prediction visualizations, and saved model weights (`*.pth` files) are stored in `./data/results/`.

## Step 3: Performing Inference

After training, use the trained model for inference. The script loads weights from `./data/results/`, runs sequential predictions on the data in `./data/RawData.mat`, and computes performance metrics.

### Executing Inference Script

From the project root:
```bash
python -m src.infer
```

### Understanding the Output

- **MSE and MAE:** Metrics for each variable ($V_x$, $\dot{\psi}$, $a_x$, $a_y$) in both scaled and unscaled forms.
- **Visualization:** Plots comparing predicted vs. actual values for each sequence, saved in `./data/results/`.
