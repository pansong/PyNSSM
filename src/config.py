# Network architecture
STATE_SIZE = 2
INPUT_SIZE = 3
OUTPUT_SIZE = 2
STATE_LAYERS = [INPUT_SIZE + STATE_SIZE] + [64] * 6 + [STATE_SIZE]
OUTPUT_LAYERS = [INPUT_SIZE + STATE_SIZE] + [64] * 3 + [OUTPUT_SIZE]

# Training hyperparameters
EPOCHS = 32000
LEARNING_RATE = 1e-3
BETAS = (0.9, 0.99)
DT = 0.01
VAL_RATIO = 0.2
RANDOM_SEED = 166

# Save/plot intervals
SAMPLE_PLOT_INTERVAL = 400
LOSS_PLOT_INTERVAL = 2000
PTH_SAVE_INTERVAL = 4000

# Physics constants (bicycle model)
WHEELBASE = 3.0
STEERING_RATIO = 13.0

# PINN loss weights
LAMBDA_VX = 0.1
LAMBDA_YAW = 0.1

# Paths
DATA_PATH = "./data/RawData.mat"
RESULTS_DIR = "./data/results/"
