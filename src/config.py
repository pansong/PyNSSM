# Network architecture
STATE_SIZE = 2
INPUT_SIZE = 3
OUTPUT_SIZE = 2
STATE_LAYERS = [INPUT_SIZE + STATE_SIZE] + [64] * 6 + [STATE_SIZE]
OUTPUT_LAYERS = [INPUT_SIZE + STATE_SIZE] + [64] * 3 + [OUTPUT_SIZE]

# Training hyperparameters
EPOCHS = 32000
LEARNING_RATE = 5e-5
BETAS = (0.9, 0.99)
DT = 0.01

# Save/plot intervals
SAMPLE_PLOT_INTERVAL = 400
LOSS_PLOT_INTERVAL = 2000
PTH_SAVE_INTERVAL = 4000

# Paths
DATA_PATH = "./data/MatlabData.mat"
RESULTS_DIR = "./data/results/"

# Scaling factor field names
SCALING_FIELDS = ['Gas___', 'Decel_m_s2_', 'Steer_deg_', 'Ax_g_', 'Ay_g_', 'Vx_km_h_', 'Yaw_deg_sec_']
