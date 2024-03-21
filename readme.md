# Project Setup

## Step 1: Setting Up the Conda Environment
Setting up a Conda environment with all necessary packages and dependencies is required before beginning project execution or development. Follow the steps outlined below to create and activate the environment:

### Creating Environment
1. Initiate the setup process by opening the terminal. For Windows users, Anaconda Prompt or PowerShell is recommended.
2. Use the `cd` command to navigate to your project's directory where the `requirements.txt` file is located. For example:
```bash
cd ~/PyNSSM
```
3. To establish the Conda environment, execute the command below, substituting `<env_name>` with the desired name for the environment:
```bash
conda create --name <env_name> --file requirements.txt
```

The command installs specified packages from `requirements.txt` into the new environment, duration varies by internet speed and computer performance.

### Activating Environment

After creating the environment, you need to activate it to use it for the project:

**Activate the environment:** Run the following command, replacing `<env_name>` with the name of your environment:
```bash
conda activate <env_name>
```

**Verification:** To ensure the environment is set up correctly and all packages are installed, you can execute:
```bash
conda list
```

This command lists all the packages installed in the active Conda environment. Verify that the output includes the packages listed in `requirements.txt`.

## Step 2: Training NSS Model

Before starting the training process, ensure all necessary data and the Python environment are prepared as described in the previous steps. The model training is conducted using the `train.py` script, which utilizes data from `./data/MatlabData.mat` and outputs the training results into `./data/results/`.

### Executing Training Script

1. Ensure you're in the root directory of your project, where you initially set up your environment. If you're following the provided example, you would be in `~/PyNSSM/`.
2. Start the training by executing the command below in your terminal:
```bash
python ./src/train.py
```
This command initiates the `train.py` script, which automatically reads the MATLAB data, trains the NSS model based on this data, and then saves both the intermediate and final results to `./data/results/`.

### Monitoring Training Progress
- **Console Output:** The script provides real-time updates through the terminal, displaying information about the current epoch, loss metrics for both the state network and the output network, as well as the time taken to complete each epoch.
- **Results and Outputs:** All training outputs, including loss plots, prediction visualizations, and saved model weights (`*.pth` files), are stored in `./data/results/`.

## Step 3: Performing Inference

After the model has been trained and the training outputs have been analyzed, the next step is to use the model for inference on new data. This process involves running the `infer.py` script, which loads the trained model weights and performs predictions on the input data specified in `./data/MatlabData.mat`. The results of the inference include various performance metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) for each state or output variable, both in scaled and unscaled forms.

### Executing Inference Script

To perform inference with the trained model, follow these steps:

1. Ensure you're in the root directory of your project, where you have executed the previous commands.
2. Run the inference script by executing the command below in your terminal:
```bash
python ./src/infer.py
```
This command initiates the `infer.py` script, which loads the trained model weights from `./data/results/`, processes the input data from `./data/MatlabData.mat`, and computes the model's predictions. The script then calculates and displays various performance metrics for the predictions.

### Understanding the Output
The output from the `infer.py` script provides detailed metrics on the model's performance, including:
- **MSE and MAE:** These metrics are provided for each output variable (e.g., $V_x$, $\dot{\psi}$, $a_x$, $a_y$), allowing you to assess the model's accuracy. Both metrics are presented in their scaled and unscaled forms to give a comprehensive view of the model's performance.
- **Visualization:** The script also generates plots comparing the predicted values against the actual values for each sequence from `./data/MatlabData.mat`. These plots are saved in `./data/results/` and serve as a visual confirmation of the model's predictive capabilities.
