import torch
import scipy.io
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

class StateNetwork(nn.Module):
    def __init__(self, state_size, input_size):
        super(StateNetwork, self).__init__()
        # Define state network layers
        self.fc1 = nn.Linear(input_size + state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.output = nn.Linear(64, state_size)
        # Initialize weights and biases
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Forward pass for the state network
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        dxdt = self.output(x)
        return dxdt

class OutputNetwork(nn.Module):
    def __init__(self, state_size, input_size, output_size):
        super(OutputNetwork, self).__init__()
        # Define output network layers
        self.fc1 = nn.Linear(input_size + state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_size)
        # Initialize weights and biases
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Forward pass for the output network
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        y = self.output(x)
        return y

state_size = 2
input_size = 3
output_size = 2

# 创建模型实例
state_net = StateNetwork(state_size, input_size)
output_net = OutputNetwork(state_size, input_size, output_size)

plot_folder = "./data/20240105/"  # 要保存图片的路径

# 加载训练好的模型参数
state_net.load_state_dict(torch.load(Path(plot_folder) / f'state_net_32000.pth'))
output_net.load_state_dict(torch.load(Path(plot_folder) / f'output_net_32000.pth'))

# 在推理时不需要梯度
state_net.eval()
output_net.eval()

# 加载转换后的MATLAB文件
matrixData = scipy.io.loadmat('./data/MatlabData.mat')

# 提取数据
CsvData_array = matrixData['matrixData'][0]
scalingFactors = matrixData['scalingFactors']

Gas_scale = scalingFactors[0][0]['Gas___']['scale'][0, 0][0, 0]
Gas_offset = scalingFactors[0][0]['Gas___']['offset'][0, 0][0, 0]
Decel_scale = scalingFactors[0][0]['Decel_m_s2_']['scale'][0, 0][0, 0]
Decel_offset = scalingFactors[0][0]['Decel_m_s2_']['offset'][0, 0][0, 0]
Steer_scale = scalingFactors[0][0]['Steer_deg_']['scale'][0, 0][0, 0]
Steer_offset = scalingFactors[0][0]['Steer_deg_']['offset'][0, 0][0, 0]
Ax_scale = scalingFactors[0][0]['Ax_g_']['scale'][0, 0][0, 0]
Ax_offset = scalingFactors[0][0]['Ax_g_']['offset'][0, 0][0, 0]
Ay_scale = scalingFactors[0][0]['Ay_g_']['scale'][0, 0][0, 0]
Ay_offset = scalingFactors[0][0]['Ay_g_']['offset'][0, 0][0, 0]
Vx_scale = scalingFactors[0][0]['Vx_km_h_']['scale'][0, 0][0, 0]
Vx_offset = scalingFactors[0][0]['Vx_km_h_']['offset'][0, 0][0, 0]
Yaw_scale = scalingFactors[0][0]['Yaw_deg_sec_']['scale'][0, 0][0, 0]
Yaw_offset = scalingFactors[0][0]['Yaw_deg_sec_']['offset'][0, 0][0, 0]

x_zero_scaled = 0.0 * Vx_scale + Vx_offset
# 初始化CsvData_norm为和CsvData_array相同结构的空列表
CsvData_norm = []
# 对CsvData_array中的每个矩阵进行处理
for matrix in CsvData_array:
    # 对每列应用缩放和偏移
    norm_matrix = matrix.copy()  # 复制原矩阵以保留原始数据
    norm_matrix[:, 0] = matrix[:, 0] * Gas_scale + Gas_offset
    norm_matrix[:, 1] = matrix[:, 1] * Decel_scale + Decel_offset
    norm_matrix[:, 2] = matrix[:, 2] * Steer_scale + Steer_offset
    norm_matrix[:, 3] = matrix[:, 3] * Ax_scale + Ax_offset
    norm_matrix[:, 4] = matrix[:, 4] * Ay_scale + Ay_offset
    norm_matrix[:, 5] = matrix[:, 5] * Vx_scale + Vx_offset
    norm_matrix[:, 6] = matrix[:, 6] * Yaw_scale + Yaw_offset

    # 将处理后的矩阵添加到CsvData_norm中
    CsvData_norm.append(norm_matrix)

# 初始化U, X, Y, U_norm, X_norm, Y_norm为空列表
U, X, Y = [], [], []
U_norm, X_norm, Y_norm = [], [], []

# 遍历CsvData_array和CsvData_norm，提取指定列
for matrix, norm_matrix in zip(CsvData_array, CsvData_norm):
    # 从原始数据中提取
    U.append(matrix[:, [0, 1, 2]])  # 前3列
    X.append(matrix[:, [5, 6]])     # 后2列
    Y.append(matrix[:, [3, 4]])     # 第4、5列

    # 从标准化数据中提取
    U_norm.append(norm_matrix[:, [0, 1, 2]])  # 前3列
    X_norm.append(norm_matrix[:, [5, 6]])     # 后2列
    Y_norm.append(norm_matrix[:, [3, 4]])     # 第4、5列

# 定义积分步长
dt = 0.01

U_torch = [torch.tensor(U_norm[i], dtype=torch.float32) for i in range(len(U_norm))]

# 初始化X_pred_torch为空列表
X_pred_torch = []

for i in range(len(U_torch)):
    # 初始状态，添加到states_pred列表中
    initial_state = torch.tensor(X_norm[i][0,:], dtype=torch.float32).unsqueeze(0)
    states_pred = [initial_state]  

    for t in range(1, U_torch[i].shape[0]):
        last_state = states_pred[-1].squeeze(0)  # 确保last_state是1D张量
        u_tensor = U_torch[i][t-1].unsqueeze(0)  # 将u_tensor转换为2D张量
        state_input = torch.cat((last_state.unsqueeze(0), u_tensor), dim=1)
        dxdt = state_net(state_input)
        # next_state = last_state + dxdt.squeeze(0) * dt
        next_state_unclamped = last_state + dxdt.squeeze(0) * dt
        next_state = next_state_unclamped.clone()
        next_state[0] = torch.clamp(next_state_unclamped[0], min=x_zero_scaled)  # 确保 Vx，即第一个元素，不小于0
        states_pred.append(next_state.unsqueeze(0))

    # 使用整个states_pred列表来创建states_pred_stacked
    states_pred_stacked = torch.cat(states_pred, dim=0)
    X_pred_torch.append(states_pred_stacked)


# 将X_pred_torch中的数据转换为NumPy数组，并保存在X_pred_norm中
X_pred_norm = [x.detach().numpy() for x in X_pred_torch]

X_pred = []

for matrix in X_pred_norm:
    denorm_matrix = matrix.copy()  # 复制原矩阵以保留原始数据
    denorm_matrix[:, 0] = ( matrix[:, 0] - Vx_offset ) / Vx_scale 
    denorm_matrix[:, 1] = ( matrix[:, 1] - Yaw_offset ) / Yaw_scale

    # 将处理后的矩阵添加到CsvData_norm中
    X_pred.append(denorm_matrix)

# 初始化输出预测列表
Y_pred_torch = []

for i in range(len(U_torch)):
    y_pred = []  # 初始化单个序列的预测列表
    for t in range(U_torch[i].shape[0]):
        state_input = X_pred_torch[i][t,:].unsqueeze(0)  # 获取当前状态
        u_input = U_torch[i][t].unsqueeze(0)  # 获取当前控制输入
        # 合并状态和控制输入
        combined_input = torch.cat((state_input, u_input), dim=1)
        y_pred_output = output_net(combined_input).squeeze(0)
        y_pred.append(y_pred_output)

    # 将单个序列的预测添加到总预测列表中
    Y_pred_torch.append(torch.stack(y_pred))

Y_pred_norm = [y.detach().numpy() for y in Y_pred_torch]

Y_pred = []

for matrix in Y_pred_norm:
    denorm_matrix = matrix.copy() 
    denorm_matrix[:, 0] = ( matrix[:, 0] - Ax_offset ) / Ax_scale 
    denorm_matrix[:, 1] = ( matrix[:, 1] - Ay_offset ) / Ay_scale
    Y_pred.append(denorm_matrix)

# 启用LaTeX渲染器和设置字体
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 9

# 定义一个函数来绘制和保存图表
def plot_and_save(data1, data2, data3, data4, sequence_number, plot_folder):
    time = np.linspace(0, (len(data1) - 1) * dt, len(data1))
    fig, axs = plt.subplots(4, 1, figsize=(3.5, 4))  # 尺寸转换为英寸

    # 设置每个子图的 X 轴范围
    x_min, x_max = 0, (len(data1) - 1) * dt

    # 设置 Y 轴标签的统一位置
    y_label_x_position = -0.15

    # 设置X轴只显示整数刻度
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 第一个图
    axs[0].plot(time, data1[:, 0], 'b-', label='Experiment', linewidth=0.5)
    axs[0].plot(time, data2[:, 0], 'r-', label='Simulation', linewidth=0.5)
    axs[0].set_ylabel('$V_x$ (km/h)')
    axs[0].set_xlim(x_min, x_max)
    axs[0].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[0].set_xticklabels([])

    # 第二个图
    axs[1].plot(time, data1[:, 1], 'b-', label='Experiment', linewidth=0.5)
    axs[1].plot(time, data2[:, 1], 'r-', label='Simulation', linewidth=0.5)
    axs[1].set_ylabel('$\dot{\psi}$ (deg/sec)')
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[1].set_xlim(x_min, x_max)
    axs[1].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[1].set_xticklabels([])
    
    # 第三个图
    axs[2].plot(time, data3[:, 0], 'b-', label='Experiment', linewidth=0.5)
    axs[2].plot(time, data4[:, 0], 'r-', label='Simulation', linewidth=0.5)
    axs[2].set_ylabel('$a_x$ ($g$)')
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].set_xlim(x_min, x_max)
    axs[2].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[2].set_xticklabels([])

    # 第四个图
    axs[3].plot(time, data3[:, 1], 'b-', label='Experiment', linewidth=0.5)
    axs[3].plot(time, data4[:, 1], 'r-', label='Simulation', linewidth=0.5)
    axs[3].set_xlabel('Time (sec)')
    axs[3].set_ylabel('$a_y$ ($g$)')
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[3].set_xlim(x_min, x_max)
    axs[3].yaxis.set_label_coords(y_label_x_position, 0.5)

    # 设置图例
    fig.legend(['Experiment', 'Simulation'], loc='upper center', ncol=2, bbox_to_anchor=(0.58, 1))

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)  # 调整顶部和底部边距

    # 保存图表
    plt.savefig(f"{plot_folder}/sequence_{sequence_number}.png", dpi=600)
    plt.close()

# 对于X和X_pred、Y和Y_pred的每个序列绘制和保存图表
for i in range(len(X)):
    plot_and_save(X[i], X_pred[i], Y[i], Y_pred[i], i, plot_folder)


def calculate_errors(actual_list, predicted_list, variable_index):
    # Concatenate all sequences for each variable
    actual_var = np.concatenate([seq[:, variable_index] for seq in actual_list])
    predicted_var = np.concatenate([seq[:, variable_index] for seq in predicted_list])

    # Calculate MSE and MAE for the concatenated sequences
    mse = mean_squared_error(actual_var, predicted_var)
    mae = mean_absolute_error(actual_var, predicted_var)
    
    return mse, mae

# Define variable indices corresponding to Vx, Yaw, Ax, Ay
variable_indices = {
    'Vx': 0,
    'Yaw': 1,
    'Ax': 0,
    'Ay': 1,
}

# Calculate MSE and MAE for each variable
for var_name, var_index in variable_indices.items():
    if var_name in ['Vx', 'Yaw']:
        mse, mae = calculate_errors(X, X_pred, var_index)
        mse_norm, mae_norm = calculate_errors(X_norm, X_pred_norm, var_index)
    else:  # for 'Ax', 'Ay'
        mse, mae = calculate_errors(Y, Y_pred, var_index)
        mse_norm, mae_norm = calculate_errors(Y_norm, Y_pred_norm, var_index)

    print(f'MSE of {var_name}: {mse}')
    print(f'MAE of {var_name}: {mae}')
    
    print(f'MSE of {var_name} (scaled): {mse_norm}')
    print(f'MAE of {var_name} (scaled): {mae_norm}')
