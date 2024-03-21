import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from pathlib import Path

# 初始化网络权重，使用Xavier初始化权重和零偏差，以保证权重在合理范围内，促进收敛
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# 构建多层全连接网络，末层不使用Tanh激活
class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(CustomNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers[:-1])  # remove the last Tanh
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

# 加载转换后的MATLAB文件并返回PyTorch张量格式的输入、状态和输出数据，以及缩放因子
def load_data(path):
    # 加载转换后的MATLAB文件
    MatlabData = scipy.io.loadmat(path)
    # 提取数据
    U_array = MatlabData['U_array'][0]
    Y_array = MatlabData['Y_array'][0]
    scalingFactors = MatlabData['scalingFactors']
    # 转换为 PyTorch 张量
    U_torch = [torch.tensor(U_array[i], dtype=torch.float32) for i in range(len(U_array))]
    Y_torch = [torch.tensor(Y_array[i], dtype=torch.float32) for i in range(len(Y_array))]
    return U_torch, Y_torch, scalingFactors

# 自定义数据集类
class ExperimentDataset(Dataset):
    def __init__(self, U, Y):
        self.U = U  # 输入
        self.X = [y[:, 0:2] for y in Y]  # 状态
        self.Y = [y[:, 2:4] for y in Y]  # 输出

    def __len__(self):
        return len(self.U)

    def __getitem__(self, idx):
        return self.U[idx], self.X[idx], self.Y[idx]

# 绘图逻辑的函数
def plot_predictions(epoch, time, states_pred, X, output_pred, Y, state_size, output_size, folder):
    # 确保文件夹存在
    folder.mkdir(parents=True, exist_ok=True)

    # Plotting X_pred (state predictions) against X (true states)
    plt.figure(figsize=(12, 6))
    for j in range(state_size):
        plt.subplot(state_size, 1, j+1)
        plt.plot(time, states_pred[:, j].detach().cpu().numpy(), label=f'x{j+1}_pred')
        plt.plot(time, X[:, j].detach().cpu().numpy(), label=f'x{j+1}', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel(f'X{j+1}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(folder / f'state_epoch_{epoch}.png')
    plt.close()

    # Plotting Y_pred (output predictions) against Y (true outputs)
    plt.figure(figsize=(12, 6))
    for k in range(output_size):
        plt.subplot(output_size, 1, k+1)
        plt.plot(time, output_pred[:, k].detach().cpu().numpy(), label=f'y{k+1}_pred')
        plt.plot(time, Y[:, k].detach().cpu().numpy(), label=f'y{k+1}', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel(f'Y{k+1}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(folder / f'output_epoch_{epoch}.png')
    plt.close()

# 绘制损失曲线的函数
def plot_loss(epoch, state_losses, output_losses, folder, loss_plot_interval):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(state_losses) + 1)
    plt.plot(epochs, state_losses, label='State Network Loss')
    plt.plot(epochs, output_losses, label='Output Network Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss from Epoch {epoch - (loss_plot_interval - 1)} to {epoch}')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder / f'loss_epoch_{epoch}.png')
    plt.close()

# 逐个时间步长更新预测状态
def train_one_epoch(state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size, device, x_zero_scaled):
    state_loss_epoch = 0  # Record the state network loss for the epoch
    output_loss_epoch = 0  # Record the output network loss for the epoch
        
    for U_batch, X_batch, Y_batch in dataloader:
        U_batch, X_batch, Y_batch = U_batch.to(device), X_batch.to(device), Y_batch.to(device)
        state_optimizer.zero_grad()
        output_optimizer.zero_grad()
            
        states_pred_batch = [X_batch[:, 0, :state_size]]
        for t in range(1, U_batch.shape[1]):
            state_input_batch = torch.cat((states_pred_batch[-1], U_batch[:, t-1, :]), dim=-1)
            dxdt_batch = state_net(state_input_batch)
            
            next_states = states_pred_batch[-1] + dxdt_batch * dt

            # 使用新的张量继续后续操作
            states_pred_batch.append(next_states)

        states_pred_batch = torch.stack(states_pred_batch, dim=1)
        loss_state = criterion(states_pred_batch, X_batch[:, :, :state_size])
        loss_state.backward(retain_graph=True)
        
        state_loss_epoch += loss_state.item()

        output_pred_batch = []
        for t in range(U_batch.shape[1]):
            output_input_batch = torch.cat((states_pred_batch[:, t, :], U_batch[:, t, :]), dim=-1)
            y_pred_batch = output_net(output_input_batch)
            output_pred_batch.append(y_pred_batch)

        output_pred_batch = torch.stack(output_pred_batch, dim=1)
        loss_output = criterion(output_pred_batch, Y_batch)
        loss_output.backward()
        
        state_optimizer.step()
        output_optimizer.step()
        output_loss_epoch += loss_output.item()

    return X_batch, Y_batch, states_pred_batch, output_pred_batch, state_loss_epoch, output_loss_epoch

# 运行指定的Epoch数量进行训练，每隔特定Epoch保存模型，绘制损失和输出预测
def training_loop(n_epochs, state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size, data_folder, device, x_zero_scaled):
    sample_plot_interval = 400
    loss_plot_interval = 2000
    PTH_save_interval = 4000

    # 定义用于存储损失的列表
    state_losses = []
    output_losses = []

    for epoch in range(n_epochs):
        # 开始计时
        start_time = time.time()
        X_batch, Y_batch, states_pred_batch, output_pred_batch, state_loss_epoch, output_loss_epoch = \
            train_one_epoch(state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size, device, x_zero_scaled)
        
        if (epoch + 1) % sample_plot_interval == 0 or epoch == n_epochs - 1 or epoch == 0:
            time_points = torch.arange(0, X_batch.shape[1]*dt, dt).cpu().numpy()
            plot_predictions(epoch + 1, time_points, states_pred_batch[-1], X_batch[-1].cpu(), output_pred_batch[-1], Y_batch[-1].cpu(), state_size, output_size, Path(data_folder))
        
        if (epoch + 1) % PTH_save_interval == 0 or epoch == n_epochs - 1:
            torch.save(state_net.state_dict(), Path(data_folder) / f'state_net_{epoch + 1}.pth')
            torch.save(output_net.state_dict(), Path(data_folder) / f'output_net_{epoch + 1}.pth')

        if (epoch + 1) % loss_plot_interval == 0 or epoch == n_epochs - 1:
            plot_loss(epoch + 1, state_losses, output_losses, Path(data_folder), loss_plot_interval)
            # 损失列表更新
            state_losses = []
            output_losses = []

        # Print average loss for the epoch and time taken
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1}/{n_epochs}, State loss: {state_loss_epoch}, Output loss: {output_loss_epoch}, Duration: {epoch_duration:.2f}s")

        # 更新当前间隔内的损失列表
        state_losses.append(state_loss_epoch)
        output_losses.append(output_loss_epoch)


# 主程序入口点
if __name__ == "__main__":
    # 定义网络结构并创建StateNetwork和OutputNetwork的实例
    state_size, input_size, output_size = 2, 3, 2
    state_net = CustomNetwork([input_size + state_size] + [64] * 6 + [state_size])
    output_net = CustomNetwork([input_size + state_size] + [64] * 3 + [output_size])

    # 创建数据集实例
    U_torch, Y_torch, scalingFactors = load_data('./data/MatlabData.mat')
    dataset = ExperimentDataset(U_torch, Y_torch)
    # 创建数据加载器，batch_size等于U_torch中元素的数量
    dataloader = DataLoader(dataset, batch_size=len(U_torch), shuffle=False)
    
    Vx_scale = scalingFactors[0][0]['Vx_km_h_']['scale'][0, 0][0, 0]
    Vx_offset = scalingFactors[0][0]['Vx_km_h_']['offset'][0, 0][0, 0]
    x_zero_scaled = 0.0 * Vx_scale + Vx_offset

    # 定义优化器和损失函数
    optimizer_state = optim.Adam(state_net.parameters(), lr=5e-5, betas=(0.9, 0.99))
    optimizer_output = optim.Adam(output_net.parameters(), lr=5e-5, betas=(0.9, 0.99))
    criterion = nn.L1Loss() # 平均绝对误差
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_net.to(device)
    output_net.to(device)
    
    # 数据保存文件夹
    data_folder = "./data/results/"
    
    # 运行训练循环
    M = int(3.2e4)  # 定义Epoch数量
    dt = 0.01  # 积分步长
    training_loop(M, state_net, output_net, optimizer_state, optimizer_output, criterion, dataloader, dt, state_size, data_folder, device, x_zero_scaled)