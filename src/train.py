import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import time
from pathlib import Path


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

# 创建StateNetwork和OutputNetwork的实例
state_net = StateNetwork(state_size, input_size)
output_net = OutputNetwork(state_size, input_size, output_size)

# 加载转换后的MATLAB文件
MatlabData = scipy.io.loadmat('./data/MATLAB/MatlabData.mat')
scalingFactors = MatlabData['scalingFactors']

# 提取数据
U_array = MatlabData['U_array'][0]
Y_array = MatlabData['Y_array'][0]

Vx_scale = scalingFactors[0][0]['Vx_km_h_']['scale'][0, 0][0, 0]
Vx_offset = scalingFactors[0][0]['Vx_km_h_']['offset'][0, 0][0, 0]

x_zero_scaled = 0.0 * Vx_scale + Vx_offset

# 转换为 PyTorch 张量
U_torch = [torch.tensor(U_array[i], dtype=torch.float32) for i in range(len(U_array))]
Y_torch = [torch.tensor(Y_array[i], dtype=torch.float32) for i in range(len(Y_array))]

# 定义一个自定的数据集
class ExperimentDataset(Dataset):
    def __init__(self, U, Y):
        self.U = U  # 输入
        self.X = [y[:, 0:2] for y in Y]  # 状态
        self.Y = [y[:, 2:4] for y in Y]  # 输出

    def __len__(self):
        return len(self.U)

    def __getitem__(self, idx):
        return self.U[idx], self.X[idx], self.Y[idx]

# 创建数据集实例
dataset = ExperimentDataset(U_torch, Y_torch)
# 创建数据加载器，batch_size等于U_torch中元素的数量
dataloader = DataLoader(dataset, batch_size=len(U_torch), shuffle=False)

# 定义优化器
optimizer_state = Adam(state_net.parameters(), lr=5e-5, betas=(0.9, 0.99))
optimizer_output = Adam(output_net.parameters(), lr=5e-5, betas=(0.9, 0.99))

# 定义损失函数
criterion = nn.L1Loss()  # 平均绝对误差

# 定义积分步长
dt = 0.01
# 定义Epoch数量
M = int(3.6e4)

# 在训练前将模型移动到 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
state_net.to(device)
output_net.to(device)

# 封装绘图逻辑的函数
def plot_predictions(epoch, time, states_pred, X, output_pred, Y, state_size, folder):
    # 确保文件夹存在
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(folder_path / f'state_epoch_{epoch}.png')
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
    plt.savefig(folder_path / f'output_epoch_{epoch}.png')
    plt.close()

# 定义绘制损失曲线的函数
def plot_loss(epoch, state_losses, output_losses, folder):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(state_losses) + 1)
    plt.plot(epochs, state_losses, label='State Network Loss')
    plt.plot(epochs, output_losses, label='Output Network Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss from Epoch {epoch - (1000-1)} to {epoch}')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder / f'loss_epoch_{epoch}.png')
    plt.close()

# 定义训练循环
def training_loop(n_epochs, state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size):
    # 定义用于存储损失的列表
    state_losses = []
    output_losses = []

    for epoch in range(n_epochs):
        # 开始计时
        start_time = time.time()

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
                
                next_states_unclamped = states_pred_batch[-1] + dxdt_batch * dt
                next_states = torch.clamp(next_states_unclamped, min=x_zero_scaled)
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
        
        # 每400个epoch进行绘图
        if (epoch % 400) == 0 or (epoch == n_epochs - 1):
            time_points = torch.arange(0, X_batch.shape[1]*dt, dt).cpu().numpy()
            plot_predictions(epoch+1, time_points, states_pred_batch[-1], X_batch[-1].cpu(), output_pred_batch[-1], Y_batch[-1].cpu(), state_size, plot_folder)

        # 每4000个epoch保存模型参数
        if ((epoch + 1) % 4000) == 0 or (epoch == n_epochs - 1):
            torch.save(state_net.state_dict(), Path(plot_folder) / f'state_net_{epoch + 1}.pth')
            torch.save(output_net.state_dict(), Path(plot_folder) / f'output_net_{epoch + 1}.pth')
        
        # Print average loss for the epoch and time taken
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1}/{n_epochs}, State loss: {state_loss_epoch}, Output loss: {output_loss_epoch}, Duration: {epoch_duration:.2f}s")

        # 更新当前间隔内的损失列表
        state_losses.append(state_loss_epoch)
        output_losses.append(output_loss_epoch)

        # 每1000个epoch绘图并重置损失列表
        if (epoch + 1) % 1000 == 0 or (epoch == n_epochs - 1):
            plot_loss(epoch + 1, state_losses, output_losses, Path(plot_folder))
            state_losses = []  # 重置列表
            output_losses = []  # 重置列表

plot_folder = "./data/20240105/"  # 要保存图片的路径
# 运行训练循环
training_loop(n_epochs=M, 
              state_net=state_net, 
              output_net=output_net, 
              state_optimizer=optimizer_state, 
              output_optimizer=optimizer_output, 
              criterion=criterion, 
              dataloader=dataloader, 
              dt=dt,
              state_size=state_size)
