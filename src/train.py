import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
from pathlib import Path

from .model import CustomNetwork
from .data import load_training_data, load_scaling_factors, compute_x_zero_scaled, ExperimentDataset
from .plotting import plot_predictions, plot_loss
from .config import (
    STATE_SIZE, INPUT_SIZE, OUTPUT_SIZE, STATE_LAYERS, OUTPUT_LAYERS,
    EPOCHS, LEARNING_RATE, BETAS, DT, DATA_PATH, RESULTS_DIR,
    SAMPLE_PLOT_INTERVAL, LOSS_PLOT_INTERVAL, PTH_SAVE_INTERVAL,
    VAL_RATIO, RANDOM_SEED,
)


def train_one_epoch(state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size, device, x_zero_scaled):
    state_loss_epoch = 0
    output_loss_epoch = 0

    for U_batch, X_batch, Y_batch in dataloader:
        U_batch, X_batch, Y_batch = U_batch.to(device), X_batch.to(device), Y_batch.to(device)
        state_optimizer.zero_grad()
        output_optimizer.zero_grad()

        states_pred_batch = [X_batch[:, 0, :state_size]]
        for t in range(1, U_batch.shape[1]):
            state_input_batch = torch.cat((states_pred_batch[-1], U_batch[:, t-1, :]), dim=-1)
            dxdt_batch = state_net(state_input_batch)
            next_states = states_pred_batch[-1] + dxdt_batch * dt
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


@torch.no_grad()
def validate(state_net, output_net, criterion, dataloader, dt, state_size, device):
    state_net.eval()
    output_net.eval()
    state_loss_epoch = 0
    output_loss_epoch = 0

    for U_batch, X_batch, Y_batch in dataloader:
        U_batch, X_batch, Y_batch = U_batch.to(device), X_batch.to(device), Y_batch.to(device)

        states_pred_batch = [X_batch[:, 0, :state_size]]
        for t in range(1, U_batch.shape[1]):
            state_input_batch = torch.cat((states_pred_batch[-1], U_batch[:, t-1, :]), dim=-1)
            dxdt_batch = state_net(state_input_batch)
            next_states = states_pred_batch[-1] + dxdt_batch * dt
            states_pred_batch.append(next_states)

        states_pred_batch = torch.stack(states_pred_batch, dim=1)
        loss_state = criterion(states_pred_batch, X_batch[:, :, :state_size])
        state_loss_epoch += loss_state.item()

        output_pred_batch = []
        for t in range(U_batch.shape[1]):
            output_input_batch = torch.cat((states_pred_batch[:, t, :], U_batch[:, t, :]), dim=-1)
            y_pred_batch = output_net(output_input_batch)
            output_pred_batch.append(y_pred_batch)

        output_pred_batch = torch.stack(output_pred_batch, dim=1)
        loss_output = criterion(output_pred_batch, Y_batch)
        output_loss_epoch += loss_output.item()

    state_net.train()
    output_net.train()
    return state_loss_epoch, output_loss_epoch


def training_loop(n_epochs, state_net, output_net, state_optimizer, output_optimizer, criterion,
                  train_loader, val_loader, dt, state_size, output_size, data_folder, device, x_zero_scaled):
    state_losses = []
    output_losses = []
    val_state_losses = []
    val_output_losses = []

    for epoch in range(n_epochs):
        start_time = time.time()
        X_batch, Y_batch, states_pred_batch, output_pred_batch, state_loss_epoch, output_loss_epoch = \
            train_one_epoch(state_net, output_net, state_optimizer, output_optimizer, criterion, train_loader, dt, state_size, device, x_zero_scaled)

        val_state_loss, val_output_loss = validate(state_net, output_net, criterion, val_loader, dt, state_size, device)

        if (epoch + 1) % SAMPLE_PLOT_INTERVAL == 0 or epoch == n_epochs - 1 or epoch == 0:
            time_points = torch.arange(0, X_batch.shape[1]*dt, dt).cpu().numpy()
            plot_predictions(epoch + 1, time_points, states_pred_batch[-1], X_batch[-1].cpu(), output_pred_batch[-1], Y_batch[-1].cpu(), state_size, output_size, Path(data_folder))

        if (epoch + 1) % PTH_SAVE_INTERVAL == 0 or epoch == n_epochs - 1:
            torch.save(state_net.state_dict(), Path(data_folder) / f'state_net_{epoch + 1}.pth')
            torch.save(output_net.state_dict(), Path(data_folder) / f'output_net_{epoch + 1}.pth')

        if (epoch + 1) % LOSS_PLOT_INTERVAL == 0 or epoch == n_epochs - 1:
            plot_loss(epoch + 1, state_losses, output_losses, Path(data_folder), LOSS_PLOT_INTERVAL,
                      val_state_losses, val_output_losses)
            state_losses = []
            output_losses = []
            val_state_losses = []
            val_output_losses = []

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1}/{n_epochs}, "
              f"Train State: {state_loss_epoch:.6f}, Train Output: {output_loss_epoch:.6f}, "
              f"Val State: {val_state_loss:.6f}, Val Output: {val_output_loss:.6f}, "
              f"Duration: {epoch_duration:.2f}s")

        state_losses.append(state_loss_epoch)
        output_losses.append(output_loss_epoch)
        val_state_losses.append(val_state_loss)
        val_output_losses.append(val_output_loss)


if __name__ == "__main__":
    state_net = CustomNetwork(STATE_LAYERS)
    output_net = CustomNetwork(OUTPUT_LAYERS)

    U_torch, Y_torch = load_training_data(DATA_PATH)
    sf = load_scaling_factors(DATA_PATH)
    x_zero_scaled = compute_x_zero_scaled(sf)

    dataset = ExperimentDataset(U_torch, Y_torch)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)

    optimizer_state = optim.Adam(state_net.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optimizer_output = optim.Adam(output_net.parameters(), lr=LEARNING_RATE, betas=BETAS)
    criterion = nn.L1Loss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_net.to(device)
    output_net.to(device)

    print(f"Training on {train_size} sequences, validating on {val_size} sequences")
    training_loop(EPOCHS, state_net, output_net, optimizer_state, optimizer_output, criterion,
                  train_loader, val_loader, DT, STATE_SIZE, OUTPUT_SIZE, RESULTS_DIR, device, x_zero_scaled)
