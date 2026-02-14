import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from pathlib import Path


def plot_predictions(epoch, time, states_pred, X, output_pred, Y, state_size, output_size, folder):
    """Plot state and output predictions vs ground truth during training."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

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


def plot_loss(epoch, state_losses, output_losses, folder, loss_plot_interval):
    """Plot training loss curves."""
    folder = Path(folder)
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


def plot_and_save(data1, data2, data3, data4, sequence_number, data_folder, dt):
    """Plot inference results: experiment vs simulation for a single sequence."""
    time = np.linspace(0, (len(data1) - 1) * dt, len(data1))

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 9

    fig, axs = plt.subplots(4, 1, figsize=(3.5, 4))

    x_min, x_max = 0, (len(data1) - 1) * dt
    y_label_x_position = -0.15

    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[0].plot(time, data1[:, 0], 'b-', label='Experiment', linewidth=0.5)
    axs[0].plot(time, data2[:, 0], 'r-', label='Simulation', linewidth=0.5)
    axs[0].set_ylabel('$V_x$ (km/h)')
    axs[0].set_xlim(x_min, x_max)
    axs[0].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[0].set_xticklabels([])

    axs[1].plot(time, data1[:, 1], 'b-', label='Experiment', linewidth=0.5)
    axs[1].plot(time, data2[:, 1], 'r-', label='Simulation', linewidth=0.5)
    axs[1].set_ylabel('$\dot{\psi}$ (deg/sec)')
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[1].set_xlim(x_min, x_max)
    axs[1].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[1].set_xticklabels([])

    axs[2].plot(time, data3[:, 0], 'b-', label='Experiment', linewidth=0.5)
    axs[2].plot(time, data4[:, 0], 'r-', label='Simulation', linewidth=0.5)
    axs[2].set_ylabel('$a_x$ ($g$)')
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].set_xlim(x_min, x_max)
    axs[2].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[2].set_xticklabels([])

    axs[3].plot(time, data3[:, 1], 'b-', label='Experiment', linewidth=0.5)
    axs[3].plot(time, data4[:, 1], 'r-', label='Simulation', linewidth=0.5)
    axs[3].set_xlabel('Time (sec)')
    axs[3].set_ylabel('$a_y$ ($g$)')
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[3].set_xlim(x_min, x_max)
    axs[3].yaxis.set_label_coords(y_label_x_position, 0.5)

    fig.legend(['Experiment', 'Simulation'], loc='upper center', ncol=2, bbox_to_anchor=(0.58, 1))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)

    plt.savefig(f"{data_folder}/sequence_{sequence_number}.png", dpi=600)
    plt.close()
