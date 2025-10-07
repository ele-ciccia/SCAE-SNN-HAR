import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns

def set_plot_style():
    """Configure global Matplotlib style for all plots."""
    plt.rcParams.update({
        'axes.labelsize': 30.0,
        'grid.alpha': 0.6,
        'legend.framealpha': 0.6,
        "text.usetex": True,
        "font.family": "serif",
        'figure.figsize': [12,6],
        "font.size": 30,
        "hatch.linewidth": 0.0,
        "hatch.color": (0,0,0,0.0),
        "axes.prop_cycle": cycler(color=sns.color_palette("tab10"))
        })
    
######################################################
# Function to plot training curves on train and valid 
######################################################
def plot_curves(train_loss, val_loss, train_acc, val_acc,
                save_fig=False, name=None):
    epochs = np.arange(1, len(train_loss) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Loss subplot
    axs[0].plot(epochs, train_loss, label='Train')
    axs[0].plot(epochs, val_loss, label='Validation')
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title("Training and Validation Loss")

    # Accuracy subplot
    axs[1].plot(epochs, train_acc, label='Train')
    axs[1].plot(epochs, val_acc, label='Validation')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title("Training and Validation Accuracy")

    plt.xticks(np.arange(1, len(train_loss)+1, 10))
    plt.tight_layout()

    if save_fig and name is not None:
        fig.savefig(f"Plots/train_val_curves_{name}.png", bbox_inches='tight')

    plt.show()

#######################################
# Function to generate the raster plot
#######################################
def raster_plot(spikes):
    spikes = spikes.squeeze(1)
    time_steps, neuron_indices = spikes.nonzero(as_tuple=True)
    plt.figure(figsize=(12, 6))
    plt.scatter(time_steps.cpu(), neuron_indices.cpu(), marker='.', color='black')
    plt.xlabel('Time step')
    plt.ylabel('Neuron index')
    plt.yticks(range(spikes.shape[1]))  
    plt.grid(True, which='both', linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.show()