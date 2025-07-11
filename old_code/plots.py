import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# plot trainign curves (loss and accuracy on train and valid set)
def plot_curves(train_loss, val_loss, 
                train_acc, val_acc,
                save_fig=False, name=None):
    plt.plot(np.arange(1, len(train_loss)+1), train_loss, label='Train')
    plt.plot(np.arange(1, len(train_loss)+1), val_loss, label='Validation')
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(1, len(train_loss), 20))
    plt.legend()
    if save_fig:
        plt.savefig("Plots/train_val_loss_"+name+".png", bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(1, len(train_acc)+1), train_acc, label='Train')
    plt.plot(np.arange(1, len(train_acc)+1), val_acc, label='Validation')
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, len(train_loss), 20))
    plt.legend()
    if save_fig:
        plt.savefig("Plots/train_val_acc_"+name+".png", bbox_inches='tight')
    plt.show()

# raster plot for spike output
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