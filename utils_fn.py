import numpy as np
import scipy
import random
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


##################################
# Function to normalize a signal
##################################
def normalize(X):
    
    XX = torch.clone(X)
    min_values, _ = torch.min(XX, dim=3, keepdim=True)
    max_values, _ = torch.max(XX, dim=3, keepdim=True)
    normalized_tensor = (X - min_values) / (max_values - min_values +1e-6)

    return normalized_tensor


############################
# Function to load a model
############################
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

    return


###########################
# Function to compute DFT
###########################
def compute_dft(signal):
    dft = scipy.fft.fft(signal) # Compute the DFT of the signal
    frequencies = scipy.fft.fftfreq(len(signal)) # Compute the corresponding frequencies

    return dft, frequencies


############################
# Function to compute RMSE
############################
def RMSE(a, b): return np.sqrt(np.mean((a-b)**2))


###############################
# Function to compute sparsity
###############################
def compute_sparsity(spike):
    sparsity = 1 - (np.sum(np.abs(spike))) / (2 * spike.shape[-1])
    return round(sparsity, 6)


############################################################
# Function to create complex vector from real and imag part
############################################################
def to_complex(real, imag):     
    return [real[k] + 1j * imag[k] for k in range(len(real))]


##########################################################
# Function to compute the trainable parameters of a model
##########################################################
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


################################################
# Function to compute the size of a model in MB
#################################################
def model_size_in_mb(model):
    total_params, total_buffers = 0, 0

    for param in model.parameters():
        total_params += param.numel()  # Total number of elements in all parameters    
    for buffer in model.buffers():
        total_buffers += buffer.numel()  # Total number of elements in all buffers
    
    # Each element is 4 bytes for float32 (the datatype we are using)
    param_size_in_bytes = total_params * 4  
    buffer_size_in_bytes = total_buffers * 4 
    total_size_in_mb = (param_size_in_bytes + buffer_size_in_bytes) / (1024 ** 2)  # Convert to MB
    
    return round(total_size_in_mb, 2)


######################################################
# Function to plot training curves on train and valid 
######################################################
def plot_curves(train_loss, val_loss, 
                train_acc, val_acc,
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



