import numpy as np
import scipy
import random
import time
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
### Use summary from torchinfo
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

######################################################
# Function to generate the delta thresholding encoding
######################################################
def binary_encoding(X):
    """
    X: list of batch tuples
    Returns: tensor of shape (2, 232, 10, 64) with values {-1, 0, 1}
    """
    cir = torch.stack([el[0] for el in X], dim=0) # cir signals
    muD = torch.stack([el[1] for el in X], dim=0)
    Y = torch.stack([el[-1] for el in X], dim=0) # label

    # compute first-order difference along dim=3 (64)
    s_diff = cir[:, :, :, :, 1:] - cir[:, :, :, :, :-1]  # shape (8, 2, 232, 10, 63)

    # repeat the first column to pad and match original shape
    first_col = s_diff[:, :, :, :, 0:1]  # shape (8, 2, 10, 1, 232)
    s = torch.cat([first_col, s_diff], dim=-1)  # shape (8, 2, 232, 10, 64)

    # compute std over dim=3 (64), then mean over dim=2 (10)
    std_over_time = s.std(dim=-1, keepdim=True)  # shape (8, 2, 232, 10, 1)
    alpha = std_over_time.mean(dim=-2, keepdim=True)  # shape (8, 2, 232, 1, 1)

    # thresholding
    output = torch.zeros_like(s)
    output[s > alpha] = 1
    output[s < -alpha] = -1
    
    return (output, muD, Y)

################################################
# Function to compute inference time on test set
################################################
def inference_time(model, test, device):
    """
    Takes as input: model, test loader and device
    Returns: average inference time for encoding and for classification
    """
    encoding_time, classif_time = [], []

    with torch.no_grad():
        for X, _, _ in test:
            X = X.to(device).float()
            
            start_time_enc = time.time()
            encoded, decoded = model.autoencoder(X)
            encoding_time.append(time.time() - start_time_enc)

            start_time_class = time.time()
            clss = model.snn(encoded)
            classif_time.append(time.time() - start_time_class)

    
    return np.mean(encoding_time), np.mean(classif_time)



