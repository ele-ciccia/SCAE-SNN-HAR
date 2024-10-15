import numpy as np
import scipy
import random
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F


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
# Function to save a model
############################

def save_model(model, path):

    torch.save(model.state_dict(), path)

    return


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
    # Compute the DFT of the signal
    dft = scipy.fft.fft(signal)
    
    # Compute the corresponding frequencies
    frequencies = scipy.fft.fftfreq(len(signal))
    
    return dft, frequencies


############################
# Function to compute RMSE
############################

def RMSE(a, b):
    
    return np.sqrt(np.mean((a-b)**2))


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
    total_params = 0
    total_buffers = 0

    for param in model.parameters():
        total_params += param.numel()  # Total number of elements in all parameters
    
    for buffer in model.buffers():
        total_buffers += buffer.numel()  # Total number of elements in all buffers
    
    # Each element is 4 bytes for float32 (the datatype we are using)
    param_size_in_bytes = total_params * 4  
    buffer_size_in_bytes = total_buffers * 4 

    total_size_in_mb = (param_size_in_bytes + buffer_size_in_bytes) / (1024 ** 2)  # Convert to MB
    
    return round(total_size_in_mb, 2)





