import numpy as np
import time
import torch
from config import DEVICE

############################
# Function to load a model
############################
def load_model(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return

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



