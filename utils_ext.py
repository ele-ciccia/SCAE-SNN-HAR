import numpy as np
import scipy
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import random
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


##################################
# Functions to evaluate robustness
##################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, out_dec, verbose=True, plot_conf_mx=True):
    model.eval()
    ground_truth, predictions = [], []

    with torch.no_grad():
        for X, _, y in dataloader:
            X = X.to(device)
        
            ground_truth.append(y.item())
            encoded, decoded, spk_out = model(X.float())
            clss = torch.argmax(torch.sum(spk_out, 0), dim=1) if out_dec.lower() == 'rate'\
                       else 1 # COMPLETARE con latency
            predictions.append(clss.to("cpu").item())

    # MODIFICARE PER MULTICLASS
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    F1 = f1_score(ground_truth, predictions)
    confusion_matrix = metrics.confusion_matrix(ground_truth, predictions)

    if verbose:
        print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 score: {F1}")

    if plot_conf_mx:
        sns.heatmap(confusion_matrix,
            annot=True,
            fmt='g',
            #xticklabels=['Not Spam','Spam'],
            #yticklabels=['Not Spam','Spam']
            )

        plt.ylabel('Actual',fontsize=12)
        plt.xlabel('Prediction',fontsize=12)
        plt.show()

    return accuracy, precision, recall, F1


