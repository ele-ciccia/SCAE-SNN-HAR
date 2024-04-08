import os
os.chdir("/home/eleonora/Documents/Papers/Learned_Spike_Encoding/PAPER EXTENSION")
import sys
sys.path.append("/data")
import numpy as np
import tqdm
import scipy
import random
from cycler import cycler
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate, utils, functional as SF

import warnings
warnings.filterwarnings("ignore")

import utils_ext
import train_utils_ext
import network


torch.manual_seed(55)

plt.rcParams.update({
                    'axes.labelsize': 28.0,
                    'grid.alpha': 0.6,
                    'legend.framealpha': 0.6,
                    "text.usetex": True,
                    "font.family": "serif",
                    'figure.figsize': [15,6],
                    "font.size": 28,
                    "hatch.linewidth": 0.0,
                    "hatch.color": (0,0,0,0.0),
                    "axes.prop_cycle": cycler(color=sns.color_palette("tab10"))
                    })


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

train_dataset = torch.load('data/train_dataset.pt')
val_dataset = torch.load('data/val_dataset.pt')
test_dataset = torch.load('data/test_dataset.pt')

batch = 32
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader =  DataLoader(val_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=1)

class LSE(nn.Module):
    def __init__(self, autoencoder, snn):
        super(LSE, self).__init__()
        self.autoencoder = autoencoder
        self.snn = snn

    def forward(self, x):
        encoded, decoded = self.autoencoder(x)
        clss = self.snn(encoded)
        return encoded, decoded, clss
    
# define hyperparams
epochs = 25
kernel = (1,1,3)
out_channels = [32, 64]
theta_ = 0.5
out_dec = 'rate'
num_act = 4
loss_cae = torch.nn.MSELoss()
loss_snn = SF.ce_rate_loss() #ce_temporal_loss()
acc_steps = 4 #None
eta = 1e-3
lambda_reg = 0.0
alfa = 1
beta = 1
patience = 100

val_mse_layers = []

# Train Loop
def train(model, train, valid, loss_fn_cae, out_dec, optimizer,
          acc_steps, alfa, beta, Lambda, epochs, patience, path):

    if not (
        out_dec.lower() in ['rate', 'latency']
    ):
        raise Exception("The chosen output decoding is not valid.")

    train_loss_list = []
    val_loss_list = []
    cae_loss_list = []
    snn_loss_list = []
    counter = 0
    best_val_loss = float('inf')
    loss_fn_snn = SF.ce_count_loss() if out_dec.lower() == 'rate' else SF.ce_temporal_loss()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        snn_loss_count = 0.0
        cae_loss_count = 0.0
        optimizer.zero_grad()

        for batch, (X, muD, y) in enumerate(train):
            del muD
            X, y = X.squeeze().to(device), y.squeeze().to(device)

            encoded, decoded, spk_out = model(X.float())

            #clss = torch.nn.Softmax(torch.sum(spk_out, 1)) if out_dec.lower() == 'rate'\
            #        else 1 # COMPLETARE con latency

            #clss = torch.nn.functional.softmax(torch.sum(spk_out, 1))

            sparsity_reg = (torch.sum(abs(encoded))) / \
                                torch.prod(torch.tensor(encoded.shape))
            
            cae_loss = loss_fn_cae(decoded, X.float()) 
            cae_loss_count += alfa*cae_loss.item()

            snn_loss = loss_fn_snn(spk_out, y)
            snn_loss_count += beta*snn_loss.item()

            total_loss = alfa*cae_loss  + beta*snn_loss + sparsity_reg
                
            total_loss.backward()
            
            if not acc_steps:
                optimizer.step()
                optimizer.zero_grad()

            if acc_steps and ((batch + 1) % acc_steps == 0):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += total_loss.item()
               
        train_loss_list.append(train_loss/len(train))
        cae_loss_list.append(cae_loss_count/len(train))
        snn_loss_list.append(snn_loss_count/len(train))
            
        with torch.no_grad():
            model.eval()
            val_loss = 0.0

            for batch, (X, muD, y) in enumerate(valid):
                del muD
                X, y = X.squeeze().to(device), y.squeeze().to(device)

                encoded, decoded, spk_out = model(X.float())

                sparsity_reg = (torch.sum(abs(encoded))) / \
                                    torch.prod(torch.tensor(encoded.shape))
                cae_loss = loss_fn_cae(decoded, X.float()) 

                snn_loss = loss_fn_snn(spk_out, y) #valid

                total_loss = alfa*cae_loss + beta*snn_loss + Lambda * sparsity_reg

                val_loss += total_loss.item()
            val_loss_list.append(val_loss/len(valid))
            
            if val_loss_list[-1] < best_val_loss:
                best_val_loss = val_loss_list[-1]
                counter = 0
                if path: torch.save(model.state_dict(), path)

                else: counter += 1
                
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1} - loss: {train_loss_list[-1]} | val_loss: {val_loss_list[-1]}")

    return train_loss_list, val_loss_list, cae_loss_list, snn_loss_list

snn =  network.snn_1(
                    input_dim = 2*10*64, hidden = [16, 1], 
                    n_classes = num_act, 
                    surr_grad = surrogate.atan(), 
                    learn_thr=True, learn_beta = True
                    ).to(device)

# CAE with 2 layers
autoencoder = network.cae_2(
                            theta=theta_ , channels=out_channels, 
                            kernel_size=kernel, 
                            stride = 1, padding=[0,0,1]
                            ).to(device)

net = LSE(autoencoder, snn).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=eta)

train_loss, val_loss, cae_loss, snn_loss = train_utils_ext.train(
                                                model=net, 
                                                train=train_loader, valid=val_loader,
                                                loss_fn_cae=loss_cae, out_dec=out_dec,
                                                optimizer=optimizer, acc_steps=acc_steps,
                                                alfa=alfa, beta=beta, Lambda=lambda_reg, 
                                                epochs=epochs, patience=patience, 
                                                path=None)

val_mse_layers.append(np.mean(val_loss))

plt.plot(train_loss, label='train')
plt.plot(val_loss, label='valid')
#plt.xticks(np.arange(epochs), labels=np.arange(epochs)+1)
plt.legend()
plt.show()

del autoencoder, optimizer, net
torch.cuda.empty_cache()
# CAE with 3 layers
autoencoder = network.cae_3(
                            theta=theta_, channels=out_channels, 
                            kernel_size=kernel, 
                            stride=1, padding=[0,0,1]
                            ).to(device)
    
net = LSE(autoencoder, snn).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=eta)

train_loss1, val_loss1, cae_loss1, snn_loss1 = train_utils_ext.train(
                                                        model=net, 
                                                        train=train_loader, valid=val_loader,
                                                        loss_fn_cae=loss_cae, out_dec=out_dec,
                                                        optimizer=optimizer, acc_steps=acc_steps,
                                                        alfa=alfa, beta=beta, Lambda=lambda_reg, 
                                                        epochs=epochs, patience=patience, 
                                                        path=None)
val_mse_layers.append(np.mean(val_loss1))