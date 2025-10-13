import os
import tqdm
import time
import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import snntorch.functional as SF
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, confusion_matrix, classification_report,\
                            balanced_accuracy_score
from src.config import PROJECT_ROOT

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

########################################
# Class to train the network end-to-end
########################################
class Trainer:
    def __init__(self, model, optimizer, device, is_spiking=True, **kwargs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.is_spiking = is_spiking 
        
        self.weights = torch.tensor([0.3, 0.3, 0.6, 1.0], dtype=torch.float32).to(self.device)
        self.recon_loss = torch.nn.MSELoss()

        if self.is_spiking:
            self.class_loss = SF.ce_count_loss(weight=self.weights) 
            # The spikes at each time step [num_steps x batch_size x num_outputs]
            # are accumulated and then passed through the Cross Entropy Loss function
        else:
            self.class_loss = nn.CrossEntropyLoss()

        self.scaler = torch.amp.GradScaler('cuda')

        # Optional training parameters
        self.gamma = kwargs.get('gamma', 1.0)
        self.acc_steps = kwargs.get('acc_steps', 1)
        self.patience = kwargs.get('patience', 20)
        self.path = kwargs.get('model_path', None)
        self.debug = kwargs.get('debug', False)

        # Tracking metrics
        self.train_loss_ls = [];   self.val_loss_ls = []
        self.train_acc_ls = [];    self.val_acc_ls = []

    def compute_loss(self, decoded, X, spk_out, y, num_spikes):
        cae_loss = self.recon_loss(decoded.squeeze(), X)
        snn_loss = self.class_loss(spk_out, y)

        if self.is_spiking == False and cae_loss.item() > 0:    print(cae_loss.item())

        total_loss = cae_loss + self.gamma * snn_loss 
        return total_loss
    
    def train_epoch(self, dataloader):
        self.model.train()
        self.optimizer.zero_grad()
        train_loss = 0.0
        all_preds, all_labels = [], []
            
        for batch, (X, _, y) in enumerate(dataloader):
            X, y = X.squeeze().to(self.device).float(), y.squeeze().to(self.device)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                spk_percent, decoded, spk_out = self.model(X)
                #print(torch.argmax(spk_out.sum(0),1))
                loss = self.compute_loss(decoded, X, spk_out, y, spk_percent)        
                #loss = loss / self.acc_steps  
                   
            self.scaler.scale(loss).backward()
            
            # DEBUG: print or log gradients -----
            if self.debug:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        grad_max = param.grad.abs().max().item()

                        too_small = abs(grad_mean) < 1e-8
                        too_large = abs(grad_mean) > 1e2 or \
                                    grad_max > 1e3 or \
                                    math.isnan(grad_mean) or \
                                    math.isinf(grad_mean)
                        if too_small or too_large:
                            print(f"[{name}] grad_mean={grad_mean:.6f}, grad_std={grad_std:.6f}, grad_max={grad_max:.6f}")
            # --------------------------------------


            if (batch+1) % self.acc_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if self.is_spiking:
                preds = torch.argmax(spk_out.sum(0), dim=1)         
            else:
                preds = torch.argmax(spk_out, dim=1)  

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            train_loss += loss.item()
        
        # free memory
        del X, y, spk_percent, decoded, spk_out, loss 
        torch.cuda.empty_cache()       

        avg_loss = train_loss / len(dataloader) / self.acc_steps   
        avg_f1 = f1_score(all_labels, all_preds, average="weighted")  
        return avg_loss, avg_f1

    def evaluate(self, dataloader):

        self.model.eval()
        val_loss, sparsity = 0.0, 0.0
        all_preds, all_labels = [], []

        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            for X, _, y in dataloader:
                X, y = X.squeeze().to(self.device).float(), y.squeeze().to(self.device)
                spk_percent, decoded, spk_out = self.model(X)

                loss = self.compute_loss(decoded, X, spk_out, y, spk_percent)
                if self.is_spiking:
                    preds = torch.argmax(spk_out.sum(0), dim=1)         
                else:
                    preds = torch.argmax(spk_out, dim=1)   

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                val_loss += loss.item()
                sparsity += 1 - spk_percent.item()

        # free memory
        del X, y, spk_percent, decoded, spk_out
        torch.cuda.empty_cache() 

        avg_loss = val_loss / len(dataloader)
        avg_f1 = f1_score(all_labels, all_preds, average="weighted")
        avg_sparsity = sparsity / len(dataloader)

        return avg_loss, avg_f1, avg_sparsity
    
    def fit(self, train_loader, val_loader, epochs):
        best_val = -float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_f1 = self.train_epoch(train_loader)
            val_loss, val_f1, sparsity = self.evaluate(val_loader)
            self.train_loss_ls.append(train_loss)
            self.train_acc_ls.append(train_f1)
            self.val_loss_ls.append(val_loss)
            self.val_acc_ls.append(val_f1)

            if val_f1 > best_val:
                best_val = val_f1
                patience_counter = 0
                if self.path:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),}, 
                                os.path.join(MODELS_DIR, self.path))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1} - train_loss: {self.train_loss_ls[-1]:.4f} | train_acc: {self.train_acc_ls[-1]:.4f} | val_loss: {self.val_loss_ls[-1]:.4f} | val_acc: {self.val_acc_ls[-1]:.4f}")

    def classification_metrics(self, dataloader, avg_type='macro', verbose=True):
        self.model.eval()
        ground_truth, predictions = [], []

        with torch.no_grad():
            for X, _, y in dataloader:
                X, y = X.to(self.device).float(), y.squeeze(0).to(self.device)
                _, _, spk_out = self.model(X)

                if self.is_spiking:
                    clss = torch.argmax(spk_out.sum(0), dim=1)         
                else:
                    clss = torch.argmax(spk_out, dim=1)

                predictions.append(clss.item())
                ground_truth.append(y.item())

        if avg_type == 'macro':
            accuracy = accuracy_score(ground_truth, predictions)
        else:
            accuracy = balanced_accuracy_score(ground_truth, predictions)    

        precision = precision_score(ground_truth, predictions, average=avg_type, zero_division=0)
        recall = recall_score(ground_truth, predictions, average=avg_type, zero_division=0)
        f1 = f1_score(ground_truth, predictions, average=avg_type)

        confusion_mx = confusion_matrix(ground_truth, predictions)
        report = classification_report(ground_truth, predictions,
                                    target_names=['WALKING', 'RUNNING', 'SITTING', 'HANDS'])
        print(F"AVERAGE TYPE: {avg_type}\n")
        print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\
              \nF1 score: {f1:.4f}")
        if verbose:
            print("\nClassification Report:\n", report)

        return accuracy, precision, recall, f1, confusion_mx
    

