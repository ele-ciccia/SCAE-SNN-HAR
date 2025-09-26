import tqdm
import time
import torch
from torch.cuda.amp import autocast
import numpy as np
import snntorch.functional as SF
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, confusion_matrix, classification_report,\
                            balanced_accuracy_score

########################################
# Class to train the network end-to-end
########################################
class Trainer:
    def __init__(self, model, optimizer, device, **kwargs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.weights = torch.tensor([0.057, 0.057, 0.057, 0.826]).to(self.device)
        self.recon_loss = torch.nn.MSELoss()
        self.class_loss = SF.ce_count_loss(weight=self.weights) 
        #The spikes at each time step [num_steps x batch_size x num_outputs]
        #are accumulated and then passed through the Cross Entropy Loss function

        self.scaler = torch.amp.GradScaler('cuda')

        # Optional training parameters
        self.alpha = kwargs.get('alpha', 0.95)
        self.Lambda = kwargs.get('Lambda', 1.0)
        self.acc_steps = kwargs.get('acc_steps', 1)
        self.patience = kwargs.get('patience', 10)
        self.path = kwargs.get('model_path', None)

        # Tracking metrics
        self.train_loss_ls = [];   self.val_loss_ls = []
        self.train_acc_ls = [];    self.val_acc_ls = []

    def compute_loss(self, decoded, X, spk_out, y, num_spikes):
        cae_loss = self.recon_loss(decoded.squeeze(), X)
        #if cae_loss.item() > 0:
        #    print(cae_loss.item())
        snn_loss = self.class_loss(spk_out, y)
        #total_loss = self.alpha * cae_loss + (1-self.alpha) * snn_loss\
        #             + self.Lambda * num_spikes
        total_loss = cae_loss + snn_loss 
        return total_loss
    
    def train_epoch(self, dataloader):
        self.model.train()
        self.optimizer.zero_grad()
        train_loss = 0.0
        #train_acc = 0.0, 0.0
        all_preds, all_labels = [], []
            
        for batch, (X, _, y) in enumerate(dataloader):
            X, y = X.squeeze().to(self.device).float(), y.squeeze().to(self.device)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                spk_percent, decoded, spk_out = self.model(X)
                loss = self.compute_loss(decoded, X, spk_out, y, spk_percent)
                
                #loss = loss / self.acc_steps    
            self.scaler.scale(loss).backward()
                
            if (batch+1) % self.acc_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            preds = torch.argmax(spk_out.sum(0), dim=1)                  
            #train_acc += (preds == y).float().mean().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            train_loss += loss.item()
            
        torch.cuda.empty_cache()       
        avg_loss = train_loss / len(dataloader) / self.acc_steps   
        avg_f1 = f1_score(all_labels, all_preds, average="weighted") 
        #avg_acc = train_acc / len(dataloader) / self.acc_steps    
        return avg_loss, avg_f1

    def evaluate(self, dataloader):
        self.model.eval()
        val_loss, sparsity = 0.0, 0.0
        #val_acc = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            for X, _, y in dataloader:
                X, y = X.squeeze().to(self.device).float(), y.squeeze().to(self.device)
                spk_percent, decoded, spk_out = self.model(X)

                loss = self.compute_loss(decoded, X, spk_out, y, spk_percent)
                preds = torch.argmax(spk_out.sum(0), dim=1)

                #val_acc += (preds == y).float().mean().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                val_loss += loss.item()
                sparsity += 1 - spk_percent.item()

        avg_loss = val_loss / len(dataloader)
        #avg_acc = val_acc / len(dataloader)
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
                                self.path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch%5 == 0:
                print(f"Epoch {epoch+1} - train_loss: {self.train_loss_ls[-1]:.4f} | train_acc: {self.train_acc_ls[-1]:.4f} | val_loss: {self.val_loss_ls[-1]:.4f} | val_acc: {self.val_acc_ls[-1]:.4f}")

    def classification_metrics(self, dataloader, avg_type='macro', verbose=True):
        self.model.eval()
        ground_truth, predictions = [], []

        with torch.no_grad():
            for X, _, y in dataloader:
                X, y = X.to(self.device).float(), y.squeeze(0).to(self.device)
                _, _, spk_out = self.model(X)
                clss = torch.argmax(spk_out.sum(0), dim=1)
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
    


############################################
# Function to train the Spiking classifier #
############################################
def train_snn(model, train, valid, optimizer, epochs, patience, path, verbose = True):
    
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list   = [], []
    counter = 0
    best_val_acc = -float('inf')
    loss_fn_snn = SF.ce_count_loss() 

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
            model.train()
            train_loss, train_acc = 0.0, 0.0
            optimizer.zero_grad()

            for batch, (X, muD, y) in enumerate(train):
                del muD
                X, y = X.squeeze().to(device), y.squeeze().to(device)

                spk_out = model(X.float())
                clss = torch.argmax(torch.sum(spk_out, 0), dim=1)
                   
                train_acc += (sum(clss==y)/len(y)).cpu().item()
                loss = loss_fn_snn(spk_out, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
               
            train_loss_list.append(train_loss/len(train))
            train_acc_list.append(train_acc/len(train))

            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                val_acc = 0.0

                for batch, (X, muD, y) in enumerate(valid):
                    del muD
                    X, y = X.squeeze().to(device), y.squeeze().to(device)

                    spk_out = model(X.float())
                    clss = torch.argmax(torch.sum(spk_out, 0), dim=1)
                    val_acc += (sum(clss==y)/len(y)).cpu().item()

                    loss = loss_fn_snn(spk_out, y) 
                    val_loss += loss.item()

                val_loss_list.append(val_loss / len(valid))
                val_acc_list.append(val_acc / len(valid))
            
                if val_acc_list[-1] > best_val_acc:
                    best_val_acc = val_acc_list[-1]
                    counter = 0
                    if path:
                        torch.save(model.state_dict(), path)

                else:
                    counter += 1
                
                if counter >= patience-1:
                    print(f'Early stopping at epoch {epoch}')
                    break

            torch.cuda.empty_cache()

            if verbose:
                print(f"Epoch {epoch+1} - loss: {round(train_loss_list[-1], 4)} | acc: {round(train_acc_list[-1], 4)} | val_loss: {round(val_loss_list[-1], 4)} | val_acc: {round(val_acc_list[-1], 4)}")

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list


