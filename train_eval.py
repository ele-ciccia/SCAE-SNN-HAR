import tqdm
import time
import torch
import numpy as np
import snntorch.functional as SF
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, confusion_matrix, classification_report,\
                            balanced_accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()

##############
# Train Loop #
##############
def train_fn(model, train, valid, loss_fn_cae, out_dec, optimizer,
          acc_steps, alpha, beta, Lambda, epochs, patience, path, 
          verbose = True):

    if not (
        out_dec.lower() in ['rate', 'latency']
    ):
        raise Exception("The chosen output decoding is not valid.")

    #assert (Lambda >= 0 & alpha >= 0 & beta >= 0)

    train_loss_list, val_loss_list = [], []
    cae_loss_list, snn_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    counter = 0
    best_val_acc = -float('inf')
    loss_fn_snn = SF.ce_count_loss() if out_dec.lower() == 'rate' \
                                     else SF.ce_temporal_loss()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            snn_loss_count = 0.0
            cae_loss_count = 0.0
            optimizer.zero_grad()

            for batch, (X, muD, y) in enumerate(train):
                del muD
                #start_time = time.time()
                X, y = X.squeeze().to(device), y.squeeze().to(device)

                encoded, decoded, spk_out = model(X.float())

                #print(torch.sum(spk_out, 0).shape)

                clss = torch.argmax(torch.sum(spk_out, 0), dim=1) if out_dec.lower() == 'rate'\
                       else 1 # COMPLETARE con latency
                
                train_acc += (sum(clss==y)/len(y)).cpu().item()

                #train_acc += SF.acc.accuracy_rate(decoded, X.float()).item() if out_dec.lower() == 'rate'\
                #            else SF.acc.accuracy_temporal(decoded, X.float()).item()

                sparsity_reg = (torch.sum(abs(encoded))) / \
                                torch.prod(torch.tensor(encoded.shape))

                cae_loss = loss_fn_cae(decoded, X.float()) 
                cae_loss_count += alpha*cae_loss.item()

                snn_loss = loss_fn_snn(spk_out, y)
                snn_loss_count += beta*snn_loss.item()

                total_loss = alpha*cae_loss  + beta*snn_loss + \
                             Lambda * sparsity_reg
                
                #scaler.scale(total_loss).backward()
                total_loss.backward()

                if not acc_steps:
                    optimizer.step()
                    #scaler.step(optimizer)
                    #scaler.update()
                    optimizer.zero_grad()

                if acc_steps and ((batch + 1) % acc_steps == 0):
                        optimizer.step()
                        #scaler.step(optimizer)
                        #scaler.update()
                        optimizer.zero_grad()

                train_loss += total_loss.item()
               
            train_loss_list.append(train_loss/len(train))
            train_acc_list.append(train_acc/len(train))

            cae_loss_list.append(cae_loss_count/len(train))
            snn_loss_list.append(snn_loss_count/len(train))

            del cae_loss_count, snn_loss_count, sparsity_reg
            
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                val_acc = 0.0

                for batch, (X, muD, y) in enumerate(valid):
                    del muD
                    X, y = X.squeeze().to(device), y.squeeze().to(device)

                    encoded, decoded, spk_out = model(X.float())

                    clss = torch.argmax(torch.sum(spk_out, 0), dim=1)
                    val_acc += (sum(clss==y)/len(y)).cpu().item()

                    sparsity_reg = (torch.sum(abs(encoded))) / \
                                    torch.prod(torch.tensor(encoded.shape))
                    cae_loss = loss_fn_cae(decoded, X.float()) 

                    snn_loss = loss_fn_snn(spk_out, y) #valid

                    total_loss = alpha*cae_loss + beta*snn_loss + Lambda * sparsity_reg

                    del cae_loss, snn_loss, sparsity_reg

                    val_loss += total_loss.item()

                val_loss_list.append(val_loss/len(valid))
                val_acc_list.append(val_acc/len(valid))
            
                if val_acc_list[-1] > best_val_acc:
                    best_val_acc = val_acc_list[-1]
                    counter = 0
                    if path:
                        torch.save(model.state_dict(), path)

                else:
                    counter += 1
                
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            torch.cuda.empty_cache()

            if verbose:
                print(f"Epoch {epoch+1} - loss: {round(train_loss_list[-1], 4)} | acc: {round(train_acc_list[-1], 4)} | val_loss: {round(val_loss_list[-1], 4)} | val_acc: {round(val_acc_list[-1], 4)}")

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list, cae_loss_list, snn_loss_list


############################################
# Function to evaluate the end-to-end model
############################################
def evaluate_(model, dataloader, loss_fn):
    model.eval()
    total_mse = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            encoded, decoded, freq, amp = model(X.float())

            mse_signal = loss_fn(decoded, X.float()).item()
            mse_freq = loss_fn(freq, y[:,0].float()).item()
            mse_amp = loss_fn(amp, y[:,1].float()).item()

            total_mse += mse_signal + mse_freq + mse_amp

    average_mse = total_mse / len(dataloader)
    return average_mse

#############
def evaluate(model, dataloader, out_dec, avg_type, verbose=True):
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

    if avg_type == 'macro':
        accuracy = round(accuracy_score(ground_truth, predictions), 4)
    else:
        accuracy = round(balanced_accuracy_score(ground_truth, predictions), 4)
        
    precision = round(precision_score(ground_truth, predictions, average=avg_type), 4)
    recall = round(recall_score(ground_truth, predictions, average=avg_type), 4)
    F1 = round(f1_score(ground_truth, predictions, average=avg_type),4)

    confusion_mx = confusion_matrix(ground_truth, predictions)
    report = classification_report(ground_truth, predictions,
                                   target_names=['WALKING', 'RUNNING', 'SITTING', 'HANDS'])
    print(F"AVERAGE TYPE: {avg_type}\n")
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 score: {F1}")
    if verbose:
        print("\nClassification Report:\n", report)

    return accuracy, precision, recall, F1, confusion_mx
