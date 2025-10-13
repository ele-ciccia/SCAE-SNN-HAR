import os
import numpy as np
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.config import DEVICE, SURROGATE_FN, INP_SHAPE, PROJECT_ROOT

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

############################
# Function to load a model
############################
def load_model(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return

##################################
# Function count params of a model
##################################
def count_params(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
def inference_time(model, test, device, path=None, autoenc=True):
    """
    Takes as input: model, test loader and device
    Returns: average inference time for encoding and for classification
    """
    if path:
        checkpoint = torch.load(os.path.join(MODELS_DIR, path), map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if autoenc:
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
    
    else:
        classif_time = []

        with torch.no_grad():
            for X, _, _ in test:
                X = X.to(device).float()
                
                start_time_enc = time.time()
                clss = model.snn(X)
                classif_time.append(time.time() - start_time_enc)
        
        return np.mean(classif_time)

#######################################
# Function to compute encoding sprsity 
#######################################
def compute_sparsity(model, path, test):
    """
    Takes as input: model, test loader 
    Returns: average sparsity of the spike encoding
    """
    checkpoint = torch.load(os.path.join(MODELS_DIR, path), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sparsity = 0.0

    with torch.no_grad():
            for X, _, _ in test:
                X = X.to(DEVICE).float()
                n_spikes, decoded, clss = model(X)
                sparsity += 1 - n_spikes.item()

    return sparsity / len(test)


#################################################
# Function to compute per layer spike actity rate 
#################################################
def spike_activity(model, dataloader, path, device, visualize=True):
    """
    Evaluates spike activity per layer in an SNN model.
    Works with networks that use snn.Leaky() layers and produce spike tensors per timestep.

    Returns:
        mean_activity: dict mapping layer name -> mean spike activity (float)
        overall_mean: average of per-layer means (float)
    """
    # Load model checkpoints
    checkpoint = torch.load(os.path.join(MODELS_DIR, path), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Storage for per-layer spike stats
    layer_spike_activity = {}

    # Helper
    def extract_tensor_from_output(output):
        """
        output may be:
          - a Tensor (spikes)
          - a tuple/list where first element is spikes (e.g. (spikes, mem))
          - nested structures; this will try to find the first Tensor
        """
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for el in output:
                if torch.is_tensor(el):
                    return el
                # recursively search
                if isinstance(el, (tuple, list)):
                    found = extract_tensor_from_output(el)
                    if found is not None:
                        return found
        return None

    # Helper to register hooks and capture spikes
    def make_hook(name):
        def hook(module, input, output):
            # Extract spike tensor (if output is e.g. (spikes, mem))
            out_t = extract_tensor_from_output(output)
            if out_t is None:
                # nothing to record
                return

            # move to CPU and detach
            out_t = out_t.detach().cpu()

            # Check for NaNs/Infs and warn
            if torch.isnan(out_t).any() or torch.isinf(out_t).any():
                print(f"Warning: NaN/Inf in output of layer {name}. NaN count: "
                      f"{torch.isnan(out_t).sum().item()}, Inf count: {torch.isinf(out_t).sum().item()}")

            # Count spikes: treat >0 as spike (works for binary or float spikes)
            # If shape is [T, B, C, ...], reduce appropriately: we compute mean over all elements
            spike_mask = (out_t > 0).float()
            mean_spike_rate = spike_mask.mean().item()

            # store spike activity per layer
            if name not in layer_spike_activity:
                layer_spike_activity[name] = []
            layer_spike_activity[name].append(mean_spike_rate)
        return hook

    # Register forward hooks for modules likely to output spikes
    hooks = []
    for name, module in model.named_modules():
        # Iterates through all submodules in the network
        if "li" in name.lower() or "spike" in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch_idx, (x, _, y) in enumerate(tqdm(dataloader, desc="Testing spike activity")):
            x = x.to(device)
            # forward pass (hooks will record)
            _ , _, _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute average per layer (only for layers that recorded values)
    mean_activity = {}
    for name, vals in layer_spike_activity.items():
        # Filter out potential NaNs in recorded rates (should not happen, but be safe)
        vals = [v for v in vals if not (np.isnan(v) or np.isinf(v))]
        if len(vals) == 0:
            continue
        mean_activity[name] = float(np.mean(vals))

    if len(mean_activity) == 0:
        print("No spike activity recorded for any layer (empty mean_activity). "
              "Check that hooks matched correct modules and that outputs contain tensors.")
        overall_mean = float("nan")
    else:
        overall_mean = float(np.mean(list(mean_activity.values())))

    print("\n===== Spike Activity per Layer =====")
    for name, rate in mean_activity.items():
        print(f"{name:<30}: {rate:.6f}")

    print(f"\nOverall mean spike activity (avg. over layers): {overall_mean}")

    # Visualization (bar plot)
    if visualize and len(mean_activity) > 0:
        #plt.figure(figsize=(8, 4))
        names = list(mean_activity.keys())
        rates = [mean_activity[n] for n in names]
        plt.bar(names, rates)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Mean Spike Rate")
        plt.title("Spike Activity per Layer")
        plt.show()

    return mean_activity, overall_mean