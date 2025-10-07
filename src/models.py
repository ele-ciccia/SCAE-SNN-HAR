from torch import nn

# ============================================
# Generic autoencoder + SNN for classification
# ============================================
class AE(nn.Module):
    def __init__(self, autoencoder, snn):
        super().__init__()
        self.autoencoder = autoencoder
        self.snn = snn

    def forward(self, x):
        encoded, decoded = self.autoencoder(x)
        clss = self.snn(encoded)  
        n_spikes = encoded.abs().mean()
        return n_spikes, decoded, clss

# ===========================================================
# SNN for classifying directly from CIR or delta encoded data    
# ===========================================================  
class SNN_CLSFF(nn.Module):
    def __init__(self, snn):
        super().__init__()
        self.snn = snn

    def forward(self, x):
        clss = self.snn(x)  
        n_spikes = x.abs().mean()
        return n_spikes, x, clss