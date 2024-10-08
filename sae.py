import torch
from torch import nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

######################################################
# Class to implement the SNN encoder with linear layer
######################################################
class sae_lin(nn.Module):

    def __init__(self, input_dim, surr_grad, learn_beta):
        super(sae_lin, self).__init__()

        self.input_dim = input_dim        
        self.surr_grad = surr_grad
        self.learn_beta = learn_beta

        self.encoder = nn.Linear(in_features=self.input_dim, out_features=self.input_dim)
        
        self.lif_code = snn.Leaky(beta=torch.rand(self.input_dim), threshold=1.0, 
                              learn_beta=self.learn_beta, spike_grad=self.surr_grad)
                                
        self.decoder = nn.Linear(in_features=self.input_dim, out_features=self.input_dim)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):

        mem_enc = self.lif_code.init_leaky()
       
        # record spikes and membrane potential
        spk_rec = []
        dec_rec = []

        for step in range(x.shape[2]): # n. timesteps = n. windows

            if len(x.shape) == 5:
                x_tmstp = torch.reshape(x[:, :, step, :, :], (x.shape[0], -1)) # ~ [batch, 2*10*64]
            else:
                print(f"Step {step}, x shape {x.shape}")

            encoded = self.encoder(x_tmstp)
            spk_enc, mem_enc = self.lif_code(encoded, mem_enc)

            decoded = self.sigmoid(self.decoder(spk_enc))

            spk_rec.append(spk_enc)
            dec_rec.append(decoded)

        print(spk_enc.shape)
        print(decoded.shape)
        return torch.stack(spk_rec, dim=0), torch.stack(dec_rec, dim=0)
    

#####################################################
# Class to implement the SNN encoder with conv layer
#####################################################
class sae_conv(nn.Module):

    def __init__(self, input_dim, channels, surr_grad, learn_beta):
        super(sae_conv, self).__init__()

        self.input_dim = input_dim  
        self.channels = channels      
        self.surr_grad = surr_grad
        self.learn_beta = learn_beta

        # layer 1 
        self.fc = nn.Conv3d(2, self.channels, self.kernel_size,
                            stride=self.stride, padding='same'),
        self.bn = nn.BatchNorm3d(num_features = self.channels),
        self.lif = snn.Leaky(beta=torch.rand(self.channels*10*64), 
                             threshold=torch.rand(self.channels*10*64),
                             learn_beta=self.learn_beta, spike_grad=self.surr_grad)
        
        self.fc1 = nn.Conv3d(self.channels, 2, self.kernel_size,
                            stride=self.stride, padding='same'),
        self.bn1 = nn.BatchNorm3d(num_features = 2),
        self.lif1 = snn.Leaky(beta=torch.rand(2*10*64), threshold=1.0,
                             learn_beta=self.learn_beta, spike_grad=self.surr_grad)
        

    def forward(self, x):

        mem = self.lif.init_leaky()
        mem1 = self.lif1.init_leaky()
        
        # Record the final layer
        spike_rec = []
        mem_rec = []

        for step in range(x.shape[2]): # n. timesteps = n. windows

            x_tmstp = torch.reshape(x[:, :, step, :, :], (x.shape[0], -1)) # ~ [batch, 2*10*64]
            
            cur = self.fc(x_tmstp) # ~ [batch, 16]
            cur = self.bn(cur)
            spike, mem = self.lif(cur, mem)

            cur1 = self.fc1(spike)
            cur1 = self.bn1(cur1)
            spike1, mem1 = self.lif1(cur1, mem1)

            spike_rec.append(spike1)
            mem_rec.append(mem1)

        return torch.stack(spike_rec, dim=0)#, torch.stack(mem_rec, dim=0)