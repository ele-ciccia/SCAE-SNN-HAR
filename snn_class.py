import torch
from torch import nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import utils

###############################################################
# Class to implement the SNN classifier starting from raw data
###############################################################
class snn_classifier(nn.Module):

    def __init__(self, input_dim, hidden, timesteps, n_classes, 
                 surr_grad, learn_thr, learn_beta):
        super(snn_classifier, self).__init__()

        self.input_dim = input_dim        
        self.hidden = hidden
        self.timesteps = timesteps
        self.n_classes = n_classes
        self.surr_grad = surr_grad
        self.learn_thr = learn_thr
        self.learn_beta = learn_beta

        # layer input
        self.fc_in = nn.Linear(in_features=input_dim[-1], out_features=self.hidden[0])
        self.lif_in = snn.Leaky(beta=torch.rand(self.hidden[0]), 
                                threshold=torch.rand(self.hidden[0]),
                                learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                spike_grad=self.surr_grad)

        # layer hidden
        self.fc_hidden = nn.Linear(in_features=self.hidden[0], out_features=self.hidden[1])
        self.lif_hidden = snn.Leaky(beta=torch.rand(self.hidden[1]), 
                                    threshold=torch.rand(self.hidden[1]),
                                    learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                    spike_grad=self.surr_grad)
        
        # layer output
        self.fc_out = nn.Linear(in_features=2*10*self.hidden[1], out_features=n_classes)
        self.li_out = snn.Leaky(beta=torch.rand(n_classes), threshold=1.0,
                                #learn_threshold=self.learn_thr, 
                                learn_beta=self.learn_beta,
                                spike_grad=self.surr_grad)

    def forward(self, x):

        mem_in = self.lif_in.init_leaky()
        mem_hid = self.lif_hidden.init_leaky()
        mem_out = self.li_out.init_leaky()
        
        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(self.timesteps): # n. timesteps = n. windows

            x_tmstp = x[:, :, step, :, :]
            #x_tmstp = torch.reshape(x[:, :, step, :, :], (x.shape[0], -1)) # ~ [batch, 2*10*64]

            cur_in = self.fc_in(x_tmstp) 
            spk_in, mem_in = self.lif_in(cur_in, mem_in)

            cur_hid = self.fc_hidden(spk_in) 
            spk_hid, mem_hid = self.lif_hidden(cur_hid, mem_hid)
            
            spk_hid = torch.reshape(spk_hid, (spk_hid.shape[0], -1))
            #print("--", spk_hid.shape)
            cur_out = self.fc_out(spk_hid) 
            spk_out, mem_out = self.li_out(cur_out, mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec, dim=0)
    

################################################################################
# Class to implement the SNN classifier with Conv layers starting from raw data
################################################################################
class snn_conv_classifier(nn.Module):

    def __init__(self, channels, kernel_size, hidden, timesteps,
                 n_classes, surr_grad, learn_thr, learn_beta):
        super(snn_conv_classifier, self).__init__()

        self.channels = channels        
        self.kernel_size = kernel_size
        self.hidden = hidden
        self.timesteps = timesteps
        self.n_classes = n_classes
        self.surr_grad = surr_grad
        self.learn_thr = learn_thr
        self.learn_beta = learn_beta

        # layer input 
        self.fc_in = nn.Conv2d(2, self.channels, self.kernel_size,
                               stride= 1, padding='same')
        self.bn_in = nn.BatchNorm2d(num_features = self.channels)
        self.lif_in = snn.Leaky(beta=0.5, threshold=1.0,
                                learn_beta=self.learn_beta, 
                                learn_threshold=self.learn_thr, 
                                spike_grad=self.surr_grad)      

        # layer hidden
        self.fc_hidden = nn.Linear(in_features=64, out_features=self.hidden[0])
        self.lif_hidden = snn.Leaky(beta=0.5, threshold=1.0,
                                    learn_beta=self.learn_beta, 
                                    learn_threshold=self.learn_thr, 
                                    spike_grad=self.surr_grad)
        
        # layer output
        self.fc_out = nn.Linear(in_features=self.channels*10*self.hidden[0], 
                                out_features=n_classes)
        self.li_out = snn.Leaky(beta=0.5, threshold=1.0,
                                learn_beta=self.learn_beta,
                                spike_grad=self.surr_grad)

    def forward(self, x):

        mem_in = self.lif_in.init_leaky()
        mem_hid = self.lif_hidden.init_leaky()
        mem_out = self.li_out.init_leaky()
        
        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(self.timesteps): # n. timesteps = n. windows

            x_tmstp = x[:, :, step, :, :]
            cur_in = self.bn_in(self.fc_in(x_tmstp)) 
            spk_in, mem_in = self.lif_in(cur_in, mem_in)

            cur_hid = self.fc_hidden(spk_in) 
            spk_hid, mem_hid = self.lif_hidden(cur_hid, mem_hid)
            spk_hid = torch.reshape(spk_hid, (spk_hid.shape[0], -1))

            cur_out = self.fc_out(spk_hid) 
            spk_out, mem_out = self.li_out(cur_out, mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec, dim=0)
        