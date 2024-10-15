import torch
from torch import nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

####################################################
# Class to implement the custom Heaviside function
####################################################
class HeavisideCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tau):
        ctx.save_for_backward(input)
        assert tau > 0
        return torch.where(torch.abs(input)<tau, 0, torch.sign(input)).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = nn.functional.hardtanh(grad_output)
        grad_tau = None
        return grad_input, grad_tau

class HeavisideCustom(nn.Module):
    def __init__(self, tau):
        super(HeavisideCustom, self).__init__()

    def forward(self, x, tau=.15):
        x = HeavisideCustomFunction.apply(x, tau)
        return x
    

######################################################
# Class to implement the bipolar CAE with 2 layers
######################################################    
class cae_2(nn.Module):
    def __init__(self, tau, channels, kernel_size, stride, padding):
        super(cae_2, self).__init__()

        self.tau = tau
        self.kernel_size = kernel_size
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.encoder = nn.Sequential(
                                        nn.Conv3d(2, self.channels[0], self.kernel_size,
                                                  stride= self.stride, padding='same'),
                                        nn.BatchNorm3d(num_features = self.channels[0]),
                                        nn.Tanh(),
                                        nn.Conv3d(self.channels[0], 2, self.kernel_size,
                                                  stride=self.stride, padding='same'),
                                        nn.BatchNorm3d(num_features = 2),
                                        HeavisideCustom(tau= self.tau)
                                    )

        self.decoder = nn.Sequential(
                                        nn.ConvTranspose3d(2, self.channels[0], self.kernel_size,
                                                           stride=self.stride, 
                                                           padding=self.padding),
                                        nn.BatchNorm3d(num_features = self.channels[0]),
                                        nn.Tanh(),
                                        nn.ConvTranspose3d(self.channels[0], 2, self.kernel_size,
                                                           stride=self.stride, 
                                                           padding=self.padding),
                                        nn.BatchNorm3d(num_features = 2),
                                        nn.Sigmoid()
                                    )

    def forward(self, x):

        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding


######################################################
# Class to implement the bipolar CAE with 3 layers
######################################################
class cae_3(nn.Module):
    def __init__(self, tau, channels, kernel_size, stride, padding):
        super(cae_3, self).__init__()

        self.tau = tau
        self.kernel_size = kernel_size
        self.channels = channels
        self.stride = stride
        self.padding = padding

        self.encoder = nn.Sequential(
                                        nn.Conv3d(2, self.channels[0], self.kernel_size,
                                                  stride=self.stride, padding='same'),
                                        nn.BatchNorm3d(num_features = self.channels[0]),
                                        nn.Tanh(),
                                        nn.Conv3d(self.channels[0], self.channels[1], 
                                                  self.kernel_size,
                                                  stride=self.stride, padding='same'),
                                        nn.BatchNorm3d(num_features = self.channels[1]),
                                        nn.Tanh(),
                                        nn.Conv3d(self.channels[1], 2, self.kernel_size,
                                                  stride=self.stride, padding='same'),
                                        nn.BatchNorm3d(num_features = 2),
                                        HeavisideCustom(tau= self.tau)
                                    )

        self.decoder = nn.Sequential(
                                        nn.ConvTranspose3d(2, self.channels[1], self.kernel_size,
                                                           stride=self.stride, 
                                                           padding=self.padding),
                                        nn.BatchNorm3d(num_features = self.channels[1]),
                                        nn.Tanh(),
                                        nn.ConvTranspose3d(self.channels[1], self.channels[0], 
                                                           self.kernel_size, stride=self.stride, 
                                                           padding=self.padding),
                                        nn.BatchNorm3d(num_features = self.channels[0]),
                                        nn.Tanh(),
                                        nn.ConvTranspose3d(self.channels[0], 2, self.kernel_size,
                                                           stride=self.stride, 
                                                           padding=self.padding),
                                        nn.BatchNorm3d(num_features = 2),
                                        nn.Sigmoid()
                                    )

    def forward(self, x):

        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding


##################################################
# Class to implement the SNN with 1 hidden layer
##################################################
class snn_1(nn.Module):

    def __init__(self, input_dim, hidden, timesteps, n_classes, surr_grad, learn_thr, learn_beta):
        super(snn_1, self).__init__()

        self.input_dim = input_dim        
        self.hidden = hidden
        self.timesteps = timesteps
        self.n_classes = n_classes
        self.surr_grad = surr_grad
        self.learn_thr = learn_thr
        self.learn_beta = learn_beta

        # layer 1 
        self.fc_in = nn.Linear(in_features=input_dim, out_features=self.hidden[0])
        self.lif_in = snn.Leaky(beta=torch.rand(self.hidden[0]), 
                                threshold=torch.rand(self.hidden[0]),
                                learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                spike_grad=self.surr_grad)

        # layer 2 
        self.fc_hidden = nn.Linear(in_features=self.hidden[0], out_features=self.hidden[1])
        self.lif_hidden = snn.Leaky(beta=torch.rand(self.hidden[1]), 
                                    threshold=torch.rand(self.hidden[1]),
                                    learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                    spike_grad=self.surr_grad)
        
        # layer output
        self.fc_out = nn.Linear(in_features=self.hidden[1], out_features=n_classes)
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

            x_tmstp = torch.reshape(x[:, :, step, :, :], (x.shape[0], -1)) # ~ [batch, 2*10*64]

            cur_in = self.fc_in(x_tmstp) # ~ [batch, 16]
            spk_in, mem_in = self.lif_in(cur_in, mem_in)

            cur_hid = self.fc_hidden(spk_in) # ~ [batch, 1]
            spk_hid, mem_hid = self.lif_hidden(cur_hid, mem_hid)
            
            cur_out = self.fc_out(spk_hid) # ~ [batch, num_classes]
            spk_out, mem_out = self.li_out(cur_out, mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec, dim=0)
    


###################################################
# Class to implement the SNN with 2 hidden layers
###################################################
class snn_2(nn.Module):

    def __init__(self, input_dim, hidden, n_classes, surr_grad, learn_thr, learn_beta):
        super(snn_2, self).__init__()

        self.input_dim = input_dim        
        self.hidden = hidden
        self.n_classes = n_classes
        self.surr_grad = surr_grad
        self.learn_thr = learn_thr
        self.learn_beta = learn_beta

        # layer 1 
        self.fc_in = nn.Linear(in_features=input_dim, out_features=self.hidden[0])
        self.lif_in = snn.Leaky(beta=torch.rand(self.hidden[0]), 
                                threshold=torch.rand(self.hidden[0]),
                                learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                spike_grad=self.surr_grad)

        # layer 2 
        self.fc_hidden1 = nn.Linear(in_features=self.hidden[0], out_features=self.hidden[1])
        self.lif_hidden1 = snn.Leaky(beta=torch.rand(self.hidden[1]), 
                                    threshold=torch.rand(self.hidden[1]),
                                    learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                    spike_grad=self.surr_grad)
        # layer 3
        self.fc_hidden2 = nn.Linear(in_features=self.hidden[1], out_features=self.hidden[2])
        self.lif_hidden2 = snn.Leaky(beta=torch.rand(self.hidden[2]), 
                                    threshold=torch.rand(self.hidden[2]),
                                    learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                    spike_grad=self.surr_grad)
        
        # layer output
        self.fc_out = nn.Linear(in_features=self.hidden[2], out_features=n_classes)
        self.li_out = snn.Leaky(beta=torch.rand(n_classes), threshold=1.0,learn_threshold=self.learn_thr, 
                                learn_beta=self.learn_beta,
                                spike_grad=self.surr_grad)

    def forward(self, x):

        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden1.init_leaky()
        mem_3 = self.lif_hidden2.init_leaky()
        mem_o = self.li_out.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(x.shape[2]): # n. timesteps = n. windows

            x_tmstp = torch.reshape(x[:, :, step, :, :], (x.shape[0], -1)) # ~ [batch, 2*10*64]
            
            cur_in = self.fc_in(x_tmstp) # ~ [batch, 16]
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            cur_hidden1 = self.fc_hidden1(spk_in) # ~ [batch, 1]
            spk_hidden1, mem_2 = self.lif_hidden1(cur_hidden1, mem_2)
            
            cur_hidden2 = self.fc_hidden2(spk_hidden1) # ~ [batch, 1]
            spk_hidden2, mem_3 = self.lif_hidden2(cur_hidden2, mem_3)

            cur_out = self.fc_out(spk_hidden2) # ~ [batch, num_classes]
            spk_out, mem_o = self.li_out(cur_out, mem_o)

            spk_rec.append(spk_out)
            mem_rec.append(mem_o)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec, dim=0)
    

###############################################
# Class to implement the SNN with Conv layers
###############################################
class SNN_conv(nn.Module):

    def __init__(self, input_dim, channels, kernel, hidden, output_dim, timesteps, surr_grad, learn_thr):
        super().__init__()

        self.input_dim = input_dim      
        self.kernel = kernel
        self.channels = channels  
        self.hidden = hidden
        self.output_dim = output_dim
        self.timesteps = timesteps
        self.spike_grad = surr_grad
        self.learn_thr = learn_thr

        # CNN layer 1 shared
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=self.channels[0], 
                               kernel_size=self.kernel, stride=1, dilation=1,
                               padding=0, padding_mode='zeros')
        #self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden[0])
        self.lif_in = snn.Leaky(beta=torch.rand(self.hidden[0]), threshold=torch.rand(self.hidden[0]),
                                learn_beta=True, learn_threshold=self.learn_thr, spike_grad=self.spike_grad)

        # layer 2 shared
        self.fc_hidden = torch.nn.Linear(in_features=self.hidden[0], out_features=self.hidden[1])
        self.lif_hidden = snn.Leaky(beta=torch.rand(self.hidden[1]), threshold=torch.rand(self.hidden[1]),
                                    learn_beta=True, learn_threshold=self.learn_thr, spike_grad=self.spike_grad)
        
        # layer to output frequencies
        self.fc_out_freq = nn.Linear(in_features=self.hidden[1], out_features=output_dim)
        self.li_out_freq = snn.Leaky(beta=torch.rand(output_dim), threshold=1.0, learn_beta=True,
                                    spike_grad=self.spike_grad, reset_mechanism="none")
        
        # layer to output amplitudes
        self.fc_out_amp = nn.Linear(in_features=self.hidden[1], out_features=output_dim)
        self.li_out_amp = snn.Leaky(beta=torch.rand(output_dim), threshold=1.0, learn_beta=True,
                                    spike_grad=self.spike_grad, reset_mechanism="none")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_f = self.li_out_freq.init_leaky()
        mem_a = self.li_out_amp.init_leaky()

        mem_freq = 0
        mem_amp = 0

        for step in range(self.timesteps):
            x_timestep = x[:, :, step]

            cur_in = self.fc_in(x_timestep)
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            cur_hidden = self.fc_hidden(spk_in)
            spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

            cur_out_f = self.fc_out_freq(spk_hidden)
            spk_output_f, mem_f = self.li_out_freq(cur_out_f, mem_f)

            mem_freq += mem_f

            cur_out_a = self.fc_out_amp(spk_hidden)
            spk_output_a, mem_a = self.li_out_amp(cur_out_a, mem_a)

            mem_amp += mem_a

            frequencies = self.sigmoid(mem_f)
            amplitudes = self.sigmoid(mem_a)

        return frequencies, amplitudes


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
        
        self.lif_code = snn.Leaky(beta=torch.rand(self.input_dim),threshold=1.0, 
                              learn_beta=self.learn_beta, spike_grad=self.surr_grad)
                                
        self.decoder = nn.Linear(in_features=self.input_dim, out_features=self.input_dim)
        self.sigmoid = nn.Sigmoid()
                                    
        
    def forward(self, x):

        mem_enc = self.lif_code.init_leaky()
       
        # encoding
        spk_rec = [], mem_rec = []

        for step in range(x.shape[2]): # n. timesteps = n. windows

            x_tmstp = torch.reshape(x[:, :, step, :, :], (x.shape[0], -1)) # ~ [batch, 2*10*64]
            encoded = self.encoder(x_tmstp)
            spk_enc, mem_enc = self.lif_code(encoded, mem_enc)
            
    

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