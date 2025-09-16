##############################################
# All functions needed to implement the SNN
##############################################

import torch
from torch import nn
import snntorch as snn
from snntorch import functional as SF



##################################################
# Class to implement the SNN with 1 hidden layer
##################################################
class SNN_1(nn.Module):

    def __init__(self, input_dim, hidden, output_dim, timesteps, surr_grad, learn_thr):
        super().__init__()

        self.input_dim = input_dim        
        self.hidden = hidden
        self.output_dim = output_dim
        self.timesteps = timesteps
        self.spike_grad = surr_grad
        self.learn_thr = learn_thr

        # layer 1 shared
        self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden[0])
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
    


###################################################
# Class to implement the SNN with 2 hidden layers
###################################################
class SNN_2(nn.Module):

    def __init__(self, input_dim, hidden, output_dim, timesteps, surr_grad, learn_thr):
        super().__init__()

        self.input_dim = input_dim        
        self.hidden = hidden
        self.output_dim = output_dim
        self.timesteps = timesteps
        self.spike_grad = surr_grad
        self.learn_thr = learn_thr

        # layer 1 shared
        self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden[0])
        self.lif_in = snn.Leaky(beta=torch.rand(self.hidden[0]), threshold=torch.rand(self.hidden[0]),
                                learn_beta=True, learn_threshold=self.learn_thr, spike_grad=self.spike_grad)

        # layer 2 shared
        self.fc_hidden = torch.nn.Linear(in_features=self.hidden[0], out_features=self.hidden[1])
        self.lif_hidden = snn.Leaky(beta=torch.rand(self.hidden[1]), threshold=torch.rand(self.hidden[1]),
                                    learn_beta=True, learn_threshold=self.learn_thr, spike_grad=self.spike_grad)

        # layer 3 shared
        self.fc_hidden1 = torch.nn.Linear(in_features=self.hidden[1], out_features=self.hidden[2])
        self.lif_hidden1 = snn.Leaky(beta=torch.rand(self.hidden[2]), threshold=torch.rand(self.hidden[2]),
                                    learn_beta=True, learn_threshold=self.learn_thr, spike_grad=self.spike_grad)
        
        # layer to output frequencies
        self.fc_out_freq = nn.Linear(in_features=self.hidden[2], out_features=output_dim)
        self.li_out_freq = snn.Leaky(beta=torch.rand(output_dim), threshold=1.0, learn_beta=True,
                                    spike_grad=self.spike_grad, reset_mechanism="none")
        
        # layer to output amplitudes
        self.fc_out_amp = nn.Linear(in_features=self.hidden[2], out_features=output_dim)
        self.li_out_amp = snn.Leaky(beta=torch.rand(output_dim), threshold=1.0, learn_beta=True,
                                    spike_grad=self.spike_grad, reset_mechanism="none")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_3 = self.lif_hidden1.init_leaky()
        mem_f = self.li_out_freq.init_leaky()
        mem_a = self.li_out_amp.init_leaky()
    
        for step in range(self.timesteps):
            x_timestep = x[:, :, step]

            cur_in = self.fc_in(x_timestep)
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            cur_hidden = self.fc_hidden(spk_in)
            spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

            cur_hidden1 = self.fc_hidden1(spk_hidden)
            spk_hidden1, mem_3 = self.lif_hidden(cur_hidden1, mem_3)

            cur_out_f = self.fc_out_freq(spk_hidden1)
            spk_output_f, mem_f = self.li_out_freq(cur_out_f, mem_f)

            cur_out_a = self.fc_out_amp(spk_hidden1)
            spk_output_a, mem_a = self.li_out_amp(cur_out_a, mem_a)

            frequencies = self.sigmoid(mem_f)
            amplitudes = self.sigmoid(mem_a)

        return frequencies, amplitudes
    

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