import torch
from torch import nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

N_WIN = 232

############################
# Custom Heaviside function
############################
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
    
####################
# CAE with 2 layers
####################
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

####################
# CAE with 3 layers
####################
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

###########################################
# SNN for classif. - 1 hidden Linear layer
###########################################
class snn_1(nn.Module):

    def __init__(self, input_shape, hidden, timesteps, kernel, stride,
                 n_classes, surr_grad, learn_thr, learn_beta):
        super(snn_1, self).__init__()

        self.input_shape = input_shape        
        self.hidden = hidden
        self.timesteps = timesteps
        self.kernel = kernel
        self.stride = stride
        self.n_classes = n_classes
        self.surr_grad = surr_grad
        self.learn_thr = learn_thr
        self.learn_beta = learn_beta
        self.input_dim = int((self.input_shape[-1]-self.kernel[-1])/self.stride[-1]+1)*\
                         int((self.input_shape[-2]-self.kernel[-2])/self.stride[-2]+1)
        self.in_feat_dim = self.input_shape[0]*self.hidden[1]*\
                           int((N_WIN//self.timesteps-self.kernel[-3])/self.stride[-3]+1)

        # layer input
        self.avg_pool = nn.AvgPool3d(self.kernel, self.stride)
        self.fc_in = nn.Linear(in_features=self.input_dim, out_features=self.hidden[0])
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
        self.fc_out = nn.Linear(in_features=self.in_feat_dim, out_features=self.n_classes)
        self.li_out = snn.Leaky(beta=torch.rand(self.n_classes), threshold=1.0,
                                #learn_threshold=self.learn_thr, 
                                learn_beta=self.learn_beta,
                                spike_grad=self.surr_grad)

    def forward(self, x):

        mem_in = self.lif_in.init_leaky()
        mem_hid = self.lif_hidden.init_leaky()
        mem_out = self.li_out.init_leaky()
        
        spk_rec = []; mem_rec = []
        chunk_size = N_WIN // self.timesteps

        for step in range(self.timesteps): 
            start = step * chunk_size; end = start + chunk_size
            x = x.squeeze()           
            x_tmstp = x[:, :, start:end, :, :] # extract the portion of windows
            x_tmstp = self.avg_pool(x_tmstp) # downsample the dimensions
            # reshape the tensor
            shape = x_tmstp.shape
            x_reshaped = x_tmstp.view(shape[0], shape[1]*shape[2], -1)

            # first layer
            cur_in = self.fc_in(x_reshaped) 
            spk_in, mem_in = self.lif_in(cur_in, mem_in)

            # second layer
            cur_hid = self.fc_hidden(spk_in) # ~ [batch, 1]
            del cur_in, spk_in
            spk_hid, mem_hid = self.lif_hidden(cur_hid, mem_hid)

            # flatten - keeping the batch size
            spk_hid = spk_hid.view(spk_hid.shape[0], -1)

            # third layer
            cur_out = self.fc_out(spk_hid) # ~ [batch, num_classes]
            del cur_hid, spk_hid
            spk_out, mem_out = self.li_out(cur_out, mem_out)

            spk_rec.append(spk_out)
            #mem_rec.append(mem_out)
            del cur_out, spk_out

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec, dim=0)
    
############################################
# SNN for classif. - 2 hidden Linear layers
############################################
class snn_2(nn.Module):

    def __init__(self, input_dim, hidden, n_classes, 
                 surr_grad, learn_thr, learn_beta):
        super(snn_2, self).__init__()

        self.input_dim = input_dim        
        self.hidden = hidden
        self.n_classes = n_classes
        self.surr_grad = surr_grad
        self.learn_thr = learn_thr
        self.learn_beta = learn_beta

        # layer input
        self.fc_in = nn.Linear(in_features=input_dim, out_features=self.hidden[0])
        self.lif_in = snn.Leaky(beta=torch.rand(self.hidden[0]), 
                                threshold=torch.rand(self.hidden[0]),
                                learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                spike_grad=self.surr_grad)

        # layer hidden 1 
        self.fc_hidden1 = nn.Linear(in_features=self.hidden[0], out_features=self.hidden[1])
        self.lif_hidden1 = snn.Leaky(beta=torch.rand(self.hidden[1]), 
                                    threshold=torch.rand(self.hidden[1]),
                                    learn_beta=self.learn_beta, learn_threshold=self.learn_thr, 
                                    spike_grad=self.surr_grad)
        # layer 3hidden 2
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

        spk_rec = []
        mem_rec = []

        for step in range(x.shape[2]): # n. timesteps = n. windows

            x_tmstp = torch.reshape(x[:, :, step, :, :], (x.shape[0], -1)) 
            
            cur_in = self.fc_in(x_tmstp) 
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            cur_hidden1 = self.fc_hidden1(spk_in) 
            spk_hidden1, mem_2 = self.lif_hidden1(cur_hidden1, mem_2)
            
            cur_hidden2 = self.fc_hidden2(spk_hidden1) 
            spk_hidden2, mem_3 = self.lif_hidden2(cur_hidden2, mem_3)

            cur_out = self.fc_out(spk_hidden2) 
            spk_out, mem_o = self.li_out(cur_out, mem_o)

            spk_rec.append(spk_out)
            mem_rec.append(mem_o)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec, dim=0)
    
#######################################
# SNN for classif. - 1 Conv + 2 Linear
#######################################
class snn_conv(nn.Module):

    def __init__(self, input_dim, channels, kernel, hidden, 
                 output_dim, timesteps, surr_grad, learn_thr):
        super(snn_conv, self).__init__()

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

################################
# SAE with 2 layers (enc + dec)    # former sae_lin
################################
class slae_1(nn.Module):  

    def __init__(self, input_dim, surr_grad, learn_beta, timesteps):
        super(slae_1, self).__init__()

        self.input_dim = input_dim        
        self.surr_grad = surr_grad
        self.learn_beta = learn_beta
        self.timesteps = timesteps

        self.encoder = nn.Linear(in_features=input_dim, out_features=input_dim)

        self.lif_code = snn.Leaky(beta=torch.rand(input_dim), threshold=1.0, 
                              learn_beta=self.learn_beta, spike_grad=self.surr_grad)
                                
        self.decoder = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):

        mem_enc = self.lif_code.init_leaky()
       
        # record spike ecoding and decoding
        spk_rec = []
        dec_rec = []

        for step in range(self.timesteps): # n. timesteps = n. windows

            x_tmstp = x[:, :, step, :, :] # select the current timestamp 

            encoded = self.encoder(x_tmstp)
            spk_enc, mem_enc = self.lif_code(encoded, mem_enc)

            decoded = self.sigmoid(self.decoder(spk_enc))

            spk_rec.append(spk_enc)
            dec_rec.append(decoded)

        # stack all the recording of encoded and decoded on dim 2 to obtain shape [8, 2, 232, 10, 64]
        stacked_spk_rec = torch.stack(spk_rec, dim=2)
        stacked_dec_rec = torch.stack(dec_rec, dim=2)

        return stacked_spk_rec, stacked_dec_rec
                       
################################
# SAE with 4 layers (enc + dec)    # former sae_lin2
################################
class slae_2(nn.Module):

    def __init__(self, input_dim, hidden_dim, surr_grad, learn_beta, timesteps):
        super(slae_2, self).__init__()

        self.input_dim = input_dim   
        self.hidden_dim = hidden_dim     
        self.surr_grad = surr_grad
        self.learn_beta = learn_beta
        self.timesteps = timesteps

        # encoder
        self.enc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.lif_enc1 = snn.Leaky(beta=torch.rand(hidden_dim), threshold=1.0, 
                              learn_beta=self.learn_beta, spike_grad=self.surr_grad)
        self.enc2 = nn.Linear(in_features=hidden_dim, out_features=input_dim)

        # code
        self.lif_code = snn.Leaky(beta=torch.rand(input_dim), threshold=1.0, 
                              learn_beta=self.learn_beta, spike_grad=self.surr_grad)

        # decoding                        
        self.dec1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.lif_dec1 = snn.Leaky(beta=torch.rand(hidden_dim), threshold=1.0, 
                              learn_beta=self.learn_beta, spike_grad=self.surr_grad)
        self.dec2 = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):

        mem_enc = self.lif_enc1.init_leaky()
        mem_code = self.lif_code.init_leaky()
        mem_dec = self.lif_dec1.init_leaky()

        # record spike ecoding and decoding
        spk_rec = []
        dec_rec = []

        for step in range(self.timesteps): # n. timesteps = n. windows

            x_tmstp = x[:, :, step, :, :] # select the current timestamp 

            cur_in = self.enc1(x_tmstp)
            spk_in, mem_enc = self.lif_enc1(cur_in, mem_enc)

            cur_code = self.enc2(spk_in)
            spk_code, mem_code = self.lif_code(cur_code, mem_code)

            cur_dec1 = self.dec1(spk_code)
            spk_dec, mem_dec = self.lif_dec1(cur_dec1, mem_dec)

            decoded = self.sigmoid(self.dec2(spk_dec))

            spk_rec.append(spk_code)
            dec_rec.append(decoded)

        # stack all the recording of encoded and decoded on dim 2 to obtain shape [8, 2, 232, 10, 64]
        stacked_spk_rec = torch.stack(spk_rec, dim=2)
        stacked_dec_rec = torch.stack(dec_rec, dim=2)

        return stacked_spk_rec, stacked_dec_rec
        
##################################
# SAE - 4 Conv layers (enc + dec)    # former sae_conv
##################################
class scae(nn.Module):

    def __init__(self, channels, kernel_size, stride, beta, threshold, 
                 surr_grad, learn_beta, learn_threshold, timesteps):
        super(scae, self).__init__()

        self.channels = channels  
        self.kernel_size = kernel_size
        self.stride = stride    
        self.beta = beta
        self.threshold = threshold
        self.surr_grad = surr_grad
        self.learn_beta = learn_beta
        self.learn_thr = learn_threshold
        self.timesteps = timesteps

        ### Encoder 
        # Layer 1
        self.enc_conv1 = nn.Conv3d(2, self.channels[0], self.kernel_size,
                                    stride=self.stride, padding='same')
        self.enc_bn1 = nn.BatchNorm3d(num_features = self.channels[0])
        self.enc_lif1 = snn.Leaky(beta=beta, threshold=threshold,
                                    learn_beta=self.learn_beta, 
                                    learn_threshold = self.learn_thr,
                                    spike_grad=self.surr_grad)
        # Layer 2
        self.enc_conv2 = nn.Conv3d(self.channels[0], 2, self.kernel_size,
                                    stride=self.stride, padding='same')
        self.enc_bn2 = nn.BatchNorm3d(num_features = 2)
        self.enc_lif2 = snn.Leaky(beta=beta, threshold=1.0,
                                    learn_beta=self.learn_beta, 
                                    spike_grad=self.surr_grad)

        ### Decoder
        # Layer 1
        self.dec_conv1 = nn.ConvTranspose3d(2, self.channels[0], self.kernel_size,
                                            stride=self.stride, 
                                            padding=[0,0,2])
        self.dec_bn1 = nn.BatchNorm3d(num_features = self.channels[0])
        self.dec_lif1 = snn.Leaky(beta=beta, threshold=threshold,
                                    learn_beta=self.learn_beta, 
                                    learn_threshold = self.learn_thr,
                                    spike_grad=self.surr_grad)
        # Layer 2
        self.dec_conv2 = nn.ConvTranspose3d(self.channels[0], 2, self.kernel_size,
                                            stride=self.stride, 
                                            padding=[0,0,2])
        self.dec_bn2 = nn.BatchNorm3d(num_features = 2)
        self.sigmoid = nn.Sigmoid()                           

    def forward(self, x):
        #utils.reset(self.enc_lif1); utils.reset(self.enc_lif2) 
        #utils.reset(self.dec_lif1)
        mem_in = self.enc_lif1.init_leaky()
        mem_enc = self.enc_lif2.init_leaky()
        mem_dec = self.dec_lif1.init_leaky()
        
        # Record the final layer
        spk_rec_enc = []; spk_rec_dec = []
        mem_rec_enc = []; mem_rec_dec = []

        assert x.shape[2] % self.timesteps == 0
        chunk_size = x.shape[2] // self.timesteps

        for step in range(self.timesteps): 
            start = step * chunk_size; end = start + chunk_size
            x_tmstp = x[:, :, start:end, :, :]
            
            cur_enc1 = self.enc_bn1(self.enc_conv1(x_tmstp))
            spk_in, mem_in = self.enc_lif1(cur_enc1, mem_in)
            
            cur_enc2 = self.enc_bn2(self.enc_conv2(spk_in))         
            spike_enc, mem_enc = self.enc_lif2(cur_enc2, mem_enc)

            spk_rec_enc.append(spike_enc)
            mem_rec_enc.append(mem_enc)

            cur_dec1 = self.dec_bn1(self.dec_conv1(spike_enc))
            spk_out, mem_dec = self.dec_lif1(cur_dec1, mem_dec)
            x_dec = self.sigmoid(self.dec_bn2(self.dec_conv2(spk_out)))

            spk_rec_dec.append(x_dec)
            mem_rec_dec.append(mem_dec)

            encoding = torch.stack(spk_rec_enc, dim=0)
            decoding = torch.stack(spk_rec_dec, dim=0)

        return encoding, decoding
