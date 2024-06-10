import torch
from snntorch import surrogate, functional as SF

params_train = {
                "epochs": 150,
                "acc_steps": None,
                "l_rate": 1e-4, 
                "lambda_reg": 0.3, 
                "alpha": 0.85, 
                "beta": 0.15, 
                "patience": 20,
}

params_cae = {
                "kernel": (1,1,5), 
                "feature_maps": [128,128], 
                "stride": 1,
                "padding": [0,0,2],
                "tau": 0.1, 
                "loss_fn": torch.nn.MSELoss(),    
}

params_snn = {
                "hidden_layers": [128,64],
                "num_classes": 4,
                "surrogate_grad": surrogate.fast_sigmoid(), 
                "learn_thr": True, 
                "learn_beta": True, 
                "output_dec": "rate",
                "loss_fn": SF.ce_rate_loss(), 
}