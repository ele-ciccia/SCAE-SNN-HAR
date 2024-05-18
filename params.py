import torch
from snntorch import surrogate, functional as SF

params_train = {
                "epochs": 150,
                "acc_steps": None,
                "l_rate": 1e-3, #OK
                "lambda_reg": 0.3, #OK
                "alpha": 0.99, #OK
                "beta": 0.01, #OK
                "patience": 15,
}

params_cae = {
                "kernel": (1,1,5), #OK
                "feature_maps": [64,128], #OK
                "stride": 1,
                "padding": [0,0,2],
                "tau": 0.9, #OK
                "loss_fn": torch.nn.MSELoss(),    
}

params_snn = {
                "hidden_layers": [128,64], #OK
                "num_classes": 4,
                "surrogate_grad": surrogate.atan(), #OK
                "learn_thr": True, #OK
                "learn_beta": True, #OK
                "output_decoding": "rate",
                "loss_fn": SF.ce_rate_loss(), #ce_temporal_loss()
}