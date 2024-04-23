import torch
from snntorch import surrogate, functional as SF

params_train = {
                "epochs": 25,
                "acc_steps": None,
                "l_rate": 1e-3,
                "lambda_reg": 0.3,
                "alpha": 0.99,
                "beta": 0.01,
                "patience": 100,
}

params_cae = {
                "kernel": (1,1,5),
                "feature_maps": [64,128],
                "stride": 1,
                "padding": [0,0,2],
                "tau": 0.9,
                "loss_fn": torch.nn.MSELoss(),    
}

params_snn = {
                "hidden_layers": [16,16,1],
                "num_classes": 4,
                "surrogate_grad": surrogate.atan(),
                "learn_thr": True,
                "learn_beta": True,
                "output_decoding": "rate",
                "loss_fn": SF.ce_rate_loss(), #ce_temporal_loss()
}