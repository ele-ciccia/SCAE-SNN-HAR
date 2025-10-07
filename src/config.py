import os
import torch
from snntorch import surrogate

# ===================
# DATASET PARAMETERS
# ===================

# Path to the raw data (folder containing directories PERSON1, PERSON2, etc.)
RAW_DATA_PATH = os.path.join(".", "data", "raw_data")

# Path to where the generated processed dataset will be saved
#DATA_PATH_V2 = os.path.join(".", "data", "processed_data")
DATA_PATH_V2 = os.path.join(".", "data", "processed_data")

N_KEPT_BINS = 10 # how many range bins ARE kept from the original 110
N_WIN = 232 # corresponding to 2-3s of recording, with T=0.27ms
WIN_LEN = 64 # window length
INP_SHAPE = [2, 232, 10, 64]

ACTIVITIES = ["WALKING", 
              "RUNNING",
              "SITTING",
              "HANDS",]

CLASS_ENC = {a:[i] for i, a in enumerate(ACTIVITIES)} # encoding of activities
CLASS_DEC = {int(v[0]): k for k, v in CLASS_ENC.items()}

SUBJECTS = [1, 2, 3, 4, 5, 6, 7]

DATAGEN_PARAMS = {"N_PASSES": 1,
                "DIST_BOUNDS": (10, 120),
                "LWIN": WIN_LEN,
                "TREP": 32,
                "NWIN": N_WIN, 
                "Nd": 64,  
                "BP_SEL": 0,}

# ===================
# TRAINING PARAMETERS
# ===================
BATCH_SIZE = 8
SURROGATE_FN = surrogate.atan()

# ====================
# DEVICE CONFIGURATION
# ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")