import torch
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ec_utils import (
    complex_to_real_vector1,
    get_subj_from_filename,
    get_cir,
    mD_spectrum_,
)


# ========== DATASET GENERATION CONSTANTS ==========

# Path to the raw data (folder containing directories PERSON1, PERSON2, etc.)
RAW_DATA_PATH = os.path.join(".", "data", "raw_data")

# Path to where the generated processed dataset will be saved
#DATA_PATH_V2 = os.path.join(".", "data", "processed_data")
DATA_PATH_V2 = os.path.abspath('/home/eleonora/Documents/Papers/Learned_Spike_Encoding/' \
                                'PAPER_EXTENSION/data/processed_data')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This constant determines how many range bins will be kept 
# in the processed data, from the original 110
N_KEPT_BINS = 10

ACTIVITIES = ["WALKING", 
              "RUNNING",
              "SITTING",
              "HANDS",]

SUBJECTS = [1, 2, 3, 4, 5, 6, 7]

DATAGEN_PARAMS = {"N_PASSES": 1,
                "DIST_BOUNDS": (10, 120),
                "LWIN": 64,
                "TREP": 32,
                "NWIN": 232, # corresponding to 2-3s of recording, with T=0.27ms
                "Nd": 64,  # original 64
                "BP_SEL": 0,}

# encoding of the activities in 0,1,2,3
class_enc = {a:[i] for i, a in enumerate(ACTIVITIES)}

class Sparse_MD_Dataset_V2(torch.utils.data.Dataset):
    def __init__(self, filenames, p_burst=0, seed=0):
        super(Sparse_MD_Dataset_V2, self).__init__()
        self.raw_filenames = filenames
        self.rng = np.random.default_rng(seed=seed)
        self.p_burst = p_burst

    @staticmethod
    def generate_dataset(out_folder=DATA_PATH_V2):
        _ = input(
            "You are about to regenerate the whole dataset(V2). If you are sure, press enter.")
        
        overlap = DATAGEN_PARAMS["NWIN"] // 2

        for pass_idx in range(DATAGEN_PARAMS["N_PASSES"]):
            for n in SUBJECTS:
                for activity in ACTIVITIES:
                    for f in os.listdir(f"{RAW_DATA_PATH}/PERSON{n}"):
                        # activity_idx = f.split("_")[0]
                        activity_idx = f.split(".")[0].split("_")[-1]
                        if activity in f:

                            complex_cir = get_cir(
                                f"{RAW_DATA_PATH}/PERSON{n}/{f}",
                                DATAGEN_PARAMS["DIST_BOUNDS"],)

                            complex_cir = complex_cir[:, :, DATAGEN_PARAMS["BP_SEL"]]
                            complex_cir -= complex_cir.mean(1, keepdims=True)

                            (chunks, IHT_output, full_IHT_mD,) = mD_spectrum_(
                                complex_cir,
                                DATAGEN_PARAMS["LWIN"],
                                DATAGEN_PARAMS["TREP"],
                                n_kept_bins=N_KEPT_BINS,
                            )
                            
                            # Now I have my couples (X, y) = (chunks, mD) inside a big list
                            # Save them one by one as a tuple (chunk, window_mD)

                            # Naming convention: {PASS_INDEX}_{SUBJECT}_{ACTIVITY}_{ACTIVITY_INDEX}

                            # check on the n. of windows
                            if chunks.shape[0] < DATAGEN_PARAMS["NWIN"]: # discard CIR with n. of windows < N_win
                                continue #pass

                            elif chunks.shape[0] >= DATAGEN_PARAMS["NWIN"] and \
                                chunks.shape[0] < DATAGEN_PARAMS["NWIN"] + DATAGEN_PARAMS["NWIN"]//2:
                                new_chunks = chunks[:DATAGEN_PARAMS["NWIN"]]

                                assert new_chunks.shape[0] == DATAGEN_PARAMS["NWIN"]

                                data_tuple = (new_chunks, IHT_output, full_IHT_mD[:DATAGEN_PARAMS["NWIN"]])
                                
                                with open(
                                    f"{out_folder}/{pass_idx}_{n}_{activity}_{activity_idx}.obj",
                                    "wb",
                                ) as out:
                                    pickle.dump(data_tuple, out)

                                print(f"Saved {f}")

                            else: #chunks.shape[0] >= N_win + N_win//2
                                div = chunks.shape[0] // overlap

                                for step in range(div-1):
                                    new_chunks = chunks[step*overlap:(step*overlap)+DATAGEN_PARAMS["NWIN"]] # overlap 1/2

                                    assert new_chunks.shape[0] == DATAGEN_PARAMS["NWIN"]

                                    new_full_IHT_mD = full_IHT_mD[step*overlap:(step*overlap)+DATAGEN_PARAMS["NWIN"]] # overlap 1/2

                                    data_tuple = (new_chunks, IHT_output, new_full_IHT_mD)
                                    
                                    with open(
                                        f"{out_folder}/{pass_idx}_{n}_{activity}_{activity_idx}_{step}.obj",
                                        "wb",
                                    ) as out:
                                        pickle.dump(data_tuple, out)

                                    print(f"Saved {f}")


    @staticmethod
    def make_splits(
        subjects,
        activities,
        sparse_dataset_path=DATA_PATH_V2,
        subsample_factor=1.0,
        seed=123,
        train=0.7,
        test=0.15,
        valid=0.15,
    ):
        if not train + valid + test == 1.0:
            raise Exception(f"train ({train})+valid ({valid})+test ({test}) != 1")

        if subsample_factor < 0 or subsample_factor > 1:
            raise Exception(
                f"Please input a valid subsampling factor (Current is {subsample_factor})"
            )

        rng = np.random.default_rng(seed=seed)
        all_filenames = os.listdir(sparse_dataset_path)

        # Select only wanted subjects
        all_filenames = [
            f for f in all_filenames if int(get_subj_from_filename(f)) in subjects
        ]

        # Select only wanted activities
        all_filenames = [f for f in all_filenames if any([a in f for a in activities])]

        nsamples = int(len(all_filenames) * subsample_factor)
        chosen_filenames = rng.choice(all_filenames, nsamples, replace=False)

        train_set, valid_test_set = train_test_split(
            chosen_filenames, train_size=train, random_state=seed
        )

        valid_set, test_set = train_test_split(
            valid_test_set,
            train_size=(valid / (valid + test)),
            random_state=seed,
        )

        # Repeat all the filenames in the training set containing the "RUNNING" activity
        # until they reach the same amount of samples as the "WALKING" activity
        # This is done to balance the dataset
        train_set = train_set.tolist()
        running_filenames = [f for f in train_set if "RUNNING" in f]
        walking_filenames = [f for f in train_set if "WALKING" in f]

        for _ in range(len(walking_filenames) - len(running_filenames)):
            train_set.append(rng.choice(running_filenames))

        train_set = np.array(train_set)
        return train_set, valid_set, test_set

    def generate_mask(
        self,
        size,
        p_remove,
        min_allowed_samples=3,
    ):
        if p_remove == 0:
            return torch.ones(size * 2).to(DEVICE)

        if self.rng.random() < self.p_burst:
            # Apply "bursting" sampling pattern
            chunk_mask = torch.zeros(size).to(DEVICE)

            burst_len1 = (torch.rand(size) > p_remove).int().sum()
            if burst_len1 < min_allowed_samples:
                burst_len1 = min_allowed_samples
            if burst_len1 == size:
                return torch.ones(size * 2).to(DEVICE)

            # select first start index for the burst
            start_idx1 = self.rng.integers(0, 64 - burst_len1)
            # select second start index for the burst
            chunk_mask[start_idx1 : start_idx1 + burst_len1] = 1

            # if self.rng.random() < 0.2:
            #     # apply second burst with 20% probability

            #     candidates = [
            #         c
            #         for c in range(64 - burst_len2)
            #         if c < start_idx1 - burst_len1 - 1
            #         or c > start_idx1 + burst_len1 + 1
            #     ]
            #     start_idx2 = np.random.choice(candidates)
            #     chunk_mask[start_idx2 : start_idx2 + burst_len2] = 1
            assert torch.sum(chunk_mask) >= min_allowed_samples
        else:
            # Apply uniform random sampling pattern
            chunk_mask = (torch.rand(size) > p_remove).int().to(DEVICE)

            # ensure mask has at least 3 non zero elements
            nonzeros = torch.sum(chunk_mask)
            if nonzeros < min_allowed_samples:
                # choose 3 random indices to not mask
                chunk_mask = torch.zeros(size).int().to(DEVICE)
                idxs = self.rng.choice(
                    np.arange(size),
                    min_allowed_samples,
                    replace=False,
                )
                chunk_mask[idxs] = 1

        # repeat mask two times
        chunk_mask = chunk_mask.repeat(2)
        return chunk_mask

    def __len__(self):
        return self.raw_filenames.shape[0]

    #def __getitem__(self, idx):
        # with open(
        #     os.path.join(DATA_PATH_V2, self.raw_filenames[idx]),
        #     "rb",
        # ) as file:
        #     Xy = pickle.load(file)

        # # Convert to real numbers
        # X = complex_to_real_vector(Xy[0])
        # IHT_output = torch.tensor(complex_to_real_vector(Xy[1]))
        # mD_columns = torch.tensor(Xy[2])

        # X = torch.clamp(torch.tensor(X), min=-150, max=150)

        # return X, IHT_output, mD_columns
    
    def __getitem__(self, idx):
        with open(
            os.path.join(DATA_PATH_V2, self.raw_filenames[idx]),
            "rb",
        ) as file:
            Xy = pickle.load(file)

        # Convert to real numbers
        X = complex_to_real_vector1(Xy[0])
        mD_columns = torch.tensor(Xy[2])
        Y = torch.Tensor(class_enc[self.raw_filenames[idx].split('_')[2]]).to(torch.int64)

        # X = torch.clamp(torch.tensor(X.detach()), min=-150, max=150) 
        X = torch.clamp(X.clone().detach(), min=-150, max=150)

        # Min-max normalization
        min_values, _ = torch.min(X, dim=3, keepdim=True)
        max_values, _ = torch.max(X, dim=3, keepdim=True)
        normalized_X = (X - min_values) / (max_values - min_values)

        assert normalized_X.shape[1] == 232

        return normalized_X, mD_columns, Y

if __name__ == "__main__":
    ## EXAMPLE USAGE

    # This static method generates and saves files in DATA_PATH_V2, using the raw data in RAW_DATA_PATH.
    # These files are then to be used by the Pytorch dataset in the __getitem__ method
    # NB: takes a long time to run
    os.makedirs(DATA_PATH_V2, exist_ok=True)
    Sparse_MD_Dataset_V2.generate_dataset(out_folder=DATA_PATH_V2)

    # This static method generates the train, valid and test filenames lists
    train_fnames, valid_fnames, test_fnames = Sparse_MD_Dataset_V2.make_splits(
        SUBJECTS, ACTIVITIES, sparse_dataset_path=DATA_PATH_V2
    )

    # Build dataset objects
    train_set = Sparse_MD_Dataset_V2(train_fnames)
    valid_set = Sparse_MD_Dataset_V2(valid_fnames)
    test_set = Sparse_MD_Dataset_V2(test_fnames)

    for (X, IHT_output, mD_columns) in train_set:
        # This dataset returns three elements:
        # X: the raw data,
        # IHT_output: the output of the IHT algorithm (you don't need it)
        # mD_columns: the mD spectrum
        print(X.shape, IHT_output.shape, mD_columns.shape)

        # You can plot the mD spectrum like this:
        plt.imshow(mD_columns.T, aspect="auto")
        plt.show()
        break
