import torch
from torch.utils.data import DataLoader
from config import DATA_PATH, DEVICE, BATCH_SIZE
from utils_fn import binary_encoding

def load_datasets():
    """Load train, validation, and test datasets."""

    train_dataset = torch.load(f'data/train_dataset_split_subj.pt', weights_only=False)
    val_dataset = torch.load('data/val_dataset_split_subj.pt', weights_only=False)
    test_dataset = torch.load('data/test_dataset_split_subj.pt', weights_only=False)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE):
    """Return dataloaders for train, validation, and test sets."""

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    return train_loader, val_loader, test_loader


def describe_data(train_loader, train_dataset, val_dataset, test_dataset):
    """Print a summary of dataset and dataloader shapes."""
    print(f"Using {DEVICE} device")
    x = next(iter(train_loader))[0]
    print(f"\nShape of each element in the dataloader: {x.shape}")
    print(f"Number of elements in train - valid - test set: "
          f"{len(train_dataset)} - {len(val_dataset)} - {len(test_dataset)}")
    
def get_dataloaders_delta(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE):
    """Return delta encoded dataloaders for train, validation, and test sets."""
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=1,
                                collate_fn=binary_encoding,
                                )

    val_loader = DataLoader(
                                val_dataset,
                                batch_size=BATCH_SIZE,
                                collate_fn=binary_encoding,
                                )

    test_loader = DataLoader(
                                test_dataset,
                                batch_size=1,
                                collate_fn=binary_encoding,
                                )
    return train_loader, val_loader, test_loader