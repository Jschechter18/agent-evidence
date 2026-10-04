from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from mas_sae.data.activation_store import ActivationStore


class ActivationDataset(Dataset):
    def __init__(
        self,
        split: str,
        location: str | Path = "data/activations",
    ):
        self.activation_store = ActivationStore(location)
        self.split = split

        self.activations: torch.Tensor = self.activation_store.load_activations(split)

    def __len__(self) -> int:
        return self.activations.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


def create_sae_dataloader(
    batch_size: int,
    split: str,
    num_workers: int,
    shuffle: bool = False,
    location: str | Path | None = None,
) -> DataLoader:
    """Dataloader helper function to construct and return the dataloader for any given split.

    Parameters
    ----------
    batch_size : int
        Number of activations per batch. Multiples of 2 are ideal.
    split : str
        Which split of the dataset to load (e.g., "train", "val", "test").
    num_workers : int
        Number of cpu cores to load the data.
    shuffle : bool, optional
        Whether to shuffle the dataset, by default False.
    location : str | Path | None, optional
        Directory containing <split>.pt, for example
        data/activations/pilot_12q/layer_17.
        Defaults to data/activations.

    Returns
    -------
    DataLoader
        The constructed DataLoader for the specified split.
    """
    if location is None:
        dataset = ActivationDataset(split=split)
    else:
        dataset = ActivationDataset(split=split, location=location)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
