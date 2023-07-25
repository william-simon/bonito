import importlib
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader


class ChunkDataSet:
    def __init__(self, chunks, targets, lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
        )

    def __len__(self):
        return len(self.lengths)


def load_script(directory, name="dataset", suffix=".py", **kwargs):
    directory = Path(directory)
    filepath = (directory / name).with_suffix(suffix)
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.Loader(**kwargs)
    return loader.train_loader_kwargs(**kwargs), loader.valid_loader_kwargs(**kwargs)


def load_numpy(directory, valid_limit=None, load_training_set=True):
    """
    Returns training and validation DataLoaders for data in directory.
    """
    if load_training_set:
        train_data = load_numpy_datasets(directory=directory)
    else:
        train_data = (np.zeros(1),np.zeros(1),np.zeros(1))
    if os.path.exists(os.path.join(directory, 'validation')):
        valid_data = load_numpy_datasets(limit=valid_limit,
            directory=os.path.join(directory, 'validation')
        )
    else:
        print("[validation set not found: splitting training set]")
        if valid_limit is None:
            split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
        else:
            split = max(0, len(train_data[0]) - valid_limit)
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    train_loader_kwargs = {"dataset": ChunkDataSet(*train_data), "shuffle": True}
    valid_loader_kwargs = {"dataset": ChunkDataSet(*valid_data), "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs


def load_numpy_datasets(limit=None, directory=None):
    """
    Returns numpy chunks, targets and lengths arrays.
    """
    if directory is None:
        directory = default_data

    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')

    indices = os.path.join(directory, "indices.npy")

    if os.path.exists(indices):
        idx = np.load(indices, mmap_mode='r')
        idx = idx[idx < lengths.shape[0]]
        if limit:
            idx = idx[:limit]
        return chunks[idx, :], targets[idx, :], lengths[idx]

    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]

    return np.array(chunks), np.array(targets), np.array(lengths)
