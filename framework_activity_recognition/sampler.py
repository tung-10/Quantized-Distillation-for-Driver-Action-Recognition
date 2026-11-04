from typing import Callable
from framework_activity_recognition.dataset import FileBasedDataset

import pandas as pd
import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, FileBasedDataset):
            return dataset.annotations_original
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class OversampleMinorityClassSampler(torch.utils.data.sampler.Sampler):
    """Oversample all classes (except class 0) by a given multiplier.

    Arguments:
        dataset: the dataset to sample from
        oversample_factor: how many times to replicate minority class samples (default: 5)
        skip_classes: set of class labels to NOT oversample (default: {0})
        callback_get_label: optional callback(dataset) -> list of labels
    """

    def __init__(
        self,
        dataset,
        oversample_factor: int = 5,
        skip_classes: set = None,
        callback_get_label: Callable = None,
    ):
        self.callback_get_label = callback_get_label
        self.oversample_factor = oversample_factor
        self.skip_classes = {0} if skip_classes is None else set(skip_classes)

        labels = self._get_labels(dataset)

        # Build the oversampled index list
        all_indices = []
        for idx, label in enumerate(labels):
            if label in self.skip_classes:
                # class 0 (or any skip_classes): keep exactly once
                all_indices.append(idx)
            else:
                # minority classes: repeat `oversample_factor` times
                all_indices.extend([idx] * self.oversample_factor)

        self.all_indices = all_indices

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, FileBasedDataset):
            return dataset.annotations_original
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        # Shuffle each epoch to avoid ordering bias
        perm = torch.randperm(len(self.all_indices)).tolist()
        return (self.all_indices[i] for i in perm)

    def __len__(self):
        return len(self.all_indices)