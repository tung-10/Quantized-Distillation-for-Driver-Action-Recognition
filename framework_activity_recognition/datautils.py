import torch
import torchvision
import torchvision.transforms as transforms
import os
from framework_activity_recognition.videotransform import RandomSelect, RandomCrop, RandomHorizontalFlip, \
    normalizeColorInputZeroCenterUnitRange, CenterCrop, ToTensor, SlidingWindowSample
from framework_activity_recognition.parser import parseDriveNActSplitFiles, \
    parseDriveNActTestSplitFiles, parseCustomSplitFiles, parseCustomTestSplitFiles
from framework_activity_recognition.dataset import VideoFileBasedDataset, DriveNActDataset, CustomDataset


# ─────────────────────────────────────────────
# Drive&Act prepare functions (unchanged)
# ─────────────────────────────────────────────

def prepare_drivenact(config_file):
    """
    Construct Drive&Act train and validation Dataset instance based on the given configuration file
    Arguments:
        config_file: configuration file to construct Dataset instance
    """
    split_nr      = config_file["data"]["split_nr"]
    folder_splits = config_file["data"]["folder_splits"]
    data_split    = config_file["data"]["data_path"]
    views         = config_file["data"]["views"]
    new_suffix    = config_file["data"]["new_suffix"]
    n_frame       = config_file["data"]["n_frame"]
    frame_size    = config_file["data"]["frame_size"]

    train_df, test_df = parseDriveNActSplitFiles(split_nr, folder_splits, data_split, views, new_suffix)

    transform_training = torchvision.transforms.Compose([
        RandomSelect(n=n_frame),
        RandomCrop(height=frame_size, width=frame_size),
        RandomHorizontalFlip(),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    transform_test = torchvision.transforms.Compose([
        RandomSelect(n=n_frame),
        CenterCrop(height=frame_size, width=frame_size),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    annotations_train = train_df['activity'].tolist()
    annotations_test  = test_df['activity'].tolist()

    dataset_train        = DriveNActDataset(train_df, annotations_train, None, transform_training)
    annotation_converter = dataset_train.__get_annotation_converter__()
    dataset_test         = DriveNActDataset(test_df, annotations_test, annotation_converter, transform_test)

    return dataset_train, dataset_test


def prepare_drivenact_test(config_file):
    """
    Construct Drive&Act test Dataset instance based on the given configuration file
    Arguments:
        config_file: configuration file to construct Dataset instance
    """
    split_nr      = config_file["data"]["split_nr"]
    folder_splits = config_file["data"]["folder_splits"]
    data_split    = config_file["data"]["data_path"]
    views         = config_file["data"]["views"]
    new_suffix    = config_file["data"]["new_suffix"]
    n_frame       = config_file["data"]["n_frame"]
    frame_size    = config_file["data"]["frame_size"]

    test_df = parseDriveNActTestSplitFiles(split_nr, folder_splits, data_split, views, new_suffix)

    transform_test = torchvision.transforms.Compose([
        RandomSelect(n=n_frame),
        CenterCrop(height=frame_size, width=frame_size),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    annotations_test = test_df['activity'].tolist()
    dataset_test     = DriveNActDataset(test_df, annotations_test, None, transform_test)

    return dataset_test


# ─────────────────────────────────────────────
# Custom dataset prepare functions (new)
# ─────────────────────────────────────────────

def prepare_custom(config_file):
    """
    Construct train and validation Dataset instances for a custom dataset
    with sliding window sampling.

    Config example:
        {
          "data": {
            "data_path":   "/data/splits/",
            "train_split": "train.csv",
            "test_split":  "val.csv",
            "n_frame":     16,
            "stride":      16,
            "frame_size":  224
          }
        }

    Arguments:
        config_file: configuration dict to construct Dataset instances

    Returns:
        dataset_train, dataset_test
    """
    data_path   = config_file["data"]["data_path"]
    train_split = config_file["data"]["train_split"]
    test_split  = config_file["data"]["test_split"]
    n_frame     = config_file["data"].get("n_frame",    16)
    stride      = config_file["data"].get("stride",      16)
    frame_size  = config_file["data"].get("frame_size", 224)

    train_df, test_df = parseCustomSplitFiles(data_path, train_split, test_split)

    # SlidingWindowSample must come first — CustomDataset detects and extracts
    # it during init so windows are pre-sliced once, not on every __getitem__.
    transform_training = torchvision.transforms.Compose([
        SlidingWindowSample(n=n_frame, stride=stride),
        RandomCrop(height=frame_size, width=frame_size),
        RandomHorizontalFlip(),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    transform_test = torchvision.transforms.Compose([
        SlidingWindowSample(n=n_frame, stride=stride),
        CenterCrop(height=frame_size, width=frame_size),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    dataset_train        = CustomDataset(train_df, annotation_converter=None,  transform=transform_training)
    annotation_converter = dataset_train.__get_annotation_converter__()
    dataset_test         = CustomDataset(test_df,  annotation_converter=annotation_converter, transform=transform_test)

    return dataset_train, dataset_test


def prepare_custom_test(config_file):
    """
    Construct a test-only Dataset for a custom dataset (e.g. for inference).

    Arguments:
        config_file: configuration dict to construct Dataset instance

    Returns:
        dataset_test
    """
    data_path  = config_file["data"]["data_path"]
    test_split = config_file["data"]["test_split"]
    n_frame    = config_file["data"].get("n_frame",    16)
    stride     = config_file["data"].get("stride",      16)
    frame_size = config_file["data"].get("frame_size", 224)

    test_df = parseCustomTestSplitFiles(data_path, test_split)

    transform_test = torchvision.transforms.Compose([
        SlidingWindowSample(n=n_frame, stride=stride),
        CenterCrop(height=frame_size, width=frame_size),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    dataset_test = CustomDataset(test_df, annotation_converter=None, transform=transform_test, return_file_path=True)

    return dataset_test