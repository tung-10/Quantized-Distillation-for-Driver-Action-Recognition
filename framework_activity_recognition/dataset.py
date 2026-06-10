import os
import cv2
import torch
import random as rn
import numpy as np
from torch.utils.data import Dataset
from framework_activity_recognition.processing import loadVideo, random_select, loadVideoSequence


class FileBasedDataset(Dataset):
    def __init__(self, data_file_list, ground_truth_label_list,
                 annotation_converter=None, transform=None,
                 dynamicLoading=True, return_file_path=False, return_triplet=False, w2w_embedding=False, model=None):

        self.data_file_list = data_file_list
        self.annotations_original = ground_truth_label_list
        self.transform = transform
        self.data_lookup = {}
        self.return_file_path = return_file_path
        self.return_triplet = return_triplet
        self.model = model
        self.w2w_embedding = w2w_embedding

        if (self.w2w_embedding):
            assert self.model is not None
            self.w2w_embeddings = word2vector_utils.get_word2vec_matrix_for_string_list(model=model, class_names=self.annotations_original)
            print("Computed W2W embeddings, Shape:")
            print(np.shape(self.w2w_embeddings))

        assert len(self.data_file_list) == len(self.annotations_original)

        if annotation_converter is None:
            self.annotation_converter = list(sorted(set(self.annotations_original)))
        else:
            self.annotation_converter = annotation_converter

        self.annotations_transformed = [self.annotation_converter.index(a) for a in self.annotations_original]
        self.nClasses = len(self.annotation_converter)

        if not dynamicLoading:
            for i in range(len(self.data_file_list)):
                self.dataLookup[self.data_file_list[i]] = self.__loaditem__(i)

    def __len__(self):
        return len(self.data_file_list)

    def __get_samples_weight__(self):
        class_sample_count = np.array(
            [len(np.where(self.annotations_transformed == t)[0]) for t in np.unique(self.annotations_transformed)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.annotations_transformed])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        return samples_weight

    def __get_annotation_converter__(self):
        return self.annotation_converter

    def get_labels(self):
        """Required by ImbalancedDatasetSampler."""
        return [self.annotation_converter.index(self.df.iloc[clip_idx]["label"])
                for clip_idx, _ in self._index]

    def __loaditem__(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        sample, label = self.__loaditem__(index)

        if self.transform:
            sample = self.transform(sample)

        if (self.return_triplet):
            ind_neg = rn.sample(
                [i for i in range(len(self.annotations_transformed)) if self.annotations_transformed[i] != label], 1)[0]
            ind_pos = rn.sample(
                [i for i in range(len(self.annotations_transformed)) if self.annotations_transformed[i] == label], 1)[0]

            sample_neg, label_neg = self.__loaditem__(ind_neg)
            sample_pos, label_pos = self.__loaditem__(ind_pos)

            if self.transform:
                sample_neg = self.transform(sample_neg)
                sample_pos = self.transform(sample_pos)

            if (self.return_file_path):
                return sample, label, sample_pos, label_pos, sample_neg, label_neg, self.data_file_list[index]
            else:
                return sample, label, sample_pos, label_pos, sample_neg, label_neg

        if (self.w2w_embedding):
            if (self.return_file_path):
                return sample, label, self.transform(self.w2w_embeddings[index]), self.data_file_list[index]
            else:
                return sample, label, self.transform(self.w2w_embeddings[index])

        if (self.return_file_path):
            return sample, label, self.data_file_list[index]
        else:
            return sample, label


class VideoFileBasedDataset(FileBasedDataset):
    def __init__(self, data_file_list, ground_truth_label_list, annotation_converter=None, transform=None, resize_string=None, selectFramesNr=None):
        super(VideoFileBasedDataset, self).__init__(data_file_list, ground_truth_label_list, annotation_converter, transform)
        self.resize_string = resize_string
        self.selectFramesNr = selectFramesNr

    def __loaditem__(self, index):
        path = self.data_file_list[index]
        _, classname = os.path.split(os.path.split(path)[0])
        data = loadVideo(path, rescale=self.resize_string)
        if self.selectFramesNr is not None:
            data = random_select(data, self.selectFramesNr)
        label = self.annotations_transformed[index]
        return data, label


class DriveNActDataset(FileBasedDataset):
    def __init__(self, df, labels, annotation_converter=None, transform=None, resize=None):
        super().__init__(df, labels, annotation_converter, transform)
        self.resize = resize

    def __loaditem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()
        file_path = self.data_file_list.loc[index, 'file_id']
        frame_start = self.data_file_list.loc[index, 'frame_start']
        frame_end = self.data_file_list.loc[index, 'frame_end']
        frames = loadVideoSequence(file_path, frame_start, frame_end, self.resize)
        label = self.annotations_transformed[index]
        return frames, label

    def __len__(self):
        return len(self.data_file_list.index)


# ─────────────────────────────────────────────
# Custom dataset with sliding window (new)
# ─────────────────────────────────────────────

class CustomDataset(Dataset):
    """
    Dataset for a custom video collection parsed from a CSV split file.

    Videos are read on demand in __getitem__ (no RAM cache).
    SlidingWindowSample is applied per-clip at __getitem__ time:
        - win_idx selects which window to return from the clip
        - Only that window's frames are kept in memory

    Arguments:
        df                  : DataFrame with columns ['clip_path', 'label']
        annotation_converter: list of label strings; pass None to build from df
                              (do this for train only, reuse for val/test)
        transform           : Compose pipeline; SlidingWindowSample must be first
        return_file_path    : if True, __getitem__ also returns the clip path
    """

    def __init__(self, df, annotation_converter=None, transform=None, return_file_path=False):
        self.df               = df.reset_index(drop=True)
        self.transform        = transform
        self.return_file_path = return_file_path

        # ── Build label → int mapping ─────────────────────────────────────
        if annotation_converter is None:
            self.annotation_converter = list(sorted(df["label"].unique()))
        else:
            self.annotation_converter = annotation_converter

        self.nClasses = len(self.annotation_converter)

        # ── Setup slicer ──────────────────────────────────────────────────
        from framework_activity_recognition.videotransform import SlidingWindowSample
        self._slicer = self._find_slicer_in_transform(transform) \
                       or SlidingWindowSample(n=16, stride=16)

        # ── Build flat index using only frame count (no video data in RAM) ─
        print("Indexing clips …")
        self._index = []   # list of (clip_idx, win_idx)

        for clip_idx in range(len(self.df)):
            clip_path = self.df.iloc[clip_idx]["clip_path"]
            try:
                n_frames  = self._get_frame_count(clip_path)
                n_windows = self._count_windows(n_frames)
                for win_idx in range(n_windows):
                    self._index.append((clip_idx, win_idx))
            except Exception as e:
                print(f"  [WARN] Skipping {clip_path}: {e}")

        print(f"  → {len(self.df)} clips, {len(self._index)} windows total, {self.nClasses} classes")

    # ── helpers ───────────────────────────────────────────────────────────

    def _find_slicer_in_transform(self, transform):
        from framework_activity_recognition.videotransform import SlidingWindowSample
        if transform is None:
            return None
        for t in getattr(transform, "transforms", [transform]):
            if isinstance(t, SlidingWindowSample):
                return t
        return None

    def _get_frame_count(self, clip_path: str) -> int:
        """Read only frame count via cv2 — does NOT load pixel data."""
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {clip_path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # CAP_PROP_FRAME_COUNT can be unreliable for some codecs; floor to 1
        return max(count, 1)

    def _count_windows(self, n_frames: int) -> int:
        """How many sliding windows fit in a clip of n_frames?"""
        n = self._slicer.n
        stride = self._slicer.stride
        if n_frames < n:
            return 1   # short clip → looped into 1 window
        count = 0
        start = 0
        while start + n <= n_frames:
            count += 1
            start += stride
        return count

    def __get_annotation_converter__(self):
        return self.annotation_converter

    def get_labels(self):
        clip_label_counts = {}
        for clip_idx, _ in self._index:
            label = self.annotation_converter.index(self.df.iloc[clip_idx]["label"])
            clip_label_counts[clip_idx] = label
        return [clip_label_counts[clip_idx] for clip_idx, _ in self._index]

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int):
        clip_idx, win_idx = self._index[index]
        row       = self.df.iloc[clip_idx]
        label_int = self.annotation_converter.index(row["label"])

        # Load full clip on demand, slice out the requested window
        frames = loadVideo(row["clip_path"])          # [T, H, W, C] np.ndarray
        windows = self._slicer(frames)                # list of [n, H, W, C]
        # cv2.CAP_PROP_FRAME_COUNT can be inaccurate for some codecs,
        # clamp win_idx to avoid IndexError
        win_idx = min(win_idx, len(windows) - 1)
        window  = windows[win_idx]                    # [n, H, W, C]

        # Apply spatial transforms; skip SlidingWindowSample (already done)
        if self.transform:
            from framework_activity_recognition.videotransform import SlidingWindowSample
            for t in getattr(self.transform, "transforms", [self.transform]):
                if not isinstance(t, SlidingWindowSample):
                    window = t(window)

        if self.return_file_path:
            return window, label_int, row["clip_path"]
        return window, label_int