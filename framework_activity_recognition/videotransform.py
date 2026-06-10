import torch
import numpy as np
from framework_activity_recognition.processing import normalize_color_input_zero_center_unit_range, \
    normalize_color_input_zero_center_unit_range_per_channel, unit_range_zero_center_to_unit_range_zero_min,\
        center_crop, random_crop, random_select, random_horizontal_flip


class normalizeColorInputZeroCenterUnitRange(object):
    def __init__(self, max_val=255.0):
        self.max_val = max_val

    def __call__(self, input_tensor):
        return normalize_color_input_zero_center_unit_range(input_tensor, max_val=self.max_val)


class normalizeColorInputZeroCenterUnitRangeChannelWise(object):
    def __call__(self, input_tensor):
        return normalize_color_input_zero_center_unit_range_per_channel(input_tensor)


class unitRangeZeroCenterToUnitRangeZeroMin(object):
    def __call__(self, input_tensor):
        return unit_range_zero_center_to_unit_range_zero_min(input_tensor)


class CenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, input_tensor):
        return center_crop(input_tensor, self.height, self.width)


class RandomCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, input_tensor):
        return random_crop(input_tensor, self.height, self.width)


class RandomSelect(object):
    """Randomly sample n frames from a clip (original behaviour, kept for compatibility)."""
    def __init__(self, n):
        self.n = n

    def __call__(self, input_tensor):
        return random_select(input_tensor, self.n)


class SlidingWindowSample(object):
    """
    Slice a clip into all sliding windows of size n with a given stride.

    Input : numpy array of shape [T, H, W, C]
    Output: list of numpy arrays, each of shape [n, H, W, C]

    Edge case — clip shorter than n frames:
        The clip is tiled until it reaches length n,
        and a single window is returned.
    """
    def __init__(self, n: int = 16, stride: int = 8):
        assert n > 0 and stride > 0, "n and stride must be positive integers"
        self.n = n
        self.stride = stride

    def __call__(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        T = input_tensor.shape[0]

        if T < self.n:
            reps = int(np.ceil(self.n / T))
            tiled = np.concatenate([input_tensor] * reps, axis=0)  # [reps*T, H, W, C]
            return [tiled[:self.n]]

        windows = []
        start = 0
        while start + self.n <= T:
            windows.append(input_tensor[start : start + self.n])
            start += self.stride

        return windows


class RandomHorizontalFlip(object):
    def __init__(self):
        super().__init__()

    def __call__(self, input_tensor):
        return random_horizontal_flip(input_tensor)


class ToTensor(object):
    def __call__(self, input_tensor):
        # numpy: [T, H, W, C]  →  torch: [C, T, H, W]
        result = input_tensor.transpose(3, 0, 1, 2)
        result = np.float32(result)
        return torch.from_numpy(result)


class WindowToTensor(object):
    """
    Apply ToTensor to every window in a list produced by SlidingWindowSample.

    Input : list of numpy arrays [n, H, W, C]
    Output: list of torch tensors [C, n, H, W]
    """
    def __init__(self):
        self._to_tensor = ToTensor()

    def __call__(self, windows: list[np.ndarray]) -> list[torch.Tensor]:
        return [self._to_tensor(w) for w in windows]