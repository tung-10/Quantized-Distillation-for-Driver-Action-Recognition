"""
inference_sliding_window.py
============================
Sliding-window inference on untrimmed videos using the
StudentTeacherQuantization_VideoMAE MobileNetLogit checkpoint.

Supports two model variants stored in the same .pth file:
  - float  : 'float_student_model_state_dict'  → runs on CUDA if available
  - int    : 'int_student_model_state_dict'     → quantized, CPU only

Config (num_classes, width_mult, pool shape …) is read automatically
from the 'annotation' key embedded in the checkpoint.

Preprocessing matches training exactly:
  BGR → RGB → resize (bilinear) → center-crop → normalize to [-1, 1]

Output
------
  logits_sequence.pkl  – {video_id: np.ndarray [n_windows, n_classes]}
                         readable by postprocess_localize.py
  window_results.csv   – one row per window with logit/prob columns

Usage
-----
    python inference_sliding_window.py --video_list   /mnt/data/quangtungbk/Continuous-Action-Recognition-v2/video_test_front.txt --checkpoint  ./model/front/StudentTeacherQuantization_VideoMAE/exp2/checkpoint_latest.pth --output_dir  ./output_continuous/ --model_type  int --stride      16 --batch_size  8 --fps         20 --config      ./config/utcda/test/mobilenetquantized.yaml
"""

import argparse
import os
import pickle
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
#  MobileNetLogit definitions (float + quantized)
def _conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True),
    )
class _Block(nn.Module):
    """Depthwise conv + Pointwise conv  (float version)."""
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3,
                               stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1   = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2   = nn.BatchNorm3d(out_planes)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
class _BlockQ(nn.Module):
    """Depthwise conv + Pointwise conv  (quant-aware version with named ReLU)."""
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3,
                               stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1   = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2   = nn.BatchNorm3d(out_planes)
        self.relu  = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out
def _build_mobilenet_logit(num_classes: int, width_mult: float,
                            dropout: float, pool_kernel: Tuple,
                            quantized: bool) -> nn.Module:
    """
    Build MobileNetLogit matching architecture/mobilenet(quant).py.
    pool_kernel : (T, H, W) for AvgPool3d
    """
    BlockCls = _BlockQ if quantized else _Block
    input_channel = int(32  * width_mult)
    last_channel  = int(1024 * width_mult)
    cfg = [
        [64,   1, (2, 2, 2)],
        [128,  2, (2, 2, 2)],
        [256,  2, (2, 2, 2)],
        [512,  6, (2, 2, 2)],
        [1024, 2, (1, 1, 1)],
    ]
    features = [_conv_bn(3, input_channel, (1, 2, 2))]
    for c, n, s in cfg:
        output_channel = int(c * width_mult)
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(BlockCls(input_channel, output_channel, stride))
            input_channel = output_channel
    features = nn.Sequential(*features)
    class _MobileNetLogit(nn.Module):
        def __init__(self):
            super().__init__()
            self.features  = features
            self.avgpool3d = nn.AvgPool3d(pool_kernel)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(last_channel, num_classes),
            )
            if quantized:
                self.quant   = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()
        def forward(self, x):
            if quantized:
                x = self.quant(x)
            x = self.features(x)
            x = self.avgpool3d(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            if quantized:
                x = self.dequant(x)
            return x
    return _MobileNetLogit()
# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing  (matches framework_activity_recognition/processing.py)
# ─────────────────────────────────────────────────────────────────────────────
def _center_crop(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """frames: [T, H, W, C]  →  [T, height, width, C]"""
    fh, fw = frames.shape[1], frames.shape[2]
    if fh < height or fw < width:
        # zero-pad then crop
        new = np.zeros((frames.shape[0], max(fh, height), max(fw, width), frames.shape[3]),
                       dtype=frames.dtype)
        new[:, :fh, :fw, :] = frames
        frames, fh, fw = new, new.shape[1], new.shape[2]
    ox = math.floor((fw - width)  / 2)
    oy = math.floor((fh - height) / 2)
    return frames[:, oy:oy + height, ox:ox + width, :]
def _normalize_zero_center(frames: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    """Pixel values [0, 255] → [-1, 1]"""
    return (frames.astype(np.float32) / max_val) * 2.0 - 1.0
def preprocess_clip(clip_rgb: np.ndarray, frame_size: int) -> torch.Tensor:
    """
    clip_rgb : [T, H, W, 3]  uint8  RGB  (matching training pipeline)
    Returns  : [C, T, H, W]  float32 tensor, values in [-1, 1]
    """
    # 1. resize each frame so shorter side == frame_size, keep aspect ratio
    T, H, W, C = clip_rgb.shape
    if H != frame_size or W != frame_size:
        scale = frame_size / min(H, W)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))
        resized = np.stack([
            cv2.resize(clip_rgb[t], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            for t in range(T)
        ], axis=0)
    else:
        resized = clip_rgb
    # 2. center crop to frame_size x frame_size
    cropped = _center_crop(resized, frame_size, frame_size)   # [T, fs, fs, 3]
    # 3. normalize to [-1, 1]
    normed = _normalize_zero_center(cropped)                  # [T, fs, fs, 3] float32
    # 4. [T, H, W, C] → [C, T, H, W]
    tensor = torch.from_numpy(normed.transpose(3, 0, 1, 2))   # [3, T, fs, fs]
    return tensor
# ─────────────────────────────────────────────────────────────────────────────
#  Streaming sliding-window generator
#  Memory-efficient: only n_frame frames are kept in RAM at any moment.
# ─────────────────────────────────────────────────────────────────────────────
from collections import deque
def stream_windows(
    video_path: str,
    n_frame: int,
    stride: int,
    target_fps: Optional[float] = None,
):
    """
    Generator that streams sliding windows directly from disk.
    Yields
    ------
    (start_frame_idx: int, clip: np.ndarray [n_frame, H, W, 3] uint8 RGB, fps: float)
    Memory usage: O(n_frame * H * W * 3) regardless of video length.
    The last window is padded by repeating the final frame if incomplete.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_fps and 0 < target_fps < native_fps:
        step = max(1, int(round(native_fps / target_fps)))
        actual_fps = native_fps / step
    else:
        step = 1
        actual_fps = native_fps
    # circular buffer: holds up to n_frame RGB frames
    buf: deque = deque(maxlen=n_frame)
    raw_idx    = 0   # frame index in native video
    kept_idx   = 0   # frame index after FPS subsampling
    next_yield = 0   # kept_idx at which the next window starts
    last_frame = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        # FPS subsampling
        if (raw_idx % step) != 0:
            raw_idx += 1
            continue
        rgb = bgr[..., ::-1].copy()   # BGR → RGB, ensure contiguous
        last_frame = rgb
        buf.append(rgb)
        raw_idx  += 1
        kept_idx += 1
        # yield a window once the buffer is full and we've reached the stride boundary
        if len(buf) == n_frame and (kept_idx - n_frame) == next_yield:
            clip = np.stack(list(buf), axis=0)   # [n_frame, H, W, 3]
            yield next_yield, clip, actual_fps
            next_yield += stride
    cap.release()
    # ── tail: video ended before filling the next window ──────────────────────
    if last_frame is None:
        return   # empty video
    tail_start = kept_idx   # kept_idx right after last yielded window's stride
    # compute how many frames we have from next_yield onward
    available = kept_idx - next_yield
    if 0 < available < n_frame:
        # pad with the last frame
        frames_in_buf = list(buf)[-available:] if available <= len(buf) else list(buf)
        pad_count = n_frame - len(frames_in_buf)
        pad = [last_frame] * pad_count
        clip = np.stack(frames_in_buf + pad, axis=0)
        yield next_yield, clip, actual_fps
# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
def load_model(
    checkpoint_path: str,
    model_type: str,            # "float" | "int" | "auto"
    preferred_device: torch.device,
    num_classes: int,
    frame_size: int,
    width_mult: float,
    dropout: float,
    pool_k: Tuple[int, int, int],
) -> Tuple[nn.Module, torch.device]:
    """
    Load MobileNetLogit from a StudentTeacherQuantization checkpoint.
    NOTE: 'annotation' stored in the checkpoint is the annotation_converter
    (a list of class name strings like ['drive', 'drink', ...]), NOT the
    training config dict.  All architecture parameters must be passed
    explicitly via function arguments (read from the YAML config file).
    Parameters
    ----------
    model_type       : "float" -> fp32 on GPU
                       "int"   -> int8 on CPU (PyTorch quantized)
                       "auto"  -> float if CUDA available, else int
    preferred_device : requested device (overridden to CPU for int model)
    num_classes      : number of output classes
    frame_size       : spatial crop size (H = W)
    width_mult       : MobileNet width multiplier
    dropout          : dropout probability
    pool_k           : (T, H, W) for AvgPool3d
    Returns
    -------
    (model, effective_device)
    """
    cpu = torch.device("cpu")
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=cpu)
    # ── show annotation (class names list) if present ─────────────────────────
    annotation = ckpt.get("annotation")
    if annotation is not None:
        if isinstance(annotation, (list, tuple)):
            print(f"  Classes ({len(annotation)}): {annotation}")
        else:
            # unexpected type — just report it
            print(f"  annotation type: {type(annotation)}")
    print(f"  Model config:")
    print(f"    num_classes={num_classes}, frame_size={frame_size}")
    print(f"    width_mult={width_mult}, pool_kernel={pool_k}, dropout={dropout}")
    # ── choose which weights to load ──────────────────────────────────────────
    has_float = "float_student_model_state_dict" in ckpt
    has_int   = "int_student_model_state_dict"   in ckpt
    if model_type == "auto":
        if preferred_device.type == "cuda" and has_float:
            model_type = "float"
        elif has_int:
            model_type = "int"
        elif has_float:
            model_type = "float"
        else:
            raise KeyError(
                "Checkpoint has neither 'float_student_model_state_dict' "
                "nor 'int_student_model_state_dict'.\n"
                f"Available keys: {list(ckpt.keys())}"
            )
    print(f"  Using model_type='{model_type}'")
    # ── build + load ──────────────────────────────────────────────────────────
    if model_type == "float":
        if not has_float:
            raise KeyError("'float_student_model_state_dict' not found in checkpoint.")
        print("  Building float MobileNetLogit ...")
        model = _build_mobilenet_logit(num_classes, width_mult, dropout,
                                       pool_k, quantized=False)
        state = ckpt["float_student_model_state_dict"]
        model.load_state_dict(state, strict=False)
        effective_device = preferred_device
        model = model.to(effective_device).eval()
        print(f"  Float model loaded -> {effective_device}")
    elif model_type == "int":
        if not has_int:
            raise KeyError("'int_student_model_state_dict' not found in checkpoint.")
        print("  Building quantized (int8) MobileNetLogit ...")
        # Step 1: build QAT-capable float model
        float_model = _build_mobilenet_logit(num_classes, width_mult, dropout,
                                             pool_k, quantized=True)
        # Step 2: fuse modules (list stored in checkpoint during training)
        fuse_module = ckpt.get("fuse_module")
        if fuse_module:
            float_model = torch.quantization.fuse_modules(float_model, fuse_module)
            print(f"  Fused {len(fuse_module)} module groups.")
        else:
            print("  [WARN] 'fuse_module' not in checkpoint; skipping fusion.")
        # Step 3: prepare QAT → convert to int8
        float_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(float_model, inplace=True)
        model = torch.quantization.convert(float_model.eval(), inplace=False)
        # Step 4: load int8 state dict
        state = ckpt["int_student_model_state_dict"]
        model.load_state_dict(state, strict=False)
        effective_device = cpu   # quantized ops are CPU-only in PyTorch
        model = model.eval()
        print("  Quantized (int8) model loaded -> CPU")
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'float', 'int', or 'auto'.")
    return model, effective_device

#  Per-video inference
@torch.no_grad()
def infer_video(
    video_path: str,
    model: nn.Module,
    n_frame: int,
    stride: int,
    batch_size: int,
    target_fps: Optional[float],
    frame_size: int,
    device: torch.device,
) -> Dict:
    """
    Run sliding-window inference on one untrimmed video.
    Uses stream_windows() so only `batch_size * n_frame` frames occupy RAM
    at any time -- safe for long videos on CPU with limited memory.
    Returns
    -------
    dict:
        "logits"       : np.ndarray  [n_windows, n_classes]
        "probs"        : np.ndarray  [n_windows, n_classes]
        "start_frames" : list[int]
        "fps"          : float
    """
    all_logits:   List[np.ndarray] = []
    start_frames: List[int]        = []
    fps_out: float = 0.0
    # accumulate a micro-batch of windows before calling model
    batch_clips:  List[torch.Tensor] = []
    batch_starts: List[int]          = []
    def _flush_batch():
        """Run the accumulated micro-batch through the model and clear it."""
        if not batch_clips:
            return
        batch_tensor = torch.stack(batch_clips).to(device)   # [B, C, T, H, W]
        out = model(batch_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        all_logits.append(out.detach().cpu().float().numpy())
        batch_clips.clear()
        batch_starts.clear()
    for win_start, clip, fps in stream_windows(video_path, n_frame, stride, target_fps):
        fps_out = fps
        start_frames.append(win_start)
        tensor = preprocess_clip(clip, frame_size)   # [C, T, H, W]
        batch_clips.append(tensor)
        batch_starts.append(win_start)
        if len(batch_clips) >= batch_size:
            _flush_batch()
    # flush remaining windows
    _flush_batch()
    if not all_logits:
        return {"logits": np.empty((0, 1)), "probs": np.empty((0, 1)),
                "start_frames": [], "fps": fps_out}
    logits_seq = np.concatenate(all_logits, axis=0)    # [n_windows, n_classes]
    probs_seq  = softmax(logits_seq, axis=1)           # [n_windows, n_classes]
    return {
        "logits":       logits_seq,
        "probs":        probs_seq,
        "start_frames": start_frames,
        "fps":          fps_out,
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sliding-window inference (MobileNetLogit student).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── video source (pick one) ──────────────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_dir",  metavar="DIR",
                     help="Directory of untrimmed videos (recursive scan)")
    src.add_argument("--video_list", metavar="TXT",
                     help="Text file with one absolute video path per line")
    # ── required ─────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",  required=True,
                   help="Path to StudentTeacherQuantization .pth checkpoint")
    # ── config source (YAML takes priority; explicit args are fallback) ──
    p.add_argument("--config",      default=None,
                   help="Path to the training YAML config file (recommended). "
                        "Used to read num_classes, frame_size, width_mult, etc.")
    # ── architecture params (used when --config is not provided) ──
    p.add_argument("--num_classes", type=int,   default=7,
                   help="Number of output classes (fallback if --config not given)")
    p.add_argument("--frame_size",  type=int,   default=224,
                   help="Spatial crop size H=W (fallback if --config not given)")
    p.add_argument("--width_mult",  type=float, default=1.0,
                   help="MobileNet width multiplier (fallback if --config not given)")
    p.add_argument("--dropout",     type=float, default=0.2,
                   help="Dropout probability (fallback if --config not given)")
    p.add_argument("--pool_kernel", type=int,   default=1,
                   help="AvgPool3d temporal kernel (fallback if --config not given)")
    p.add_argument("--pool_h",      type=int,   default=7,
                   help="AvgPool3d height (fallback if --config not given)")
    p.add_argument("--pool_w",      type=int,   default=7,
                   help="AvgPool3d width  (fallback if --config not given)")
    # ── inference params ──
    p.add_argument("--output_dir",  default="./output/",
                   help="Where to save logits_sequence.pkl and window_results.csv")
    p.add_argument("--model_type",  default="auto",
                   choices=["float", "int", "auto"],
                   help="'float'=fp32 GPU; 'int'=int8 CPU; 'auto'=choose by device")
    p.add_argument("--n_frame",     type=int,   default=16,
                   help="Frames per sliding window")
    p.add_argument("--stride",      type=int,   default=8,
                   help="Frame stride between windows (8 = 50%% overlap for n_frame=16)")
    p.add_argument("--batch_size",  type=int,   default=4,
                   help="Windows per forward pass")
    p.add_argument("--fps",         type=float, default=0,
                   help="Target decode FPS (0 = native video FPS)")
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--ext",         default="mp4,avi,mov,mkv",
                   help="Comma-separated video extensions (only used with --video_dir)")
    return p.parse_args()
def collect_videos(video_dir: str, extensions: List[str]) -> List[Path]:
    """Recursively find all video files under video_dir matching extensions."""
    vdir = Path(video_dir)
    videos = []
    for ext in extensions:
        videos.extend(vdir.rglob(f"*.{ext}"))
    return sorted(videos)
def load_videos_from_list(list_path: str) -> List[Path]:
    """
    Read a text file with one video path per line.
    Blank lines and lines starting with '#' are ignored.
    Exits with an error if any listed path does not exist.
    """
    list_path = Path(list_path)
    if not list_path.exists():
        print(f"[ERROR] --video_list file not found: {list_path}")
        sys.exit(1)
    videos = []
    missing = []
    with open(list_path, "r") as f:
        for lineno, line in enumerate(f, 1):
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            vp = Path(p)
            if not vp.exists():
                missing.append(f"  line {lineno}: {p}")
            else:
                videos.append(vp)
    if missing:
        print(f"[ERROR] {len(missing)} path(s) in --video_list do not exist:")
        print("\n".join(missing[:10]))
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more.")
        sys.exit(1)
    return videos
def main():
    args = parse_args()
    # device
    if args.device == "auto":
        preferred = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        preferred = torch.device(args.device)
    print(f"Preferred device : {preferred}")
    os.makedirs(args.output_dir, exist_ok=True)
    # read architecture config
    num_classes = args.num_classes
    frame_size  = args.frame_size
    width_mult  = args.width_mult
    dropout     = args.dropout
    pool_k      = (args.pool_kernel, args.pool_h, args.pool_w)
    if args.config is not None:
        import yaml
        print(f"Reading config from: {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("train", {})
        data_cfg  = cfg.get("data",  {})
        arch_cfg  = cfg.get("student", cfg.get("architecture", {}))
        pool_cfg  = arch_cfg.get("pool", {})
        num_classes = train_cfg.get("num_classes", num_classes)
        frame_size  = data_cfg.get("frame_size",  frame_size)
        width_mult  = arch_cfg.get("width_mult",  width_mult)
        dropout     = arch_cfg.get("dropout_prob", dropout)
        pool_k      = (
            pool_cfg.get("kernel", pool_k[0]),
            pool_cfg.get("height", pool_k[1]),
            pool_cfg.get("width",  pool_k[2]),
        )
        print(f"  num_classes={num_classes}, frame_size={frame_size}, "
              f"width_mult={width_mult}, pool_k={pool_k}")
    else:
        print("[WARN] --config not provided. Using explicit args or defaults.")
        print(f"  num_classes={num_classes}, frame_size={frame_size}, "
              f"width_mult={width_mult}, pool_k={pool_k}")
    # load model
    model, device = load_model(
        checkpoint_path = args.checkpoint,
        model_type      = args.model_type,
        preferred_device= preferred,
        num_classes     = num_classes,
        frame_size      = frame_size,
        width_mult      = width_mult,
        dropout         = dropout,
        pool_k          = pool_k,
    )
    print(f"Effective device : {device}")
    print(f"Num classes      : {num_classes}")
    print(f"Frame size       : {frame_size}")
    # collect videos
    if args.video_list is not None:
        videos = load_videos_from_list(args.video_list)
        print(f"\nLoaded {len(videos)} video(s) from list: {args.video_list}\n")
    else:
        extensions = [e.strip().lower().lstrip(".") for e in args.ext.split(",")]
        videos = collect_videos(args.video_dir, extensions)
        if not videos:
            print(f"[ERROR] No videos found in {args.video_dir} with extensions {extensions}")
            sys.exit(1)
        print(f"\nFound {len(videos)} video(s) in {args.video_dir}\n")
    # inference
    logits_dict: Dict[str, np.ndarray] = {}
    csv_rows: List[dict] = []
    for video_path in tqdm(videos, desc="Inference", unit="vid"):
        video_id = video_path.stem
        t0 = time.time()
        try:
            result = infer_video(
                video_path  = str(video_path),
                model       = model,
                n_frame     = args.n_frame,
                stride      = args.stride,
                batch_size  = args.batch_size,
                target_fps  = args.fps if args.fps > 0 else None,
                frame_size  = frame_size,
                device      = device,
            )
        except Exception as e:
            print(f"\n  [WARN] Skipping {video_id}: {e}")
            continue
        logits_seq   = result["logits"]
        probs_seq    = result["probs"]
        start_frames = result["start_frames"]
        fps          = result["fps"]
        elapsed      = time.time() - t0
        logits_dict[video_id] = logits_seq
        print(f"  {video_id}: {len(logits_seq)} windows "
              f"| fps={fps:.1f}, stride={args.stride} | {elapsed:.1f}s")
        for w_idx, (sf, lw, pw) in enumerate(zip(start_frames, logits_seq, probs_seq)):
            ef  = sf + args.n_frame
            row = {
                "video_id":    video_id,
                "window_idx":  w_idx,
                "start_frame": sf,
                "end_frame":   ef,
                "start_sec":   sf / fps,
                "end_sec":     ef / fps,
                "pred_class":  int(np.argmax(pw)),
                "pred_prob":   float(np.max(pw)),
            }
            for c in range(lw.shape[0]):
                row[f"logit_cls{c}"] = float(lw[c])
                row[f"prob_cls{c}"]  = float(pw[c])
            csv_rows.append(row)
    # save outputs
        pickle.dump(logits_dict, f)
    total_wins = sum(v.shape[0] for v in logits_dict.values())
    print(f"\n[Saved] logits_sequence.pkl  →  {pkl_path}")
    print(f"        {len(logits_dict)} video(s), {total_wins} total windows")
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(args.output_dir, "window_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Saved] window_results.csv   →  {csv_path}  ({len(df)} rows)")
    else:
        print("No windows processed — CSV not saved.")
    print("\nDone.")
if __name__ == "__main__":
    main()