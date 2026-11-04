"""
inference.py
Run sliding-window inference on untrimmed videos using decord and DataLoader.
Based on evaluate_loc.py logic.

Usage:
    python inference.py --config  config/utcda/test/mobilenetquantized.yaml --videos  /mnt/data/quangtungbk/Continuous-Action-Recognition-v2/video_test_front.txt --output  ./output/front/exp2/logits_sequence.pkl
"""

import argparse
import copy
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import traceback
from decord import VideoReader, cpu

# ID Map from postprocessing.py
ID_MAP = {
    "01_011_01_0_front": 11,
    "01_011_01_0_rear": 11,
    "01_012_01_0_front": 12,
    "01_012_01_0_rear": 12,
    "01_013_01_0_front": 13,
    "01_013_01_0_rear": 13,
    "01_014_01_0_front": 14,
    "01_014_01_0_rear": 14,
    "01_015_01_0_front": 15,
    "01_015_01_0_rear": 15,
    "01_016_01_0_front": 16,
    "01_016_01_0_rear": 16,
    "01_017_01_0_front": 17,
    "01_017_01_0_rear": 17,
    "01_018_01_0_front": 18,
    "01_018_01_0_rear": 18,
    "01_011_01_0": 11,
    "01_012_01_0": 12,
    "01_013_01_0": 13,
    "01_014_01_0": 14,
    "01_015_01_0": 15,
    "01_016_01_0": 16,
    "01_017_01_0": 17,
    "01_018_01_0": 18,
}

def map_filename_to_id(filename):
    basename = os.path.basename(str(filename))
    name_no_ext = os.path.splitext(basename)[0]
    return ID_MAP.get(name_no_ext, filename)

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_model(config: dict):
    from framework_activity_recognition.processing import get_entity_by_module_path

    cfg = copy.deepcopy(config)     
    arch_cfg = cfg["architecture"]

    model_path = os.path.join(arch_cfg["model"], "checkpoint_latest.pth")
    print(f"    Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    state_key = next(
        (key for key in (
            "model_state_dict",
            "student_model_state_dict",
            "float_student_model_state_dict",
        ) if key in ckpt),
        None,
    )
    if state_key is None:
        raise KeyError(
            "Checkpoint does not contain a float model state dict "
            "('model_state_dict', 'student_model_state_dict', or "
            f"'float_student_model_state_dict'). Available keys: {list(ckpt)}"
        )
    state_dict = ckpt[state_key]
    
    if "annotation" in ckpt:
        print(f"    [INFO] Checkpoint classes: {ckpt['annotation']}")
    
    if ckpt.get("annotation") is None:
        # Fallback if annotation is missing, try to count classes from weight
        print("    [WARN] Checkpoint missing 'annotation'. Inferring num_classes from weights.")
        # Linear layer usually has 'classifier.1.weight' or similar
        classifier_weights = [
            key for key in state_dict
            if 'classifier' in key and key.endswith('weight')
        ]
        if not classifier_weights:
            raise KeyError(
                f"Cannot infer num_classes from '{state_key}': "
                "no classifier weight was found."
            )
        num_classes = state_dict[classifier_weights[-1]].shape[0]
    else:
        num_classes = len(ckpt["annotation"])
    
    cfg["train"]["num_classes"] = num_classes
    model = get_entity_by_module_path(arch_cfg["location"])(cfg, arch_cfg)

    if state_key == "float_student_model_state_dict":
        fuse_module = ckpt.get("fuse_module")
        if not fuse_module:
            raise KeyError(
                "Quantization checkpoint is missing 'fuse_module'; "
                "cannot reconstruct the trained QAT model."
            )
        quantization_config = cfg.get("train", {}).get("quantization", {})
        backend = quantization_config.get("backend", "fbgemm")
        model.train()
        model = torch.quantization.fuse_modules(model, fuse_module)
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(model, inplace=True)

    model.load_state_dict(state_dict, strict=True)
    print(f"    Loaded weights from '{state_key}' (strict=True)")
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

class VideoInferDataset(Dataset):
    def __init__(self, video_files, n_frame=16, stride=16, frame_size=224):
        self.n_frame = n_frame
        self.stride = stride
        self.frame_size = frame_size
        
        self.data_samples = []
        self.start_frame_idxs = []
        
        print(f"Initializing Dataset with {len(video_files)} videos...")
        for video_path in video_files:
            try:
                vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
                T = len(vr)
                # Logic tương tự evaluate_loc.py
                for start_idx in range(0, max(T - n_frame + 1, 1), stride):
                    self.data_samples.append(video_path)
                    self.start_frame_idxs.append(start_idx)
                
                # Đảm bảo có ít nhất 1 window nếu video quá ngắn
                if T < n_frame and not self.data_samples:
                     self.data_samples.append(video_path)
                     self.start_frame_idxs.append(0)

<<<<<<< HEAD
def preprocess_window(frames: list, frame_size: int) -> torch.Tensor:
    """
    Sử dụng trực tiếp các class Transform của project để đảm bảo tính nhất quán 100% với Training.
    frames : list of np.ndarray [H, W, C] (BGR)
    returns: torch.Tensor [1, C, T, H, W]
    """
    from framework_activity_recognition.videotransform import CenterCrop, normalizeColorInputZeroCenterUnitRange, ToTensor

    # 1. Chuyển list frames thành numpy array [T, H, W, C]
    clip_np = np.array(frames)

    # 2. Khởi tạo các transform (giống hệt prepare_custom trong datautils.py)
    crop = CenterCrop(frame_size, frame_size)
    norm = normalizeColorInputZeroCenterUnitRange()
    to_tensor = ToTensor()

    # 3. Áp dụng theo thứ tự
    clip_np = crop(clip_np)           # Cắt khung hình [T, 224, 224, 3]
    clip_np = norm(clip_np)           # Chuẩn hóa về [-1, 1]
    clip_torch = to_tensor(clip_np)   # Chuyển sang Tensor [C, T, 224, 224]

    return clip_torch.unsqueeze(0)    # Thêm batch dim [1, C, T, H, W]


def read_video_frames(video_path: str) -> list:
    """
    Read all frames from a video file.
    Returns list of BGR np.ndarray (OpenCV default) to match training.

    FIX #4: raise lỗi rõ ràng nếu video corrupt hoặc không đọc được frame nào.
    Harden: bỏ qua các frame None để tránh lỗi NoneType sau này.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            # Bỏ qua frame None thay vì thêm vào list
            continue
        # OpenCV reads in BGR. We keep it as-is to match training logic.
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(
            f"Video đọc được 0 frame — có thể file bị corrupt hoặc rỗng: {video_path}"
        )

    return frames


def pad_window(window: list, n_frame: int) -> list:
    """
    Loop-pad window đến đúng n_frame.

    FIX #5: tách riêng hàm pad, guard window rỗng, dùng list slicing rõ ràng.
    """
    if not window:
        raise ValueError("pad_window nhận được window rỗng.")
    if len(window) >= n_frame:
        return window[:n_frame]
    # lặp vòng tròn đến đủ độ dài
    padded = []
    while len(padded) < n_frame:
        padded.extend(window)
    return padded[:n_frame]


def sliding_window_inference(
    model,
    frames: list,
    n_frame: int,
    stride: int,
    frame_size: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Run model on every sliding window of `frames`.
    Returns (logits, starts).

    FIX #6: loại bỏ window trùng lặp ở cuối bằng cách dùng set;
            stride tối thiểu là 1 để tránh vòng lặp vô hạn.
    FIX #7: bỏ Variable (deprecated), chỉ dùng torch.no_grad().
    FIX #8: in debug shape window đầu tiên để dễ kiểm tra.
    """
    T      = len(frames)
    stride = max(stride, 1)

    # Tạo danh sách start index, loại trùng bằng set
    starts = list(range(0, max(T - n_frame + 1, 1), stride))

    # Đảm bảo window cuối luôn cover phần đuôi video
    last_start = max(T - n_frame, 0)
    if last_start not in starts:
        starts.append(last_start)

    starts = sorted(set(starts))   # ← loại trùng, giữ thứ tự

    print(f"    T={T}, n_frame={n_frame}, stride={stride} → {len(starts)} windows")

    logits_seq = []

    for i, start in enumerate(starts):
        end    = start + n_frame
        window = frames[start:end]
        window = pad_window(window, n_frame)   # đảm bảo đúng độ dài

        clip = preprocess_window(window, frame_size)

        if torch.cuda.is_available():
            clip = clip.cuda()

        # FIX #7: không dùng Variable (deprecated từ PyTorch 0.4)
        with torch.no_grad():
            out = model(clip)   # mong đợi [1, n_classes]

        # FIX #9: kiểm tra output shape bất thường
        if out.ndim != 2 or out.shape[0] != 1:
            raise RuntimeError(
                f"Model output shape bất thường: {out.shape}, mong đợi [1, n_classes]"
            )

        if i == 0:
            print(f"    [DEBUG] clip shape: {clip.shape}, logits shape: {out.shape}")
            print(f"    [DEBUG] logits window 0 (raw): {out[0].cpu().numpy()}")

        logits_seq.append(out.squeeze(0).cpu().numpy())

    return np.array(logits_seq), starts


# ── main ─────────────────────────────────────────────────────────────────────
=======
            except Exception as e:
                print(f"  [ERROR] Cannot open {video_path}: {e}")

        from framework_activity_recognition.videotransform import CenterCrop, normalizeColorInputZeroCenterUnitRange, ToTensor
        self.crop = CenterCrop(frame_size, frame_size)
        self.norm = normalizeColorInputZeroCenterUnitRange()
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        video_path = self.data_samples[index]
        start_idx = self.start_frame_idxs[index]
        
        try:
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            T = len(vr)
            
            # Lấy indices cho window
            end_idx = start_idx + self.n_frame
            indices = np.arange(start_idx, end_idx)
            indices = np.clip(indices, 0, T - 1)
            
            # Load frames [T, H, W, C]
            buffer = vr.get_batch(indices).asnumpy()
            buffer = buffer[..., ::-1].copy()  # Decord RGB -> BGR, matching training

            # Transform
            buffer = self.crop(buffer)
            buffer = self.norm(buffer)
            buffer = self.to_tensor(buffer) # [C, T, H, W]
            
            name = os.path.basename(video_path).split('.')[0]
            return buffer, name, start_idx
        except Exception as e:
            # Trả về tensor rỗng nếu lỗi để DataLoader không crash (nhưng nên xử lý tốt hơn)
            return torch.zeros((3, self.n_frame, self.frame_size, self.frame_size)), "error", -1
>>>>>>> e45ef9b (update)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True,  help="Test config YAML")
    parser.add_argument("--videos",  required=True,  help="Text file containing video paths")
    parser.add_argument("--output",  default="./output/logits_sequence.pkl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    config = load_config(args.config)
    n_frame = config["data"].get("n_frame", 16)
    stride = config["data"].get("stride", 16)
    frame_size = config["data"].get("frame_size", 224)

    print("Loading model...")
    model = load_model(config)
    
    with open(args.videos, "r") as f:
        video_files = [line.strip() for line in f if line.strip()]

<<<<<<< HEAD
    results: dict[int, dict] = {}

    # Đọc danh sách video từ file txt
    with open(args.videos, "r", encoding="utf-8") as f:
        video_files = sorted([
            line.strip() for line in f
            if line.strip() and line.strip().endswith(args.ext)
        ])

    if not video_files:
        raise RuntimeError(
            f"Không tìm thấy file '{args.ext}' nào trong: {args.videos}"
        )

    print(f"Tìm thấy {len(video_files)} video(s).\n")

    for fname in video_files:
        name_no_ext = os.path.splitext(fname)[0]


        if name_no_ext in FILENAME_TO_ID:
            video_id = FILENAME_TO_ID[name_no_ext]
        else:
            print(f"  [WARN] '{name_no_ext}' không có trong FILENAME_TO_ID, bỏ qua.")
            continue

        video_path = os.path.join(os.path.dirname(args.videos), fname)

        print(f"  Processing {fname}  (id={video_id}) …")
        try:
            frames = read_video_frames(video_path)
            logits, starts = sliding_window_inference(
                model, frames, n_frame, stride, frame_size
            )
            results[video_id] = {
                "logits": logits,
                "starts": starts
            }
            print(
                f"  → {len(frames)} frames, {len(logits)} windows, "
                f"shape={logits.shape}, "
                f"mean={logits.mean():.4f}, std={logits.std():.4f}\n"
            )
        except Exception as e:
            traceback.print_exc()  # In chi tiết lỗi để debug
            print(f"  [ERROR] {fname}: {e}\n")
            continue

    if not results:
        print("[WARN] Không có kết quả nào được lưu — kiểm tra lại video paths và FILENAME_TO_ID.")
    else:
        with open(args.output, "wb") as f:
            pickle.dump(results, f)
        print(f"Logits saved to: {args.output}")

        # FIX #11: in summary để dễ kiểm tra kết quả ngay
        print("\nSummary")
        for vid_id, data in results.items():
            logits = data["logits"]
            pred_classes = logits.argmax(axis=1)
            print(
                f"  video_id={vid_id}: {logits.shape[0]} windows, "
                f"pred classes = {np.unique(pred_classes, return_counts=True)}"
            )
=======
    dataset = VideoInferDataset(video_files, n_frame=n_frame, stride=stride, frame_size=frame_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    results_raw = []
    
    print(f"Starting inference on {len(dataset)} windows...")
    model.eval()
    with torch.no_grad():
        for i, (inputs, names, starts) in enumerate(loader):
            if i % 10 == 0:
                print(f"  Batch {i}/{len(loader)}")
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            logits = model(inputs) # [B, num_classes]
            
            print("model:", type(model).__module__, type(model).__name__)
            print("output range:", logits.min().item(), logits.max().item())
            print("first output:", logits[0].detach().cpu().tolist())
            print("argmax classes:", logits.argmax(dim=1)[:20].detach().cpu().tolist())
            
            logits_np = logits.cpu().numpy()
            starts_np = starts.numpy()
            
            for j in range(len(names)):
                if names[j] == "error": continue
                results_raw.append({
                    "name": names[j],
                    "start": starts_np[j],
                    "logits": logits_np[j]
                })

    # Group by video
    final_results = {}
    df = pd.DataFrame(results_raw)
    for name in df["name"].unique():
        sub = df[df["name"] == name].sort_values("start")
        video_id = map_filename_to_id(name)
        final_results[video_id] = {
            "logits": np.stack(sub["logits"].values),
            "starts": sub["start"].values.tolist()
        }
>>>>>>> e45ef9b (update)

    with open(args.output, "wb") as f:
        pickle.dump(final_results, f)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
