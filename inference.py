"""
inference.py
Run sliding-window inference on untrimmed videos and save per-window logits.

Usage:
    python inference.py \
        --config  config/utcda/test/mobilenet.yaml \
        --videos  /path/to/video_list.txt \
        --output  ./output/logits_sequence.pkl
"""

import argparse
import copy
import os
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml

FILENAME_TO_ID: dict[str, int] = {
    "01_011_01_0_front": 11,
    "01_012_01_0_front": 12,
    "01_013_01_0_front": 13,
    "01_014_01_0_front": 14,
    "01_015_01_0_front": 15,
    "01_016_01_0_front": 16,
    "01_017_01_0_front": 17,
    "01_018_01_0_front": 18,
}

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    from framework_activity_recognition.processing import get_entity_by_module_path

    cfg = copy.deepcopy(config)     
    arch_cfg = cfg["architecture"]

    ckpt = torch.load(
        os.path.join(arch_cfg["model"], "best_model.pth"),
        map_location="cpu",
    )
    num_classes = len(ckpt["annotation"])
    cfg["train"]["num_classes"] = num_classes

    model = get_entity_by_module_path(arch_cfg["location"])(cfg, arch_cfg)

    if "model_state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
    elif "student_model_state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(
            ckpt["student_model_state_dict"], strict=False
        )
    else:
        raise KeyError("Checkpoint does not contain 'model_state_dict' or 'student_model_state_dict'.")

    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_preprocess_transform(frame_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(frame_size * 1.15)),
        transforms.CenterCrop(frame_size),
        transforms.ToTensor(),
        # Normalize to [-1, 1] to match training's normalizeColorInputZeroCenterUnitRange
        # (x - 0.5) / 0.5 = 2x - 1
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std =[0.5, 0.5, 0.5]),
    ])


def preprocess_window(frames: list, transform) -> torch.Tensor:
    """
    frames : list of np.ndarray [H, W, C] uint8 RGB
    returns: torch.Tensor [1, C, T, H, W]
    """
    if not frames:
        raise ValueError("preprocess_window obtains empty frame list.")

    tensors = [transform(f) for f in frames]           # list of [C, H, W]
    clip    = torch.stack(tensors, dim=1).unsqueeze(0)  # [1, C, T, H, W]
    return clip


def read_video_frames(video_path: str) -> list:
    """
    Read all frames from a video file.
    Returns list of RGB np.ndarray.

    FIX #4: raise lỗi rõ ràng nếu video corrupt hoặc không đọc được frame nào.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(
            f"Video đọc được 0 frame — có thể file bị corrupt: {video_path}"
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
    transform,
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

        clip = preprocess_window(window, transform)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True,  help="Test config YAML")
    parser.add_argument("--videos",  required=True,  help="Text file với danh sách đường dẫn video (mỗi dòng 1 file)")
    parser.add_argument("--output",  default="./output/logits_sequence.pkl")
    parser.add_argument("--ext",     default=".avi",  help="Phần mở rộng video dùng để lọc")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    config     = load_config(args.config)
    n_frame    = config["data"].get("n_frame",    16)
    stride     = config["data"].get("stride",     16)
    frame_size = config["data"].get("frame_size", 224)

    print(f"Config: n_frame={n_frame}, stride={stride}, frame_size={frame_size}")
    print("Loading model …")
    model = load_model(config)
    print("Model loaded.\n")

    transform = get_preprocess_transform(frame_size)

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
                model, frames, n_frame, stride, transform
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


if __name__ == "__main__":
    main()