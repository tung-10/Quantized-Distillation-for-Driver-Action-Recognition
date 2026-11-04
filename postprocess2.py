"""
postprocess_v2.py
==================
Hậu xử lý kết quả inference từ inference_sliding_window.py.

Pipeline:
    logits_sequence.pkl
        → softmax → probabilities [n_windows, n_classes]
        → smoothing (mean hoặc gaussian)
        → activity_localization (threshold-based)
        → convert index → seconds
        → merge_and_remove (gộp segment gần nhau, loại overlap, lọc ngắn)
        → evaluate vs ground truth

Usage
-----
    python postprocess2.py --input      ./output_continuous/front/exp2/logits_sequence.pkl --gt         /mnt/data/quangtungbk/Continuous-Action-Recognition-v2/test_grt.csv --output_dir ./output_continuous/front/exp2 --n_frame    16 --fps        20 --stride     16 --threshold  0.3 --smooth     mean --smooth_k   4 --merge_threshold 2 --min_len    0 --num_classes 7

Format file GT (test_grt.csv):
    video_id,label,start,end
    11,1,10.5,14.2
    ...
    label=0 là background, bị bỏ qua khi evaluate.
"""

import argparse
import os
import pickle
import warnings
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.special import softmax

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("mode.chained_assignment", None)


# ─────────────────────────────────────────────────────────────────────────────
#  ID mapping (filename → numeric video_id)
#  Chỉnh sửa theo dataset của bạn
# ─────────────────────────────────────────────────────────────────────────────

ID_MAP = {
    "01_011_01_0_front": 11,
    "01_011_01_0_rear":  11,
    "01_012_01_0_front": 12,
    "01_012_01_0_rear":  12,
    "01_013_01_0_front": 13,
    "01_013_01_0_rear":  13,
    "01_014_01_0_front": 14,
    "01_014_01_0_rear":  14,
    "01_015_01_0_front": 15,
    "01_015_01_0_rear":  15,
    "01_016_01_0_front": 16,
    "01_016_01_0_rear":  16,
    "01_017_01_0_front": 17,
    "01_017_01_0_rear":  17,
    "01_018_01_0_front": 18,
    "01_018_01_0_rear":  18,
    "01_011_01_0": 11,
    "01_012_01_0": 12,
    "01_013_01_0": 13,
    "01_014_01_0": 14,
    "01_015_01_0": 15,
    "01_016_01_0": 16,
    "01_017_01_0": 17,
    "01_018_01_0": 18,
}

CLASS_NAMES = ["Background", "Smoking", "Calling", "Texting",
               "Drinking", "Yawning", "Joking"]


def map_filename_to_id(filename: str):
    """Map video filename stem to numeric ID.  Falls back to original string."""
    basename = os.path.basename(str(filename))
    name_no_ext = os.path.splitext(basename)[0]
    return ID_MAP.get(name_no_ext, name_no_ext)


# ─────────────────────────────────────────────────────────────────────────────
#  Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def mean_smoothing(x: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Mean filter along time axis.  window = 2*k.
    x : [T, C]
    """
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y


def gaussian_smoothing(x: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Gaussian filter along time axis per class channel.
    x : [T, C]
    """
    l, num_classes = x.shape
    kernel_size = 2 * k + 1
    sigma = max(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, 1.0)
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
    border = k

    y = np.zeros_like(x, dtype=np.float32)
    for c in range(num_classes):
        hm = x[:, c].astype(np.float32)
        origin_max = np.max(hm)
        if origin_max == 0:
            y[:, c] = hm
            continue
        dr = np.zeros(l + 2 * border, dtype=np.float32)
        dr[border:-border] = hm
        dr_filtered = cv2.filter2D(dr.reshape(-1, 1), -1, kernel_1d)[:, 0]
        hm_filtered = dr_filtered[border:-border]
        if np.max(hm_filtered) > 0:
            hm_filtered *= origin_max / np.max(hm_filtered)
        y[:, c] = hm_filtered
    return y


# ─────────────────────────────────────────────────────────────────────────────
#  Activity localization
# ─────────────────────────────────────────────────────────────────────────────

def activity_localization(
    prob_seq: np.ndarray,
    action_threshold: float,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Per-window argmax + threshold → list of (label, start_window, end_window).

    prob_seq : [T, C]  probabilities (after softmax + smoothing)
    Returns  : (activities_idx, startings, endings)  all in window index units
    """
    labels_idx = np.argmax(prob_seq, axis=1)     # [T]
    probs_max  = np.max(prob_seq, axis=1)        # [T]

    action_tag = (probs_max >= action_threshold).astype(int)

    activities_idx, startings, endings = [], [], []
    for i in range(len(action_tag)):
        if action_tag[i] == 1:
            activities_idx.append(int(labels_idx[i]))
            startings.append(i)
            endings.append(i + 1)

    return activities_idx, startings, endings


# ─────────────────────────────────────────────────────────────────────────────
#  Post-processing: merge + remove overlaps + filter short
# ─────────────────────────────────────────────────────────────────────────────

def process_overlap(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove overlapping segments (keep first by start time).
    Segments that start before a previous segment ends are zeroed out.
    """
    if data.empty:
        return data
    data = data.sort_values(by="start").reset_index(drop=True)
    for j in range(len(data) - 1):
        for i in range(j + 1, len(data)):
            if data.loc[i, "start"] < data.loc[j, "end"]:
                data.loc[i, "end"]   = 0
                data.loc[i, "start"] = 0
    data = data[data["end"] != 0]
    return data


def merge_and_remove(
    data: pd.DataFrame,
    merge_threshold: float = 2.0,
    min_len: float = 3.0,
) -> pd.DataFrame:
    """
    For each video and class:
        1. Merge consecutive segments whose gap <= merge_threshold (seconds).
        2. Remove overlapping segments within a video.
        3. Drop segments shorter than min_len seconds.

    Parameters
    ----------
    data            : DataFrame with columns [video_id, label, start, end]
    merge_threshold : max gap in seconds to merge two segments of same class
    min_len         : minimum segment length in seconds to keep

    Returns
    -------
    DataFrame with columns [video_id, label, start, end]
    """
    result_parts = []
    data = data.reset_index(drop=True)

    for vid_id in data["video_id"].unique():
        data_video = data[data["video_id"] == vid_id]
        vid_all_parts = []

        for label in data_video["label"].unique():
            seg = (data_video[data_video["label"] == label]
                   .reset_index(drop=False)
                   .sort_values("start")
                   .reset_index(drop=True))

            # merge segments whose gap <= merge_threshold
            for j in range(len(seg) - 1):
                gap = seg.loc[j + 1, "start"] - seg.loc[j, "end"]
                if gap <= merge_threshold:
                    seg.loc[j + 1, "start"] = seg.loc[j, "start"]
                    seg.loc[j, "end"]   = 0
                    seg.loc[j, "start"] = 0

            vid_all_parts.append(seg)

        if not vid_all_parts:
            continue

        vid_all = pd.concat(vid_all_parts, ignore_index=True)
        vid_all = vid_all[vid_all["end"] != 0]

        print(f"  vid {vid_id}: {len(vid_all)} segments before overlap removal")
        vid_all = process_overlap(vid_all)

        result_parts.append(vid_all)

    if not result_parts:
        return pd.DataFrame(columns=["video_id", "label", "start", "end"])

    df_total = pd.concat(result_parts, ignore_index=True)
    df_total = df_total[df_total["end"] != 0]

    # filter short segments
    df_total["_len"] = df_total["end"] - df_total["start"]
    df_total = df_total[df_total["_len"] >= min_len].drop(columns=["_len"])

    # drop extra columns that may have come from reset_index(drop=False)
    for col in ["index", "level_0"]:
        if col in df_total.columns:
            df_total = df_total.drop(columns=[col])

    df_total = df_total.sort_values(["video_id", "start"]).reset_index(drop=True)
    return df_total


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def segment_iou(
    pred_seg: np.ndarray,
    gt_segs: np.ndarray,
) -> np.ndarray:
    """
    Temporal IoU between one prediction and multiple GT segments.

    pred_seg : [2]   (start, end)  in seconds
    gt_segs  : [N, 2]
    Returns  : [N]  IoU values in [0, 1]
    """
    inter_start = np.maximum(pred_seg[0], gt_segs[:, 0])
    inter_end   = np.minimum(pred_seg[1], gt_segs[:, 1])
    inter       = np.maximum(0.0, inter_end - inter_start)
    union       = (pred_seg[1] - pred_seg[0]) + (gt_segs[:, 1] - gt_segs[:, 0]) - inter
    return inter / np.maximum(union, 1e-6)


def compute_metrics(
    prediction: pd.DataFrame,
    gt: pd.DataFrame,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Tuple[Dict, float, float, float]:
    """
    Compute per-class precision / recall / F1 at a given IoU threshold.
    Class 0 (background) is always skipped.

    Returns
    -------
    (class_results dict, mean_precision, mean_recall, mean_f1)
    """
    gt_eval   = gt[gt["label"] != 0].copy()
    pred_eval = prediction[prediction["label"] != 0].copy()

    gt_eval["video_id"]   = gt_eval["video_id"].astype(str)
    pred_eval["video_id"] = pred_eval["video_id"].astype(str)

    gt_by_vid   = gt_eval.groupby("video_id")

    class_results = {}

    for label in range(1, num_classes):
        gt_cls   = gt_eval[gt_eval["label"] == label].reset_index(drop=True)
        pred_cls = pred_eval[pred_eval["label"] == label].reset_index(drop=True)

        if len(gt_cls) == 0 and len(pred_cls) == 0:
            continue

        tp = 0
        matched_gt = set()

        for _, pred_row in pred_cls.iterrows():
            vid_id = str(pred_row["video_id"])
            vid_gt = gt_cls[gt_cls["video_id"] == vid_id]
            if vid_gt.empty:
                continue
            ious = segment_iou(
                pred_row[["start", "end"]].values,
                vid_gt[["start", "end"]].values,
            )
            best_idx = int(np.argmax(ious))
            if ious[best_idx] >= iou_threshold:
                gt_global_idx = vid_gt.index[best_idx]
                if gt_global_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(gt_global_idx)

        fp        = max(len(pred_cls) - tp, 0)
        fn        = max(len(gt_cls) - tp, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        class_results[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "n_gt": len(gt_cls), "n_pred": len(pred_cls),
        }

    if class_results:
        mean_p  = float(np.mean([v["precision"] for v in class_results.values()]))
        mean_r  = float(np.mean([v["recall"]    for v in class_results.values()]))
        mean_f1 = float(np.mean([v["f1"]        for v in class_results.values()]))
    else:
        mean_p = mean_r = mean_f1 = 0.0

    return class_results, mean_p, mean_r, mean_f1


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-processing: smoothing → localization → merge → evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,
                   help="logits_sequence.pkl from inference_sliding_window.py")
    p.add_argument("--gt",         required=True,
                   help="Ground truth CSV (video_id, label, start_sec, end_sec)")
    p.add_argument("--output_dir", default="./output/",
                   help="Where to save pred_ff.csv and localization_summary.txt")

    # window → time conversion
    p.add_argument("--n_frame",    type=int,   default=16,
                   help="Frames per window (same as inference)")
    p.add_argument("--fps",        type=float, default=20.0,
                   help="Sampling FPS used during inference")
    p.add_argument("--stride",     type=int,   default=8,
                   help="Window stride in frames (same as inference)")

    # localization
    p.add_argument("--threshold",  type=float, default=0.4,
                   help="Confidence threshold for activity_localization")

    # smoothing
    p.add_argument("--smooth",     choices=["mean", "gaussian", "none"],
                   default="mean")
    p.add_argument("--smooth_k",   type=int, default=4,
                   help="Smoothing half-window size")

    # post-processing
    p.add_argument("--merge_threshold", type=float, default=2.0,
                   help="Max gap (seconds) to merge two segments of the same class")
    p.add_argument("--min_len",    type=float, default=3.0,
                   help="Minimum segment length (seconds) to keep")

    # evaluation
    p.add_argument("--num_classes", type=int, default=7,
                   help="Total number of classes including background (class 0)")
    p.add_argument("--iou_thresh",  type=float, default=0.5,
                   help="IoU threshold for TP in evaluation")

    # options
    p.add_argument("--no_id_map",  action="store_true",
                   help="Skip ID_MAP conversion of video_id")
    p.add_argument("--input_is_probs", action="store_true",
                   help="If set, assume pickle stores probabilities (not logits) "
                        "-- skip softmax step")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load logits ───────────────────────────────────────────────────────────
    print(f"Loading: {args.input}")
    with open(args.input, "rb") as f:
        logits_dict = pickle.load(f)   # {video_id: np.ndarray [T, C]}
    print(f"  {len(logits_dict)} video(s) loaded.")

    # ── localization loop ─────────────────────────────────────────────────────
    localization_rows = []

    for video_id, logits_seq in logits_dict.items():
        logits_seq = np.array(logits_seq, dtype=np.float32)

        # handle list-of-arrays format (some checkpoints save probs as list)
        if logits_seq.ndim == 1:
            logits_seq = np.stack(logits_seq)

        # ① softmax → probabilities
        if args.input_is_probs:
            prob_seq = logits_seq
        else:
            prob_seq = softmax(logits_seq, axis=1)   # [T, C]

        # ② smoothing
        if args.smooth == "mean":
            prob_seq = mean_smoothing(prob_seq, k=args.smooth_k)
        elif args.smooth == "gaussian":
            prob_seq = gaussian_smoothing(prob_seq, k=args.smooth_k)
        # "none" → no smoothing

        # ③ activity localization
        activities_idx, startings, endings = activity_localization(
            prob_seq, action_threshold=args.threshold
        )

        # ④ convert window index → seconds
        #    window i starts at frame  i * stride
        #    window i ends   at frame  i * stride + n_frame
        #    time (sec) = frame / fps
        #    BUT: localization returns window indices, not frame indices
        #    start_sec = win_idx * stride / fps
        #    end_sec   = (win_idx + 1) * stride / fps   (1 window duration = stride frames)
        #    (matches: start = s * 16 / 20  when stride==n_frame, fps==20)
        for label, s, e in zip(activities_idx, startings, endings):
            start_sec = s * args.stride / args.fps
            end_sec   = e * args.stride / args.fps
            localization_rows.append({
                "video_id": video_id,
                "label":    label,
                "start":    start_sec,
                "end":      end_sec,
            })

    # ── build raw dataframe ───────────────────────────────────────────────────
    rough_loc = pd.DataFrame(localization_rows,
                             columns=["video_id", "label", "start", "end"])
    print(f"\nRaw localization: {len(rough_loc)} segments across "
          f"{rough_loc['video_id'].nunique()} video(s).")

    if rough_loc.empty:
        print("[WARN] No segments detected. Try lowering --threshold.")
        sys.exit(0)

    # cast start/end to int (seconds, frame-level granularity)
    rough_loc["start"] = rough_loc["start"].map(lambda x: int(float(x)))
    rough_loc["end"]   = rough_loc["end"].map(lambda x: int(float(x)))

    # ── merge & remove ────────────────────────────────────────────────────────
    print("\nRunning merge_and_remove ...")
    prediction = merge_and_remove(
        rough_loc,
        merge_threshold=args.merge_threshold,
        min_len=args.min_len,
    )
    print(f"After post-processing: {len(prediction)} segments.")

    # ── map video IDs ─────────────────────────────────────────────────────────
    if not args.no_id_map:
        prediction["video_id"] = prediction["video_id"].apply(map_filename_to_id)

    # ── save predictions ──────────────────────────────────────────────────────
    pred_csv = os.path.join(args.output_dir, "pred_ff.csv")
    prediction.to_csv(pred_csv, columns=["video_id", "label", "start", "end"],
                      index=False)
    print(f"\n[Saved] predictions → {pred_csv}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    print(f"\nLoading GT: {args.gt}")
    gt = pd.read_csv(args.gt)
    gt["video_id"] = gt["video_id"].astype(str)
    prediction["video_id"] = prediction["video_id"].astype(str)

    pred_eval = prediction[prediction["label"] != 0]
    gt_eval   = gt[gt["label"] != 0]
    M = len(gt_eval)    # total GT actions
    N = len(pred_eval)  # total predictions

    print(f"Total GT actions  : {M}")
    print(f"Total predictions : {N}")
    print(f"Evaluating at IoU threshold = {args.iou_thresh} ...")

    class_results, mean_p, mean_r, mean_f1 = compute_metrics(
        prediction, gt,
        num_classes=args.num_classes,
        iou_threshold=args.iou_thresh,
    )

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Mean Precision : {mean_p*100:.2f}%")
    print(f"  Mean Recall    : {mean_r*100:.2f}%")
    print(f"  Mean F1        : {mean_f1*100:.2f}%")
    print("=" * 60)
    header = f"{'Class':>6}  {'Name':<12}  {'GT':>5}  {'Pred':>5}  " \
             f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
    print(header)
    print("-" * len(header))
    for label, r in sorted(class_results.items()):
        cname = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"cls{label}"
        print(f"  {label:>4}  {cname:<12}  {r['n_gt']:>5}  {r['n_pred']:>5}  "
              f"{r['tp']:>4}  {r['fp']:>4}  {r['fn']:>4}  "
              f"{r['precision']*100:>5.1f}%  {r['recall']*100:>5.1f}%  {r['f1']*100:>5.1f}%")

    # ── save summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "localization_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"smooth        : {args.smooth} (k={args.smooth_k})\n")
        f.write(f"threshold     : {args.threshold}\n")
        f.write(f"merge_thresh  : {args.merge_threshold} s\n")
        f.write(f"min_len       : {args.min_len} s\n")
        f.write(f"iou_threshold : {args.iou_thresh}\n\n")
        f.write(f"GT actions    : {M}\n")
        f.write(f"Predictions   : {N}\n\n")
        f.write(f"Mean Precision: {mean_p*100:.2f}%\n")
        f.write(f"Mean Recall   : {mean_r*100:.2f}%\n")
        f.write(f"Mean F1       : {mean_f1*100:.2f}%\n\n")
        for label, r in sorted(class_results.items()):
            cname = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"cls{label}"
            f.write(f"Class {label} ({cname}): "
                    f"P={r['precision']*100:.1f}%  R={r['recall']*100:.1f}%  "
                    f"F1={r['f1']*100:.1f}%  "
                    f"(GT={r['n_gt']}, Pred={r['n_pred']}, TP={r['tp']})\n")
    print(f"\n[Saved] summary → {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()