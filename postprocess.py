"""
postprocess.py
Smooth logits → temporal localization → merge → evaluate.
Based on postprocessing.py logic from original VideoMAE.

Usage:
    python postprocess.py --input       ./output/front/exp2/logits_sequence.pkl --gt          /mnt/data/quangtungbk/Continuous-Action-Recognition-v2/test_grt.csv --output      ./output/front/exp2
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from scipy.special import softmax

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
    '111': 11,
    '121': 12,
    '131': 13,
    '141': 14,
    '151': 15,
    '161': 16,
    '171': 17,
    '181': 18
}

def map_filename_to_id(filename):
    basename = os.path.basename(str(filename))
    name_no_ext = os.path.splitext(basename)[0]
    return ID_MAP.get(name_no_ext, filename)

# ── Smoothing ────────────────────────────────────────────────────────────────

def mean_smoothing(x, k=2):
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y

def gaussian_smoothing(x, k=3):
    l = len(x)
    border = k
    kernel_size = 2 * k + 1
    
    if len(x.shape) == 1:
        width = l
        num_classes = 1
        x = x.reshape(-1, 1)
    else:
        width = l
        num_classes = x.shape[1]
    
    y = np.zeros(x.shape, dtype=np.float32)
    for c in range(num_classes):
        hm = x[:, c].copy()
        origin_max = np.max(hm)
        if origin_max == 0:
            y[:, c] = hm
            continue
        
        dr = np.zeros(width + 2 * border, dtype=np.float32)
        dr[border: -border] = hm
        sigma = max(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, 1.0)
        kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
        dr_filtered = cv2.filter2D(dr.reshape(-1, 1), -1, kernel_1d)[:, 0]
        hm_filtered = dr_filtered[border: -border]
        if np.max(hm_filtered) > 0:
            hm_filtered *= origin_max / np.max(hm_filtered)
        y[:, c] = hm_filtered
        
    return y.reshape(x.shape)

# ── Localization & Merging ───────────────────────────────────────────────────

def localize(prob_seq, starts, threshold, n_frame, fps):
    labels_idx = np.argmax(prob_seq, axis=1)
    probs_max = np.max(prob_seq, axis=1)
    
    active = (probs_max >= threshold)
    
    rows = []
    for i in np.where(active)[0]:
        if labels_idx[i] == 0: continue # background
        
        start_frame = starts[i]
        end_frame = start_frame + n_frame
        rows.append({
            "label": int(labels_idx[i]),
            "start": start_frame / fps,
            "end": end_frame / fps
        })
    return pd.DataFrame(rows)

def process_overlap(data):
    if data.empty: return data
    data = data.sort_values(by=["start"]).reset_index(drop=True)
    for j in range(len(data)-1):
        for i in range(j+1, len(data)):
            if data.loc[i, "start"] < data.loc[j, "end"]:
                data.loc[i, "end"] = 0
                data.loc[i, "start"] = 0
    return data[data["end"] != 0]

def merge_and_remove(data, merge_threshold_sec=0.8, min_len_sec=0.5):
    if data.empty: return data
    
    df_total = []
    for video_id in data["video_id"].unique():
        data_video = data[data["video_id"] == video_id]
        list_label = data_video["label"].unique()
        
        vid_all = []
        for label in list_label:
            data_video_label = data_video[data_video["label"] == label].sort_values("start").reset_index(drop=True)
            
            for j in range(len(data_video_label)-1):
                if data_video_label.loc[j+1, "start"] - data_video_label.loc[j, "end"] <= merge_threshold_sec:
                    data_video_label.loc[j+1, "start"] = data_video_label.loc[j, "start"]
                    data_video_label.loc[j, "end"] = 0
                    data_video_label.loc[j, "start"] = 0
            
            vid_all.append(data_video_label[data_video_label["end"] != 0])
        
        if not vid_all: continue
        vid_all_df = pd.concat(vid_all).reset_index(drop=True)
        vid_all_df = process_overlap(vid_all_df)
        df_total.append(vid_all_df)
        
    if not df_total: return pd.DataFrame(columns=["video_id", "label", "start", "end"])
    
    df_res = pd.concat(df_total).reset_index(drop=True)
    df_res["length"] = df_res["end"] - df_res["start"]
    df_res = df_res[df_res["length"] >= min_len_sec].drop(columns=["length"])
    return df_res

# ── Evaluation ───────────────────────────────────────────────────────────────

def segment_iou(target, candidates):
    """target: [start, end], candidates: [[start, end], ...]"""
    t_start, t_end = target
    c_starts, c_ends = candidates[:, 0], candidates[:, 1]
    
    inter_start = np.maximum(t_start, c_starts)
    inter_end = np.minimum(t_end, c_ends)
    inter = np.maximum(0.0, inter_end - inter_start)
    
    union = (t_end - t_start) + (c_ends - c_starts) - inter
    return inter / np.maximum(union, 1e-6)

def evaluate(pred, gt, num_classes=7, iou_threshold=0.5):
    gt = gt[gt["label"] != 0].copy()
    pred = pred[pred["label"] != 0].copy()
    
    gt["video_id"] = gt["video_id"].astype(str)
    pred["video_id"] = pred["video_id"].astype(str)
    
    results = {}
    for label in range(1, num_classes):
        gt_cls = gt[gt["label"] == label]
        pred_cls = pred[pred["label"] == label]
        
        if len(gt_cls) == 0 and len(pred_cls) == 0: continue
        
        tp = 0
        matched_gt = set()
        
        for _, p_row in pred_cls.iterrows():
            g_vid = gt_cls[gt_cls["video_id"] == p_row["video_id"]].reset_index(drop=True)
            if g_vid.empty: continue
            
            ious = segment_iou([p_row["start"], p_row["end"]], g_vid[["start", "end"]].values)
            best_idx = np.argmax(ious)
            if ious[best_idx] >= iou_threshold:
                match_key = (p_row["video_id"], best_idx)
                if match_key not in matched_gt:
                    tp += 1
                    matched_gt.add(match_key)
        
        prec = tp / len(pred_cls) if len(pred_cls) > 0 else 0
        recall = tp / len(gt_cls) if len(gt_cls) > 0 else 0
        f1 = 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0
        
        results[label] = {"tp": tp, "prec": prec, "recall": recall, "f1": f1, "n_gt": len(gt_cls), "n_pred": len(pred_cls)}
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", default="./output/")
    parser.add_argument("--n_frame", type=int, default=16)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--smooth", choices=["mean", "gaussian", "none"], default="mean")
    parser.add_argument("--smooth_k", type=int, default=4)
    parser.add_argument("--merge_gap", type=float, default=2) # Giống postprocessing.py (2 frames ~ 0.1s)
    parser.add_argument("--min_len", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.input, "rb") as f:
        data_dict = pickle.load(f)
    
    gt = pd.read_csv(args.gt)
    
    all_preds = []
    for video_id, data in data_dict.items():
        logits = data["logits"]
        starts = data["starts"]
        
        probs = softmax(logits, axis=1)
        if args.smooth == "mean":
            probs = mean_smoothing(probs, k=args.smooth_k)
        elif args.smooth == "gaussian":
            probs = gaussian_smoothing(probs, k=args.smooth_k)
            
        df_loc = localize(probs, starts, args.threshold, args.n_frame, args.fps)
        if not df_loc.empty:
            df_loc["video_id"] = str(video_id)
            all_preds.append(df_loc)
            
    if not all_preds:
        print("No predictions found. Try lowering --threshold.")
        return
        
    pred_df = pd.concat(all_preds)
    print(f"DEBUG: Number of raw segments: {len(pred_df)}")
    
    pred_df = merge_and_remove(pred_df, merge_threshold_sec=args.merge_gap, min_len_sec=args.min_len)
    print(f"DEBUG: Number of segments after merge: {len(pred_df)}")
    
    pred_df["video_id"] = pred_df["video_id"].apply(map_filename_to_id).astype(str)
    
    # Debug IDs
    unique_pred_ids = sorted(pred_df["video_id"].unique())
    unique_gt_ids = sorted(gt["video_id"].astype(str).unique())
    print(f"DEBUG: Unique Pred IDs: {unique_pred_ids}")
    print(f"DEBUG: Unique GT IDs: {unique_gt_ids}")

    pred_csv = os.path.join(args.output, "pred_final.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved predictions to {pred_csv}")
    
    res = evaluate(pred_df, gt)
    print("\nEvaluation Results:")
    for label, metrics in res.items():
        print(f"Class {label}: Prec={metrics['prec']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f} (TP={metrics['tp']}, GT={metrics['n_gt']})")
    
    mean_f1 = np.mean([m['f1'] for m in res.values()])
    print(f"\nMean F1: {mean_f1:.4f}")

if __name__ == "__main__":
    main()
