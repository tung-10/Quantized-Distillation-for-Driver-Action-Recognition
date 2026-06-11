"""
postprocess_localize.py
Smooth logits sequences → temporal localization → evaluate against GT.

Usage:
    python postprocess_localize.py \
        --input       ./output/logits_sequence.pkl \
        --gt          ./test_grt.csv \
        --output      ./output/ \
        --n_frame     16 \
        --fps         20 \
        --threshold   0.4 \
        --smooth_k    4
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.ndimage import uniform_filter1d


# ── smoothing ────────────────────────────────────────────────────────────────

def mean_smoothing(x: np.ndarray, k: int = 2) -> np.ndarray:
    """Mean filter, window = 2*k."""
    return uniform_filter1d(x, size=2 * k, axis=0, mode="nearest")


def gaussian_smoothing(x: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Gaussian filter along time axis.

    FIX #5: bỏ slice [:len(col)] không cần thiết; thêm assertion để bắt
    nếu convolution output length bất ngờ thay đổi.
    """
    import cv2

    kernel_size = 2 * k + 1
    sigma       = max(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, 1.0)
    kernel      = cv2.getGaussianKernel(kernel_size, sigma)
    y           = np.zeros_like(x, dtype=np.float32)

    for c in range(x.shape[1]):
        col     = x[:, c].astype(np.float32)
        padded  = np.pad(col, k, mode="edge")
        # mode="valid" → output length == len(col) khi padding = k mỗi phía
        filtered = np.convolve(padded, kernel[:, 0], mode="valid")
        assert len(filtered) == len(col), (
            f"gaussian_smoothing: output length mismatch "
            f"({len(filtered)} vs {len(col)}) — kiểm tra k={k}"
        )
        y[:, c] = filtered
    return y


# ── localization ─────────────────────────────────────────────────────────────

def localize(
    prob_seq: np.ndarray,
    starts: list,
    action_threshold: float,
    n_frame: int,
    fps: float,
    background_label: int = 0,
) -> pd.DataFrame:
    """
    prob_seq : np.ndarray [n_windows, n_classes]  (after softmax)
    starts   : actual start frame for each window
    Returns DataFrame with columns [label, start, end]  (in seconds)

    FIX #1: Sử dụng starts thực tế từ inference để tránh lệch thời gian (temporal shift).
    """
    labels_idx = np.argmax(prob_seq, axis=1)   # [n_windows]
    probs_max  = np.max(prob_seq,    axis=1)   # [n_windows]

    active = (probs_max >= action_threshold) & (labels_idx != background_label)

    n_active = int(active.sum())
    if n_active == 0:
        print(
            f"    [WARN] localize: 0/{len(prob_seq)} windows vượt threshold={action_threshold}. "
            f"Max prob = {probs_max.max():.3f} (class {labels_idx[probs_max.argmax()]}). "
            f"Thử giảm --threshold."
        )
        return pd.DataFrame(columns=["label", "start", "end"])

    rows = []
    active_indices = np.where(active)[0]
    for i in active_indices:
        start_frame = starts[i]
        end_frame   = start_frame + n_frame
        rows.append({
            "label": int(labels_idx[i]),
            "start": start_frame / fps,
            "end":   end_frame   / fps,
        })

    return pd.DataFrame(rows)


# ── merge overlapping / adjacent segments ────────────────────────────────────

def merge_segments(
    df: pd.DataFrame,
    merge_gap: float = 1.0,
    min_len:   float = 0.5,
) -> pd.DataFrame:
    """
    Merge segments of the same label that are <= merge_gap seconds apart.
    Drop segments shorter than min_len seconds.
    """
    if df.empty:
        return df

    df = df.sort_values(["label", "start"]).reset_index(drop=True)
    merged = []

    for label, group in df.groupby("label"):
        group = group.sort_values("start").reset_index(drop=True)
        cur_start = group.loc[0, "start"]
        cur_end   = group.loc[0, "end"]

        for i in range(1, len(group)):
            s = group.loc[i, "start"]
            e = group.loc[i, "end"]
            if s - cur_end <= merge_gap:
                cur_end = max(cur_end, e)
            else:
                if cur_end - cur_start >= min_len:
                    merged.append({"label": label, "start": cur_start, "end": cur_end})
                cur_start, cur_end = s, e

        if cur_end - cur_start >= min_len:
            merged.append({"label": label, "start": cur_start, "end": cur_end})

    return (
        pd.DataFrame(merged) if merged
        else pd.DataFrame(columns=["label", "start", "end"])
    )


# ── IoU & evaluation ─────────────────────────────────────────────────────────

def segment_iou(pred_seg: list, gt_segs: np.ndarray) -> np.ndarray:
    """
    pred_seg : [start, end]
    gt_segs  : np.ndarray [N, 2]
    Returns array of IoU scores.
    """
    inter_start = np.maximum(pred_seg[0], gt_segs[:, 0])
    inter_end   = np.minimum(pred_seg[1], gt_segs[:, 1])
    inter       = np.maximum(0.0, inter_end - inter_start)
    union       = (pred_seg[1] - pred_seg[0]) + (gt_segs[:, 1] - gt_segs[:, 0]) - inter
    return inter / np.maximum(union, 1e-6)


def evaluate(
    all_predictions: pd.DataFrame,
    gt: pd.DataFrame,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[dict, float, float, float]:
    """
    Compute per-class and overall precision / recall / F1 at given IoU threshold.
    Ignores label 0 (background) by default.

    FIX #2: video_id trong cả pred và gt đều được ép sang str để tránh type mismatch.
    FIX #3: matched_gt dùng tuple (video_id, gt_row_position) thay vì index gốc của
            DataFrame (dễ bị sai sau reset_index).
    FIX #4: bỏ max(..., 0) để lộ ra nếu tp > n_pred (logic sai).
    """
    # FIX #2: thống nhất kiểu video_id
    gt   = gt.copy()
    pred = all_predictions.copy()
    gt["video_id"]   = gt["video_id"].astype(str)
    pred["video_id"] = pred["video_id"].astype(str)

    gt_eval   = gt[gt["label"] != 0].reset_index(drop=True)
    pred_eval = pred[pred["label"] != 0].reset_index(drop=True)

    class_results = {}

    for label in range(1, num_classes):
        gt_cls   = gt_eval[gt_eval["label"] == label].reset_index(drop=True)
        pred_cls = pred_eval[pred_eval["label"] == label].reset_index(drop=True)

        if len(gt_cls) == 0 and len(pred_cls) == 0:
            continue

        tp = 0
        # FIX #3: key = (video_id, row_position_in_gt_cls) để tránh index ambiguity
        matched_gt: set[tuple] = set()

        for _, pred_row in pred_cls.iterrows():
            vid_gt = gt_cls[gt_cls["video_id"] == str(pred_row["video_id"])].reset_index(drop=True)
            if vid_gt.empty:
                continue

            ious     = segment_iou(
                [pred_row["start"], pred_row["end"]],
                vid_gt[["start", "end"]].values,
            )
            best_idx = int(np.argmax(ious))

            if ious[best_idx] >= iou_threshold:
                # FIX #3: key dùng (video_id, position trong vid_gt)
                match_key = (str(pred_row["video_id"]), best_idx)
                if match_key not in matched_gt:
                    tp += 1
                    matched_gt.add(match_key)

        # FIX #4: assert thay vì max(..., 0) để phát hiện logic sai
        assert tp <= len(pred_cls), (
            f"Label {label}: tp={tp} > n_pred={len(pred_cls)} — lỗi logic matching"
        )
        assert tp <= len(gt_cls), (
            f"Label {label}: tp={tp} > n_gt={len(gt_cls)} — lỗi logic matching"
        )

        fp        = len(pred_cls) - tp
        fn        = len(gt_cls)   - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,        help="logits_sequence.pkl")
    parser.add_argument("--gt",          required=True,        help="Ground truth CSV (video_id,label,start,end)")
    parser.add_argument("--output",      default="./output/",  help="Output directory")
    parser.add_argument("--n_frame",     type=int,   default=16)
    parser.add_argument("--stride",      type=int,   default=16,
                        help="Chỉ dùng để log hoặc fallback format cũ.")
    parser.add_argument("--fps",         type=float, default=20.0)
    parser.add_argument("--threshold",   type=float, default=0.4,  help="Action confidence threshold")
    parser.add_argument("--smooth",      choices=["mean", "gaussian", "none"], default="mean")
    parser.add_argument("--smooth_k",    type=int,   default=4)
    parser.add_argument("--merge_gap",   type=float, default=1.0,  help="Merge gap in seconds")
    parser.add_argument("--min_len",     type=float, default=0.5,  help="Min segment length in seconds")
    parser.add_argument("--iou_thresh",  type=float, default=0.5)
    parser.add_argument("--num_classes", type=int,   default=7)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Config: n_frame={args.n_frame}, stride={args.stride}, fps={args.fps}")
    print(f"        threshold={args.threshold}, smooth={args.smooth}(k={args.smooth_k})")
    print(f"        merge_gap={args.merge_gap}s, min_len={args.min_len}s\n")

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"Loading logits from: {args.input}")
    with open(args.input, "rb") as f:
        data_dict = pickle.load(f)  # {video_id: {"logits": ..., "starts": ...}}

    gt = pd.read_csv(args.gt)

    # FIX #2: convert video_id → str ngay tại điểm load, nhất quán toàn pipeline
    gt["video_id"] = gt["video_id"].astype(str)

    print(f"Loaded GT: {len(gt)} segments, {gt['video_id'].nunique()} videos")
    print(f"GT label distribution:\n{gt['label'].value_counts().sort_index().to_string()}\n")

    # ── per-video inference → localization ───────────────────────────────────
    all_preds = []

    for video_id, data in data_dict.items():
        # Hỗ trợ cả format cũ (array) và format mới (dict)
        if isinstance(data, dict):
            logits_seq = data["logits"]
            starts     = data["starts"]
        else:
            logits_seq = data
            # Fallback cho format cũ: assume stride đều
            starts = [i * args.stride for i in range(len(logits_seq))]

        video_id_str = str(video_id)

        if logits_seq.ndim != 2:
            print(f"  [ERROR] video {video_id_str}: logits shape bất thường {logits_seq.shape}, bỏ qua.")
            continue

        n_windows, n_classes = logits_seq.shape

        # debug: in phân phối raw logits
        print(f"  video {video_id_str}: {n_windows} windows, {n_classes} classes")
        print(f"    raw logits — mean={logits_seq.mean():.3f}, std={logits_seq.std():.3f}, "
              f"max={logits_seq.max():.3f}")

        # softmax
        prob_seq = softmax(logits_seq, axis=1)

        # smooth
        if args.smooth == "mean":
            prob_seq = mean_smoothing(prob_seq, k=args.smooth_k)
        elif args.smooth == "gaussian":
            prob_seq = gaussian_smoothing(prob_seq, k=args.smooth_k)

        # clip để tránh numerical issues sau smoothing
        prob_seq = np.clip(prob_seq, 0.0, 1.0)

        pred_classes = np.argmax(prob_seq, axis=1)
        pred_maxprob = np.max(prob_seq, axis=1)
        print(f"    after smooth — max_prob={pred_maxprob.max():.3f}, "
              f"pred class distribution: {dict(zip(*np.unique(pred_classes, return_counts=True)))}")

        # localize
        df_loc = localize(
            prob_seq,
            starts=starts,
            action_threshold=args.threshold,
            n_frame=args.n_frame,
            fps=args.fps,
        )

        # merge
        df_merged = merge_segments(df_loc, merge_gap=args.merge_gap, min_len=args.min_len)

        if not df_merged.empty:
            df_merged["video_id"] = video_id_str   # FIX #2: str
            all_preds.append(df_merged)

        print(f"    → {len(df_merged)} segments after merge\n")

    # ── assemble predictions ──────────────────────────────────────────────────
    if all_preds:
        pred_df = pd.concat(all_preds, ignore_index=True)
    else:
        pred_df = pd.DataFrame(columns=["video_id", "label", "start", "end"])
        print("[WARN] Không có segment nào được tạo ra. "
              "Kiểm tra --threshold.\n")

    pred_csv = os.path.join(args.output, "temporal_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"Predictions saved to: {pred_csv}")
    print(f"Total predicted segments: {len(pred_df)}")
    if not pred_df.empty:
        print(f"Pred label distribution:\n{pred_df['label'].value_counts().sort_index().to_string()}\n")

    # ── evaluate ──────────────────────────────────────────────────────────────
    print(f"Evaluating at IoU threshold = {args.iou_thresh} …")
    class_results, mean_p, mean_r, mean_f1 = evaluate(
        pred_df,
        gt,
        num_classes=args.num_classes,
        iou_threshold=args.iou_thresh,
    )

    print("\n" + "=" * 58)
    print(f"  Mean Precision : {mean_p  * 100:.2f}%")
    print(f"  Mean Recall    : {mean_r  * 100:.2f}%")
    print(f"  Mean F1        : {mean_f1 * 100:.2f}%")
    print("=" * 58)
    print(f"\n{'Class':>6}  {'GT':>5}  {'Pred':>5}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  "
          f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    for label, r in sorted(class_results.items()):
        print(
            f"  {label:>4}  {r['n_gt']:>5}  {r['n_pred']:>5}  "
            f"{r['tp']:>4}  {r['fp']:>4}  {r['fn']:>4}  "
            f"{r['precision']*100:>5.1f}%  {r['recall']*100:>5.1f}%  {r['f1']*100:>5.1f}%"
        )

    summary_path = os.path.join(args.output, "localization_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"IoU threshold : {args.iou_thresh}\n")
        f.write(f"Mean Precision: {mean_p  * 100:.2f}%\n")
        f.write(f"Mean Recall   : {mean_r  * 100:.2f}%\n")
        f.write(f"Mean F1       : {mean_f1 * 100:.2f}%\n\n")
        for label, r in sorted(class_results.items()):
            f.write(
                f"Class {label}: P={r['precision']*100:.1f}%  "
                f"R={r['recall']*100:.1f}%  F1={r['f1']*100:.1f}%  "
                f"(GT={r['n_gt']}, Pred={r['n_pred']}, TP={r['tp']})\n"
            )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
