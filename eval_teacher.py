"""
eval_teacher.py
Chay rieng teacher (VideoMAE) tren test split va tinh recall / precision.
Dung lai cach dung dataset + teacher cua train() trong repo.

Chay tu thu muc goc cua repo, vi du:
    python eval_teacher.py --config ./config/utcda/train/studentteacher.yaml --checkpoints ./pretrained/utcda_front/checkpoint-29.pth --splits test_clip_rear.csv test_clip_front.csv

Them --check_device de so sanh logits CPU vs GPU tren 1 batch.
"""
import argparse
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from framework_activity_recognition.processing import get_entity_by_module_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="yaml train config (co muc teacher, data, pretraining, train)")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="1 hoac nhieu checkpoint teacher (mac dinh: teacher.checkpoint trong config)")
    p.add_argument("--splits", nargs="+", default=None,
                   help="1 hoac nhieu file test csv (mac dinh: data.test_split trong config)")
    p.add_argument("--test_util", default="framework_activity_recognition.datautils.prepare_custom_test",
                   help="ham dung test set (chi tra ve test_set, giong test_benchmark)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_batches", type=int, default=None, help="debug: gioi han so batch")
    p.add_argument("--check_device", action="store_true",
                   help="so sanh logits CPU vs GPU tren batch dau tien")
    return p.parse_args()


def build_teacher(cfg, ckpt_path):
    """Dung teacher + load checkpoint giong het train()."""
    t_cfg = cfg["teacher"]
    cfg["train"]["num_classes"] = cfg["pretraining"]["teacher_model_num_classes"]
    teacher = get_entity_by_module_path(t_cfg["location"])(cfg, t_cfg)

    sd = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in sd:
        state, target = sd["model_state_dict"], teacher
    elif "student_model_state_dict" in sd:
        state, target = sd["student_model_state_dict"], teacher
    elif "model" in sd:
        state = OrderedDict((k.replace("module.", ""), v) for k, v in sd["model"].items())
        target = teacher.model
    else:
        raise KeyError(f"Khong nhan ra checkpoint, keys: {list(sd)[:10]}")

    # goi nn.Module.load_state_dict de bo qua ham override (nuot loi) cua VideoMAELogit
    state = {k: v for k, v in state.items() if k != "logit_scale"}
    res = nn.Module.load_state_dict(target, state, strict=False)
    print(f"  missing keys   : {len(res.missing_keys)} {res.missing_keys[:5]}")
    print(f"  unexpected keys: {len(res.unexpected_keys)} {res.unexpected_keys[:5]}")

    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.eval()


def build_loader(cfg, split, args):
    c = copy.deepcopy(cfg)
    c["data"]["test_split"] = split
    c["train"]["num_classes"] = c["pretraining"]["teacher_model_num_classes"]
    test_set = get_entity_by_module_path(args.test_util)(c)
    bs = args.batch_size or c["train"].get("batch_size", 8)
    loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                        num_workers=args.num_workers)
    return test_set, loader


def unpack(batch, verbose=False):
    """Lay (input, label) tu batch co the co nhieu hon 2 phan tu.
    Gia dinh: batch[0] = video tensor (B,C,T,H,W), batch[1] = nhan (B,) kieu so nguyen.
    Neu thu tu cua dataset khac, sua ham nay."""
    if verbose:
        desc = []
        for j, b in enumerate(batch):
            if torch.is_tensor(b):
                desc.append(f"[{j}] tensor {tuple(b.shape)} {b.dtype}")
            else:
                desc.append(f"[{j}] {type(b).__name__} {str(b)[:60]}")
        print("  batch structure: " + " | ".join(desc))
    x, y = batch[0], batch[1]
    if not (torch.is_tensor(x) and x.dim() == 5):
        raise ValueError("batch[0] khong phai video tensor 5D, hay sua unpack()")
    if not (torch.is_tensor(y) and y.dim() == 1 and not y.is_floating_point()):
        raise ValueError("batch[1] khong phai nhan (B,), hay sua unpack()")
    return x, y


@torch.no_grad()
def evaluate(teacher, loader, num_classes, device, max_batches=None):
    teacher = teacher.to(device).eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, pred]
    first = True
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = unpack(batch, verbose=first)
        if first:
            print(f"  input shape {tuple(x.shape)} | mean {x.mean():.3f} std {x.std():.3f} "
                  f"min {x.min():.3f} max {x.max():.3f}")
            first = False
        logits = teacher(x.to(device))
        pred = logits.argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
    return cm


def report(cm, name):
    present = [c for c in range(cm.shape[0]) if cm[c].sum() > 0]
    recall = {c: cm[c, c] / cm[c].sum() for c in present}
    prec = {c: (cm[c, c] / cm[:, c].sum() if cm[:, c].sum() > 0 else float("nan"))
            for c in present}
    acc = np.trace(cm) / cm.sum()
    print(f"\n  [{name}]")
    print(f"  window accuracy : {acc:.4f}")
    print(f"  mean recall     : {np.mean(list(recall.values())):.4f}   <- metric giong test() trong wrapper")
    print(f"  mean precision  : {np.nanmean(list(prec.values())):.4f}")
    for c in present:
        print(f"    class {c}: n={cm[c].sum():5d} recall={recall[c]:.3f} precision={prec[c]:.3f}")
    print("  confusion matrix (hang = true, cot = pred):")
    print(cm)
    return float(np.mean(list(recall.values())))


def check_device(teacher, loader):
    if not torch.cuda.is_available():
        print("Khong co GPU, bo qua check_device")
        return
    x, y = unpack(next(iter(loader)))
    with torch.no_grad():
        out_cpu = teacher.to("cpu").eval()(x)
        out_gpu = teacher.to("cuda").eval()(x.cuda()).cpu()
    print(f"  max |CPU - GPU| = {(out_cpu - out_gpu).abs().max().item():.6f}")
    print(f"  pred CPU: {out_cpu.argmax(1).tolist()}")
    print(f"  pred GPU: {out_gpu.argmax(1).tolist()}")
    print(f"  label   : {y.tolist()}")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpts = args.checkpoints or [cfg["teacher"]["checkpoint"]]
    splits = args.splits or [cfg["data"]["test_split"]]

    summary = []
    for ckpt in ckpts:
        print("=" * 70)
        print(f"Checkpoint: {ckpt}")
        teacher = build_teacher(cfg, ckpt)
        for split in splits:
            print("-" * 70)
            print(f"Split: {split}")
            test_set, loader = build_loader(cfg, split, args)
            num_classes = test_set.nClasses
            if args.check_device:
                check_device(teacher, loader)
            cm = evaluate(teacher, loader, num_classes, args.device, args.max_batches)
            mr = report(cm, f"{ckpt.split('/')[-2]}/{ckpt.split('/')[-1]} on {split}")
            summary.append((ckpt, split, mr))

    print("\n" + "=" * 70)
    print("TONG KET (mean recall, window-level)")
    for ckpt, split, mr in summary:
        print(f"  {mr:.4f} | {split:28s} | {ckpt}")


if __name__ == "__main__":
    main()