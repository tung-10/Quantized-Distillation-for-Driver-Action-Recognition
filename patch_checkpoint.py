import torch
ckpt_path = "/mnt/data/quangtungbk/qd-driver-activity-reco/pretrained/utcda_front/checkpoint_latest.pth"  # ← sửa path
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Keys gốc:", list(ckpt.keys()) if isinstance(ckpt, dict) else "state_dict thẳng")
# Nếu checkpoint là state_dict thẳng (không có wrapper)
if not isinstance(ckpt, dict) or all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    ckpt = {"model_state_dict": ckpt}
# Đảm bảo có key "model_state_dict"
for candidate in ["state_dict", "model", "weights", "net", "teacher_model_state_dict"]:
    if candidate in ckpt and "model_state_dict" not in ckpt:
        ckpt["model_state_dict"] = ckpt[candidate]
        break
# Thêm các key mà driver.py cần nhưng VideoMAE checkpoint không có
ckpt.setdefault("annotation", None)   # ← fix lỗi hiện tại
ckpt.setdefault("fuse_module", [])    # ← fix lỗi trước
torch.save(ckpt, ckpt_path)
print("✅ Checkpoint đã được patch!")
print("Keys mới:", list(ckpt.keys()))