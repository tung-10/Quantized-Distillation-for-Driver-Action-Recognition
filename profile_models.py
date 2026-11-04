# profile_models.py
import torch
import time
import os

# ── Cấu hình ─────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 1
N_FRAMES    = 16
IMG_SIZE    = 224
N_WARMUP    = 10
N_REPEAT    = 50
NUM_CLASSES = 7

# Path checkpoint quantized MobileNet — sửa lại đúng path của bạn
QUANT_CKPT_PATH = "./model/StudentTeacherQuantization_VideoMAE/exp0/checkpoint_latest.pth"
QUANT_BACKEND   = "fbgemm"   # hoặc "qnnpack" nếu chạy ARM/mobile

# ── Helpers ───────────────────────────────────────────────────────────────────
import torch

ckpt = torch.load(QUANT_CKPT_PATH, map_location="cpu")
sd = ckpt["int_student_model_state_dict"]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def model_size_mb(model):
    tmp = "/tmp/_profile_tmp.pth"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 ** 2)
    os.remove(tmp)
    return size

def measure_latency(model, dummy_input, device=None):
    dev = device or DEVICE
    model.eval().to(dev)
    x = dummy_input.to(dev)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(N_REPEAT):
                model(x)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / N_REPEAT
        else:
            t0 = time.perf_counter()
            for _ in range(N_REPEAT):
                model(x)
            return (time.perf_counter() - t0) / N_REPEAT * 1000

def measure_peak_memory(model, dummy_input):
    if DEVICE != "cuda":
        return -1
    model.eval().to(DEVICE)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(dummy_input.to(DEVICE))
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def measure_flops(model, dummy_input):
    try:
        from thop import profile as thop_profile
        model.eval().cpu()
        flops, _ = thop_profile(model, inputs=(dummy_input.cpu(),), verbose=False)
        return flops / 1e9
    except ImportError:
        print("  [!] pip install thop để đo FLOPs")
        return -1

def print_report(name, model, dummy_input, device=None):
    dev = device or DEVICE
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        params = count_parameters(model)
        print(f"  Parameters     : {params/1e6:.2f} M")
    except Exception:
        print(f"  Parameters     : N/A (quantized int8)")

    size = model_size_mb(model)
    print(f"  Model size     : {size:.1f} MB")

    lat = measure_latency(model, dummy_input, device=dev)
    print(f"  Latency        : {lat:.2f} ms  (batch={BATCH_SIZE}, device={dev})")
    print(f"  Throughput     : {1000/lat*BATCH_SIZE:.1f} samples/s")

    if dev == "cuda":
        mem = measure_peak_memory(model, dummy_input)
        if mem > 0:
            print(f"  Peak GPU mem   : {mem:.1f} MB")

    gflops = measure_flops(model, dummy_input)
    if gflops > 0:
        print(f"  FLOPs          : {gflops:.2f} GFLOPs")
    print(f"{'='*60}")


# ── Load quantized MobileNet (giống driver.py) ────────────────────────────────

def load_quantized_mobilenet_for_profile(ckpt_path, config_file):
    from architecture.mobilenet import MobileNetLogit
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Auto-detect width_mult
    sd_int  = ckpt["int_student_model_state_dict"]
    sd_float = ckpt["float_student_model_state_dict"]
    out_ch     = sd_int["features.0.0.weight"].shape[0]
    width_mult = out_ch / 32.0
    print(f"[Quantized] Detected width_mult = {width_mult}")
    arch_config = {"width_mult": width_mult, "dropout_prob": 0.0}
    # ── Model float: để đo LATENCY (torch.quantization Conv3d không chạy được CPU) 
    model_float = MobileNetLogit(config_file, arch_config)
    model_float.load_state_dict(sd_float, strict=False)
    model_float.eval().cpu()
    # ── Đo model size thực của int8 bằng cách lưu int state_dict ──────
    tmp = "/tmp/_int8_size.pth"
    torch.save(sd_int, tmp)
    int8_size_mb = os.path.getsize(tmp) / (1024 ** 2)
    os.remove(tmp)
    # ── Đếm params (từ float weights) ────────────────────────────────
    n_params = sum(p.numel() for p in model_float.parameters())
    print(f"[Quantized] ✅ Ready")
    print(f"[Quantized] int8 weights size = {int8_size_mb:.1f} MB")
    return model_float, int8_size_mb, n_params

def print_report_quant(name, model_float, int8_size_mb, n_params, dummy_input):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Parameters     : {n_params/1e6:.2f} M")
    print(f"  Model size (int8 weights) : {int8_size_mb:.1f} MB")
    lat = measure_latency(model_float, dummy_input, device="cpu")
    print(f"  Latency (float, CPU)      : {lat:.2f} ms  (batch={BATCH_SIZE})")
    print(f"  Throughput                : {1000/lat*BATCH_SIZE:.1f} samples/s")
    print(f"  Note: Latency là float32 trên CPU — int8 Conv3D chưa được")
    print(f"        PyTorch hỗ trợ (chỉ có Conv1D/2D). Kích thước là thực.")
    print(f"{'='*60}")
# ── Khởi tạo models ───────────────────────────────────────────────────────────

config_file = {"train": {"num_classes": NUM_CLASSES}}
arch_config = {"width_mult": 1.0, "dropout_prob": 0.0}

dummy_video = torch.randn(BATCH_SIZE, 3, N_FRAMES, IMG_SIZE, IMG_SIZE)

# --- VideoMAE-Large ---
from architecture.videomae import VideoMAELogit
videomae = VideoMAELogit(config_file, {"dropout_prob": 0.0, "name": "vit_large"})

# --- MobileNet-3D float32 (không quantized) ---
from architecture.mobilenet import MobileNetLogit
mobilenet_fp32 = MobileNetLogit(config_file, arch_config)

# --- MobileNet-3D quantized int8 từ checkpoint ---
# mobilenet_int8 = load_quantized_mobilenet(
#     ckpt_path   = QUANT_CKPT_PATH,
#     config_file = config_file,
# )

# ── Chạy đánh giá ─────────────────────────────────────────────────────────────

print_report("VideoMAE-Large       (ViT-L, float32)", videomae,       dummy_video, device=DEVICE)
print_report("MobileNet-3D         (float32)",        mobilenet_fp32, dummy_video, device=DEVICE)
model_float, int8_size_mb, n_params = load_quantized_mobilenet_for_profile(
    ckpt_path   = QUANT_CKPT_PATH,
    config_file = config_file,
)
print_report_quant(
    "MobileNet-3D Quant (int8 weights, 0.5x)",
    model_float, int8_size_mb, n_params, dummy_video
)
