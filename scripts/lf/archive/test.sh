CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import os, torch, time
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count =", torch.cuda.device_count())
print("current_device =", torch.cuda.current_device())
x = torch.empty((1024,1024,1024), device="cuda")
print("allocated on", torch.cuda.get_device_name(0))
time.sleep(60)
PY
