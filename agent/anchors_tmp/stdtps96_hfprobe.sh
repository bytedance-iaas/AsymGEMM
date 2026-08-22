#!/bin/bash
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
echo "HF_ENDPOINT=${HF_ENDPOINT:-unset} HF_TOKEN=${HF_TOKEN:+set}"
.venv/bin/python - <<'PY'
import os, requests
for url in ["https://huggingface.co/api/models/openai/gpt-oss-20b",
            os.environ.get("HF_ENDPOINT","https://huggingface.co").rstrip("/")+"/api/models/openai/gpt-oss-20b"]:
    try:
        r = requests.head(url, timeout=15, headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN','')}"} if os.environ.get('HF_TOKEN') else {})
        print(url, "->", r.status_code)
    except Exception as e:
        print(url, "->", type(e).__name__, str(e)[:80])
from huggingface_hub import HfApi
try:
    info = HfApi().model_info("openai/gpt-oss-20b")
    print("model_info OK, siblings:", len(info.siblings))
except Exception as e:
    print("model_info FAIL:", type(e).__name__, str(e)[:200])
PY
