# third-party/

Vendored dependencies live here. Two kinds:

### Build submodules (tracked, git submodules)

Needed to JIT-compile the AsymGEMM CUDA kernels. Pulled in with
`git submodule update --init --recursive`:

- `cutlass/` — NVIDIA CUTLASS / CuTe
- `fmt/` — {fmt} formatting library

### Runtime dependency repos (NOT tracked — you install them here)

The LoRA-SFT + profiling harness under `../scripts/lf/` expects these repos to be
present in this directory. They are `.gitignore`d (each is its own upstream repo); clone
them here **before** running `../scripts/lf/bootstrap_lf_venv.sh`:

| Path (under `third-party/`) | Repo | Notes |
| --- | --- | --- |
| `LlamaFactory/`  | Your AsymGEMM-integrated LLaMA-Factory fork | apply your LF patch here; installed editable |
| `deepspeed/`     | DeepSpeed source tree | ZeRO-3 / SuperOffload / CPUAdam backends |
| `Liger-Kernel/`  | Liger-Kernel | fused Triton loss/kernels |

Example:

```bash
cd third-party
git clone <your-llama-factory-fork-url> LlamaFactory
git clone https://github.com/deepspeedai/DeepSpeed.git deepspeed
git clone https://github.com/linkedin/Liger-Kernel.git Liger-Kernel
```

Then bootstrap the environment (creates `../.venv` and pip-installs each repo editable):

```bash
cd ..
bash scripts/lf/bootstrap_lf_venv.sh
```

The bootstrap and run scripts resolve these paths relative to this repo
(`<AsymGEMM>/third-party/<repo>`). Override any of them with `LF_DIR=`, `DEEPSPEED_DIR=`,
or `LIGER_DIR=` if you keep a repo elsewhere. See `../scripts/lf/README.md` for the full
install → run flow.
