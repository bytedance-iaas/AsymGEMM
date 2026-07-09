# Tensor NUMA placement — deliberate host-page policy for the 2-GPU + 2-node box

STATUS: DESIGN ONLY (2026-07-08, user-directed). NOT implemented. Companion to
fix_gb200_ep.md (sEP campaign; fabric + pool receipts referenced below).

## TLDR

Both GPUs are local to ONE NUMA node (node0); node1 is enlisted purely for extra
RAM. Total footprint (weights ~99 GB fabric + ~590 GB activation saves at s60000)
exceeds node0, so cross-socket traffic is UNAVOIDABLE — the design question is not
"avoid the hop" but "spend the hop on the traffic class that cannot feel it".
Today page placement is FIRST-TOUCH LUCK (bounded by NUMACTL_MEMBIND=0,1); it has
rolled well every session (a bad roll would balloon the 213 s-class steps — never
observed; but the PR-2 probe once measured 7.3 GB/s on a bad roll, so the hazard is
real). The policy below makes placement deliberate + receipt-checked.

THE RULE — bin by SENSITIVITY, not by volume:
  node0 (GPU-adjacent; latency/bandwidth-critical, synchronous consumers):
    (a) the shared weight fabric — every byte re-read ~3x/step (fwd + recompute +
        bwd-dX), TMA-streamed INSIDE GEMMs: a stall here stalls the MMA pipeline;
    (b) GEMM-STREAMED activation saves — the cpu_left LoRA-A forward operands and
        the cpu_right LoRA-grad saved X/S: host bytes read mid-kernel, exactly as
        latency-critical as weights (the correction that motivated this doc: NOT
        all activations are latency-blind);
    (c) Adam masters/optimizer state (CPU-thread-hot, small, ~30 GB/rank class).
  node1 (remote; high-volume, latency-blind):
    (d) copy-then-use activation pools — attention saves + unsloth-GC boundary
        tensors: async side-stream memcpy, event-chained, PREFETCHED before their
        consumer runs => only AVERAGE bandwidth matters, and the demand is trivial
        (~400-600 GB/step over ~215 s ~= 2-3 GB/s vs a Grace-Grace link with
        orders of magnitude more headroom).

Sanity numbers (s60000|8|1 pair, session 2026-07-08): weight streaming is
seq-INDEPENDENT (~170 GB/step); activation volume dominates at long seq (~400-600
GB/step) but is the tolerant class; at s2048 the ranking flips — the policy does
not care, because it bins by sensitivity, not volume.

## Context / receipts

- Topology: GB200-class box, 4 GPUs, campaign uses the 0,1 pair — BOTH local to
  node0's Grace; node1 = second Grace used as a RAM extension. ~1.69 TB total.
- Fabric: /dev/shm mmap, rank0 builds + copies banks (first touch => placement =
  wherever rank0's copy threads ran), ONE cudaHostRegister over the used range,
  both GPUs zero-copy TMA-read it (asym_gemm/training/shared_fabric.py).
- PR-2 receipts (scripts/testing/shared_fabric_probe.py): dual-rank concurrent
  streaming from one registered range sustained 156.2 GB/s/lane on a good
  placement; 7.3 GB/s once on a bad roll ("probes need numactl" note). The e2e
  never hit the bad roll (steps would scream), but nothing PREVENTS it today.
- Pinned pools: lazily cudaHostAlloc'd/registered on demand (cap = ceiling, not a
  reservation), recycled; DISTINCT allocation sites already exist per class
  (expact/lora-save pool vs attention_activation_offload wrapper buffers vs
  unsloth boundary buffers vs CPU-Adam masters) => per-class binding is
  implementable WITHOUT restructuring.
- Traffic classes, and why the bins are what they are:
    weights: highest per-byte reuse (every layer, every step), synchronous.
    cpu_left/cpu_right saves: read mid-GEMM during recompute/bwd — synchronous.
    attn/GC saves: 2 touches per byte per step (D2H once, H2D once), async,
      overlapped; the ONLY class big enough to need node1 anyway.
    masters: CPU-side hot loop (DeepSpeed Adam), small.

## Needed impls (staged; each independently landable + receipt-gated)

I1  PLACEMENT RECEIPT (measure first — zero behavior change):
    - After fabric seal and after first-step pool high-water: walk the key
      mappings with move_pages(2) (nodes=NULL query mode) or parse
      /proc/<pid>/numa_maps, and emit a heartbeat:
      {region -> {node0_pages, node1_pages}} for the fabric, each pool class,
      masters. Persist into profile.json.
    - GATE: receipts on both ranks; placement reproduces across two runs.

I2  FABRIC BIND (node0):
    - shared_fabric.py: after mmap, BEFORE rank0's copy-in: mbind(range,
      MPOL_BIND, nodemask={node0}) via ctypes/libnuma. Env
      ASYM_FABRIC_NUMA_NODE (default: the GPUs' local node, discoverable via
      /sys/bus/pci/devices/<gpu>/numa_node); die loudly if the bind fails.
    - Rank1's latecomer map inherits physical placement (same tmpfs file) — I1
      receipt confirms.
    - OPTION (measure, don't assume): MPOL_INTERLEAVE over {0,1} instead of
      BIND(node0) if node0's memory controller contends under 2x streaming +
      masters traffic; a PR-2-style A/B decides. Expectation: BIND wins because
      the bulk activation traffic is moving to node1 anyway.

I3  POOL-CLASS BINDS:
    - Copy-then-use pools (attn-act `_empty_strided_cpu_like`, unsloth boundary
      buffers, expact chunks) -> node1: scoped set_mempolicy(MPOL_BIND, {node1})
      around the allocation sites (allocate, restore policy), or mbind the fresh
      region before first touch. Pinning unaffected (register AFTER placement).
    - GEMM-streamed save pools (cpu_left/cpu_right X/S banks) -> node0.
    - CPU-Adam pinned masters -> node0.
    - Env kill-switch ASYM_NUMA_POLICY=0 restores first-touch (bisection aid).

I4  VALIDATION LADDER (B1-style, one change per run):
    - I1 alone (receipts only) -> I2 (fabric) -> I3 (pools). Per rung: s2048
      smoke parity + s60000 pair steady vs the banked 212.6/215.3 s twins.
      EXPECT: no step-time change on good-roll baselines (the win is
      variance-kill + a guarantee, not speed). RED FLAG: any step change > 2%
      either way => placement was NOT luck-neutral, attribute which class moved.
    - Adversarial receipt: force ASYM_FABRIC_NUMA_NODE=1 once and CONFIRM visible
      degradation — proves the hazard + receipt are real and calibrates the cost
      of a bad roll.

## Risks / notes

- mbind does NOT migrate already-faulted pages (needs MPOL_MF_MOVE +
  CAP_SYS_NICE) — bind BEFORE first touch everywhere; order is load-bearing.
- cudaHostRegister AFTER mbind is fine (pinning freezes physical pages where
  they landed); never register-then-bind.
- /dev/shm (tmpfs) honors per-mapping mempolicy; pages materialize at write time.
- The "1 CPU + NVMe" future variant is this same policy with tiers renamed:
  weights + GEMM-streamed saves keep RAM; copy-then-use saves absorb the NVMe
  tier (they are the latency-blind class by construction).
