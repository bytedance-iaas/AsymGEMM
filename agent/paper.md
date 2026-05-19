1. MoE direct-fetch training runtime
      - Not generic MoE routing.
      - The contribution is making routed expert forward + backward-input work when expert weights
        stay in CPU memory.
      - Needs CUDA packing/scatter/backward, grouped expert execution, empty/skewed expert
        handling.
      - This is real systems work.
2. Placement scheduler
    - Decide per layer/expert:

    direct CPU fetch vs staged H2D copy vs GPU-resident

    - Based on active tokens, expert skew, shape, HBM budget, bandwidth.
    - This is probably the strongest paper contribution because direct fetch will not always
    win.

3. Host-weight training runtime
    - Not just pin_memory.
    - Persistent CPU weight pools, forward/backward layouts, descriptor table, resume/rebuild,
    NUMA placement, no accidental HBM copies.
    - This is paper-worthy only if it is robust and measured, not just a wrapper.
4. End-to-end backend evidence
    - Real LLaMA-Factory MoE SFT.
    - Compare against KTransformers AMX BF16, optimized staging, GPU-resident.
    - Show HBM savings and speed/break-even curves.

