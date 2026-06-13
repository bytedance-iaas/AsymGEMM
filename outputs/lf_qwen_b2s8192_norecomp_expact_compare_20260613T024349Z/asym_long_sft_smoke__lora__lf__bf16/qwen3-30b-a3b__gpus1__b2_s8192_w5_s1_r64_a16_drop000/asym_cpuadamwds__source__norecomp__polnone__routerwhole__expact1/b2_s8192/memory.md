# LF LoRA-SFT Memory

| Metric | MiB |
|---|---:|
| peak_allocated_hbm_bytes | 129345.36 |
| peak_reserved_hbm_bytes | 134948.00 |
| reserved_unallocated_bytes | 5602.64 |

## Persistent Tensor Accounting

These rows are exact tensor-size accounting for parameters, buffers, gradients, and host/pinned tensors. They are not a full peak allocated HBM attribution by themselves.

| Category | Component | Device | MiB |
|---|---|---|---:|
| host_weight | routed_experts | cpu | 55296.00 |
| pinned_host_weight | routed_experts | cpu | 55296.00 |
| trainable_param | routed_experts | gpu | 6336.00 |
| host_weight | attention | cpu | 1728.00 |
| pinned_host_weight | attention | cpu | 1728.00 |
| host_weight | embed_tokens | cpu | 593.50 |
| host_weight | lm_head | cpu | 593.50 |
| pinned_host_weight | lm_head | cpu | 593.50 |
| trainable_param | attention | gpu | 102.00 |
| host_weight | router | cpu | 24.00 |
| pinned_host_weight | router | cpu | 24.00 |
| host_weight | norms | cpu | 0.40 |
