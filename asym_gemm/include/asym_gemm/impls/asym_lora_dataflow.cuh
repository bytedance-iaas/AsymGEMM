#pragma once
// =============================================================================
// The training-dataflow asymmetric-GEMM design space (goal 1b-1, 2026-07-29).
//
// Every asym GEMM in this tree is one point in a 3-axis space. The upstream
// inference kernel occupies exactly one corner; the LoRA training dataflows
// force the opposite corners, and the AsymLoRA kernels are their
// instantiations. This header names the axes; the kernels static_assert
// their coordinates so the family claim is compile-checked, not prose.
//
//   axis 1 — StreamReuse: what the streamed (host-resident) operand meets.
//     kReusePipelined : streamed tile re-read by many output tiles → deep
//                       multi-buffered SMEM pipeline amortizes link bytes.
//     kFetchOnce      : reuse ≡ 1 (rank-64 LoRA) → single persistent slot;
//                       pipelining the stream would buy nothing.
//   axis 2 — GridKey: what the launch grid enumerates.
//     kOutputKeyed    : blocks ↔ output tiles (many per streamed tile).
//     kStreamKeyed    : blocks ↔ segments of the STREAM itself (row chunks;
//                       N0 adds sub-stream splits when segments are few —
//                       parallelism from splitting the stream, which an
//                       output-keyed engine cannot express).
//   axis 3 — Retirement: when accumulators leave the core.
//     kRetireEarly    : partial tiles retire to memory every reduction step
//                       (TMA reduce-add) — legal only because the reduction
//                       axis is RESIDENT.
//     kRetireStreamEnd: nothing retires until the streamed axis is fully
//                       consumed (the reduction RUNS OVER the stream);
//                       [rank × k-tile] accumulators are register-stationary
//                       per (sub)stream and write exactly once. N0's
//                       hierarchical form: per-substream residency + one merge.
//
//   Consumers (orthogonal multiplier): how many dataflows one stream feeds.
//     1 (upstream, by construction) · k adapters (pair=2 ✅, triple=3 ✅,
//     kNumOutputs in the K1 kernel) · fwd+grad simultaneously (N2, proposed).
//
//   Points:                       StreamReuse      GridKey       Retirement
//     upstream inference GEMM     kReusePipelined  kOutputKeyed  kRetireEarly
//     K1  S = X·Aᵀ  (cpu_left)    kFetchOnce       kStreamKeyed  kRetireEarly*
//     K2  dA = dSᵀ·X (cpu_right)  kFetchOnce       kStreamKeyed  kRetireStreamEnd
//   (*K1 retires per output tile AFTER its full resident-K reduction — early
//    along the stream axis is impossible for it because rows are independent;
//    its inversion lives on axes 1-2.)
// =============================================================================

namespace asym_gemm::lora_dataflow {

enum class StreamReuse { kReusePipelined, kFetchOnce };
enum class GridKey { kOutputKeyed, kStreamKeyed };
enum class Retirement { kRetireEarly, kRetireStreamEnd };

template <StreamReuse R, GridKey G, Retirement T, int kMaxConsumers>
struct Point {
    static constexpr StreamReuse reuse = R;
    static constexpr GridKey grid = G;
    static constexpr Retirement retire = T;
    static constexpr int max_consumers = kMaxConsumers;
};

// The three occupied points (compile-checked from the kernels).
using UpstreamInference = Point<StreamReuse::kReusePipelined, GridKey::kOutputKeyed,
                                Retirement::kRetireEarly, 1>;
using K1ForwardLoRA     = Point<StreamReuse::kFetchOnce, GridKey::kStreamKeyed,
                                Retirement::kRetireEarly, 3>;   // kNumOutputs ≤ 3
using K2AdapterGrad     = Point<StreamReuse::kFetchOnce, GridKey::kStreamKeyed,
                                Retirement::kRetireStreamEnd, 2>;  // pair grad

static_assert(K1ForwardLoRA::reuse != UpstreamInference::reuse &&
              K1ForwardLoRA::grid != UpstreamInference::grid,
              "K1 must invert the reuse and grid axes");
static_assert(K2AdapterGrad::retire != UpstreamInference::retire,
              "K2 must invert the retirement axis");

}  // namespace asym_gemm::lora_dataflow
