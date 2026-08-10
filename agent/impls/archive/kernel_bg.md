# AsymGEMM vs DeepGEMM

- **Diff 1 — where an operand lives:** DeepGEMM's inputs all sit in
  GPU memory. AsymGEMM moves one operand out to CPU memory, so every
  tile of it must arrive over the slow CPU-GPU wire — the wire
  becomes part of the kernel's design.

- **Diff 2 — what one block does:** DeepGEMM's block owns one output
  tile, makes one pass, and is done. AsymGEMM's block pins each
  wire-crossed tile and walks all tokens against it before fetching
  the next — two nested loops instead of one, so a single expensive
  crossing gets reused thousands of times.

- **Diff 3 — where the sum completes:** DeepGEMM finishes the whole
  sum on-chip and writes the output once. AsymGEMM's sum completes
  in GPU memory: the output is revisited and added to once per
  slice, many times over.

- **Diff 4 — one supply line vs two:** DeepGEMM feeds compute from
  one uniform conveyor of tiles. AsymGEMM runs two different supply
  lines — the held wire-fed tile, and a fast conveyor of
  GPU-resident tiles — each with its own coordination.
