# sEP-planned (asym_sepplan2_cpuadamwds) 2-rank mirror campaign — ledger

User directive 2026-08-09: the 2-rank asym cells for GLM-4.5-Air (arena 240),
GLM-4.7-Flash, Mixtral-8x22B (arena 285) were measured with sDP
(asym_sdp2_cpuadamwds). Run **sepplan2** at the SAME seq lengths /
tiers / batches and compare throughput. TWO RUNNERS, opposite orders:
the other agent runs Air -> Flash -> Mixtral (shallow-first); c17 (this
ledger's runner) runs REVERSE: Flash-T2-deep (during mixtral fuse rebuild)
-> Mixtral 304k->32k -> Flash T1 832k->32k -> Air 320k->16k.

PROTOCOL: before each cell `git pull --rebase`; SKIP any (model,seq) with a
DONE line here (any runner); CLAIM before launch, DONE after harvest, push
each. Verdicts GOOM/COOM are honest walls (count as done). Cell recipe =
sdp2 twin: same tier token, same batch(walk), same arena cap, POL none,
w1+m2, GLOBAL tok/s = 2x per-rank eff. Fused mixtral ckpt is node-local —
c17 rebuilds it first (mx_fuse_local.py; bit-identical weights).

Anchors (sdp2, GLOBAL tok/s):
- mixtral-8x22b T1: 32k b8 4439 · 64k b4 3823 · 128k b2 2513 · 192k b1 1987
  · 256k b1 1635 · 288k b1 1129 · 304k b1 1110
- glm4.7-flash T1: 32k b12 7400 · 64k b6 4362 · 96k b4 3086 · 128k b3 2386
  · 160k b2 1934 · 192k b4 1526 · 256k b3 1152 · 320k b2 984 · 416k b2 721
  · 512k b1 617 · 576k b1 548 · 640k b1 493 · 704k b1 448 · 768k b1 405
  · 832k b1 371; T2: 896k b1 340 · 960k b1 313 · 1024k b1 294
- glm4.5-air T1 (arena 240): 16k b16 6844 · 32k b8 5646 · 48k b4 4730
  · 64k b4 4156 · 96k b2 3302 · 128k b2 2162 · 160k b(3,2) 1686
  · 192k b(2,1) 1573 · 256k b(2,1) 1233 · 320k b1 989

sepqueue2: NOT in scope for the mirror (judged: sepplan is the asked
variant; queue flavor only if sepplan shows pathology or time remains).

## Log (append-only; c17 = this runner)
- [08-10 00:53Z] c17 NOTE c17 fused mixtral ckpt absent on this node — rebuilding (mx_fuse_local.py) before mixtral cells
- [08-10 00:53Z] c17 CLAIM sepplan glm4.7-flash s=1024000 (spgf1024)
