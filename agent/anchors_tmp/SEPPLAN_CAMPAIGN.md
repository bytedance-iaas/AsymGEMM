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
- [08-10 06:52Z] c17 DONE sepplan glm4.7-flash s=1024000 (spgf1024) -> TRAINED | spgf1024   glm4.7-flash   s=1024000 b1   sepplan=  297 sdp2=294 delta=  +1.1% resv= 181.4G rss= 466G
- [08-10 06:52Z] c17 CLAIM sepplan glm4.7-flash s=960000 (spgf960)
- [08-10 12:10Z] c17 DONE sepplan glm4.7-flash s=960000 (spgf960) -> TRAINED | spgf960    glm4.7-flash   s= 960000 b1   sepplan=  317 sdp2=313 delta=  +1.3% resv= 181.5G rss= 479G
- [08-10 12:10Z] c17 CLAIM sepplan glm4.7-flash s=896000 (spgf896)
- [08-10 16:47Z] c17 DONE sepplan glm4.7-flash s=896000 (spgf896) -> TRAINED | spgf896    glm4.7-flash   s= 896000 b1   sepplan=  340 sdp2=340 delta=  +0.0% resv= 181.4G rss= 483G
- [08-10 16:47Z] c17 CLAIM sepplan mixtral-8x22b s=304000 (spmx304)
- [08-10 17:23Z] c17 DONE sepplan mixtral-8x22b s=304000 (spmx304) -> TRAINED | spmx304    mixtral-8x22b  s= 304000 b1   sepplan= 1075 sdp2=1110 delta=  -3.1% resv= 181.8G rss= 700G
- [08-10 17:23Z] c17 CLAIM sepplan mixtral-8x22b s=288000 (spmx288)
- [08-10 17:57Z] c17 DONE sepplan mixtral-8x22b s=288000 (spmx288) -> TRAINED | spmx288    mixtral-8x22b  s= 288000 b1   sepplan= 1062 sdp2=1129 delta=  -6.0% resv= 181.8G rss= 722G
- [08-10 17:57Z] c17 CLAIM sepplan mixtral-8x22b s=256000 (spmx256)
- [08-10 18:19Z] c17 DONE sepplan mixtral-8x22b s=256000 (spmx256) -> TRAINED | spmx256    mixtral-8x22b  s= 256000 b1   sepplan= 1591 sdp2=1635 delta=  -2.7% resv= 175.7G rss= 722G
- [08-10 18:20Z] c17 CLAIM sepplan mixtral-8x22b s=192000 (spmx192)
- [08-10 18:36Z] c17 DONE sepplan mixtral-8x22b s=192000 (spmx192) -> TRAINED | spmx192    mixtral-8x22b  s= 192000 b1   sepplan= 1925 sdp2=1987 delta=  -3.1% resv= 133.3G rss= 722G
- [08-10 18:36Z] c17 CLAIM sepplan mixtral-8x22b s=128000 (spmx128)
- [08-10 18:52Z] c17 DONE sepplan mixtral-8x22b s=128000 (spmx128) -> TRAINED | spmx128    mixtral-8x22b  s= 128000 b2   sepplan= 2440 sdp2=2513 delta=  -2.9% resv= 173.8G rss= 723G
- [08-10 18:52Z] c17 CLAIM sepplan mixtral-8x22b s=64000 (spmx64)
- [08-10 19:05Z] c17 DONE sepplan mixtral-8x22b s=64000 (spmx64) -> TRAINED | spmx64     mixtral-8x22b  s=  64000 b4   sepplan= 3668 sdp2=3823 delta=  -4.1% resv= 173.2G rss= 715G
- [08-10 19:05Z] c17 CLAIM sepplan mixtral-8x22b s=32000 (spmx32)
- [08-10 19:16Z] c17 DONE sepplan mixtral-8x22b s=32000 (spmx32) -> TRAINED | spmx32     mixtral-8x22b  s=  32000 b8   sepplan= 4269 sdp2=4439 delta=  -3.8% resv= 176.2G rss= 722G
- [08-10 19:16Z] c17 CLAIM sepplan glm4.7-flash s=832000 (spgf832)
- [08-10 23:13Z] c17 DONE sepplan glm4.7-flash s=832000 (spgf832) -> TRAINED | spgf832    glm4.7-flash   s= 832000 b1   sepplan=  370 sdp2=371 delta=  -0.2% resv= 181.8G rss= 354G
- [08-10 23:13Z] c17 CLAIM sepplan glm4.7-flash s=768000 (spgf768)
- [08-11 02:33Z] c17 DONE sepplan glm4.7-flash s=768000 (spgf768) -> TRAINED | spgf768    glm4.7-flash   s= 768000 b1   sepplan=  404 sdp2=405 delta=  -0.3% resv= 181.8G rss= 354G
- [08-11 02:33Z] c17 CLAIM sepplan glm4.7-flash s=704000 (spgf704)
- [08-11 05:22Z] c17 DONE sepplan glm4.7-flash s=704000 (spgf704) -> TRAINED | spgf704    glm4.7-flash   s= 704000 b1   sepplan=  447 sdp2=448 delta=  -0.2% resv= 173.2G rss= 353G
- [08-11 05:22Z] c17 CLAIM sepplan glm4.7-flash s=640000 (spgf640)
- [08-11 07:42Z] c17 DONE sepplan glm4.7-flash s=640000 (spgf640) -> TRAINED | spgf640    glm4.7-flash   s= 640000 b1   sepplan=  492 sdp2=493 delta=  -0.3% resv= 158.8G rss= 353G
- [08-11 07:42Z] c17 CLAIM sepplan glm4.7-flash s=576000 (spgf576)
