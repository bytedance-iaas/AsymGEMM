1. /home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM this current repo has a lot to new model additions 
2. /home/kevinni/AsymGEMM-SFT-46/third_party/AsymGEMM this toehr repo has some chagns to run qwwn3.5 thprgith resutls not sure if tehra re changes made tho might be only scripts level changes please amke sures
3. /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM here are some kenrel changes mainly i believe.

The goal is to merge all 3 together here so that /home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM
this serves as the base for alll the new changes. After u merge these 3, lets run some heavy (near capacity) worklaod for Qwnw3 30b and Qwen3.5 35b and GLM's 2 MoEs near capacity. 
The goal is to confirm no regression in memory / latency / throughput. We had extenisvle records form vaiour repos abotut past perforamnce so just ensure that all these new changes will merge correctl wihtout regression. If there is we need to diagnose them carefullt and try to fix all the issues. Dont stop util this repo is a clear merge of  AsymGEMM-SFT-39/third_party/AsymGEMM and AsymGEMM-SFT-46/third_party/AsymGEMM and AsymGEMM-SFT/third_party/AsymGEMM

Remember when testing the code run asym42_enroot_run this starts the current container so that u can run the testing code.
NEVER run on the host directly.
After these merges are done u can contnue to merge this dir as well
DOnt miss any critical details/features but alos dont keep old/stale paths that is not used annymore at all.
use /home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/impls/merge_progress.md
as a record for the merging. for conflcit resolving decision etc dont ask me, do ur best judgsment to resolve conflicts and to move forward. 
Dont stop until all these merges are confirmed to be correct / no regression on these MoE models with near capacity workload.

4. /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM this was working on a very old path. 
it was trying to improve the model capacity it ahd some code tjhat assists laoding large models but not sure it stil applcaitbi or how to adapt ont the current repo. Be very caretuf cauwe this repo was branhc off a ver old aoth. Just chek what the its nchange smade to bebakbl to load larger models otherwise dont take any old / stale paths.


