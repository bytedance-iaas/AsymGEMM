This is the current repo: /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM

There are other sibling repos that have other small devs indepdeont (to speeedup multipel features impletnaiton adn then merge later), but i dont think they have tooo many different cahnges anymore
/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM
/home/kevinni/AsymGEMM-SFT-46/third_party/AsymGEMM
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

The goal is to merge all 4 together here so that /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM this serves as the base for alll the new changes. 
After u merge these, lets run some heavy (near capacity) worklaod for Qwnw3 30b and Qwen3.5 35b and GLM's 2 MoEs near capacity. The goal is to confirm no regression in memory / latency / throughput. We had extenisvle records form vaiour repos abotut past perforamnce so just ensure that all these new changes will merge correctl wihtout regression. If there is, we need to diagnose them carefullt and try to fix all the issues. If the code change re very light we might not even need such validations for now.
When u run vaiaditons need to use asym45_enroot_run this starts the correct container that u will run the ssytem.

But again i dont think there is much deep code cahnges but stil be vry careult and careuflt make ur we retian 1. all the soruce code changes. no features should be lost. 2. improat scritps adn docs cahgnes no informatio shoud be lost.


But before u start the merging first do the disano and elt kenow how diffiucl is this 4 way merge i dont thi too nay cahnges too but assess carefully and let meknow.


Dont stop util this repo is a clear merge of all these individual repos.