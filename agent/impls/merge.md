This is the current repo: /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM

There are other sibling repos that have other small devs indepdeont (to speeedup multipel features impletnaiton adn then merge later), but i dont think they have tooo many different cahnges anymore
/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM
/home/kevinni/AsymGEMM-SFT-46/third_party/AsymGEMM
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
Each sibling has their own LlamaFactory folder as well for eahc AsymGEMM-SFT-39 AsymGEMM-SFT-46 AsymGEMM-SFT

The goal is to merge all 4 together here so that /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM this serves as the base for alll the new changes. and that its /home/kevinni/AsymGEMM-SFT-38/third_party/LlamaFactory has all the sibling consolidation too.
After u merge these, lets run some heavy (near capacity) worklaod for Qwnw3 30b and Qwen3.5 35b and GLM's 2 MoEs near capacity. The goal is to confirm no regression in memory / latency / throughput. We had extenisvle records form vaiour repos abotut past perforamnce so just ensure that all these new changes will merge correctl wihtout regression. If there is, we need to diagnose them carefullt and try to fix all the issues. If the code change re very light we might not even need such validations.
When u run vaiaditons need to use asym40_enroot_run this starts the correct container that u will run the ssytem.

But again i DONT think there is much deep code cahnges but stil be vry careult and careuflt make ur we retian 1. all the soruce code changes. no features should be lost. 2. incorporate scritps and docs cahgnes no informatio shoud be lost.
Becasue honeslty all the gpus here on this machine are wedged.
All the repos are alreuad dmeo prignal to relev the same strufu so they hsjldnbe easilt merged. Afain dont lose any nontrial features and docs and artifacts (results of runs). If there is no reaon to rerurn vlaidatinos tho (liek mostl jsut trivial code chaneg and scfrip changes we dont need validation runs)

Dont stop util this repo is a clear merge of all these individual repos.


