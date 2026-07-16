i am trying to merge 2 verison of the same repo: this current one and /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

This one i was working on fixing the qwen3.5 memory usage etc.
The other one was working on testing memory and latency and might did some fixes as well.
the other one is up to date with main_kevin
I wanna push the current folder's work as a backup branch call it main_kevin_qwen35
And then switch back to main_kevin and merge /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM onto this one. 
so that we have all the changes included and up to date
Note any nontrivial conflicts that u want my opinion to resolve. I assuma a lot of them will be only triival conflits or no conflcits at all

After all the merges are done we need to do some testing on qwen3.5 features to ensure that it did not break. CHekc any previous records forn th qwne3.5 runs in laencya dn memory as a reference.
BUT VERY IMPORTANTLY we need to run experiments within containers for this machine it is asym39_enroot_run this starts the right container to run tests etc. Note that we NEVER run outside the container directly.

