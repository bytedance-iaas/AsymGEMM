I am trying to get tp throuhpgut reuslts from various baselines (zero3, superoffload, etc). Check the thrpugput plot right. However, now I also wanna do nemo which si https://github.com/NVIDIA-NeMo/Megatron-Bridge.git (some megatron-lm but wiht lora support) baiscal create a reproducible scripts/lf/bootstrap_nemo_venv.sh (and scripts/lf/bootstrap_nemo_venv_fa4.sh after we validat ethe bootstrap_nemo_venv.sh works)
Then this scitr hsodlcrea a .venv-nemo isntead. os ths all the nemo runs will us ethis envisne but eh emicsm etc hsod be exacl tthe same as our scripts/lf/profile_lora_lf_test_source.sh scripts/lf/profile_lora_lf_test_both.sh
but mly supporot nemo is fine.
This needs to be mult-rank normal EP and witho/wihtout actiaitno offloading. THis shoudl fail even agaisn superoffloand becasue nemo did not offload model weights. This is the expations
So help us uld this basleine jsu o shw that this fails as well *(even wiht some actiation offlaoding stil insufficeint)
Dont sotp unit this env ahs beebn supt for nemo and that we get througptu results form nemo proving that it sitll underproms for Q3 30b a3b and q3.5 35b on the same seq lengths.
DOnt stop until the evn is vladiaitn and these expected thrpgt[p resitls rare obtained
