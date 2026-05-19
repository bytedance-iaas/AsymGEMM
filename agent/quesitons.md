lora/adapters on CPU or GPU?

lora or full finetuning?

move to GH200 or BH200?

Wht ar ethe technical details needed to make this a research paper?

Design a CPU layout that serves both W.T and W access efficiently? Done

or write a second direct-fetch kernel whose tile mapping reads original W efficiently for dX?

Fused LoRA inside the kernel

To accomodate QLoRA do dequant inside the kernel

