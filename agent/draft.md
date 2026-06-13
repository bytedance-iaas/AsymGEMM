

1. 

2. windows needed?
  for rows in row_chunks:
      act_hbm = stage(act_cpu[rows])          # only this chunk in HBM
      y_base = act_hbm @^ W_down_cpu.T
      s_down = act_hbm @ A_down.T             # or act_cpu[rows] @^^ A_down.T
      y = y_base + scale * (s_down @ B_down.T)
      write y into output
      release act_hbm, s_down


3. i ahve a question 
Method 1 
S_down    = act_cpu @^^ A_down^T                      # [M, r] HBM
LoRA_down = scale * (S_down @ B_down^T)               # [M, H] Temp
S_down_cpu = offload(S_down)

Method 2
S_down    = act_cpu @^^ A_down^T                      # [M, r] HBM
S_down_cpu = offload(S_down)
LoRA_down = scale * (S_down_cpu @^^ B_down^T)               # [M, H] Temp

These 2 actually the same peak mmeory usage right?



4. Why KT makes sense? beause kt only offloads experts but deepspeed can offload anything. What is the motivation for kt comparing to deepspeed?
