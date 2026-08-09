"""A/B: temporarily drop jamba from the asym generic-moe liger set (class-path only)."""
import sys
path = "/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/asym_gemm/integrations/liger_loss.py"
s = open(path).read()
tag = '    "jamba",\n'
if sys.argv[1] == "off" and tag in s:
    s = s.replace(tag, '    # "jamba",  # AB: class-path-only test\n')
elif sys.argv[1] == "on":
    s = s.replace('    # "jamba",  # AB: class-path-only test\n', tag)
open(path, "w").write(s)
print("bridge", sys.argv[1])
