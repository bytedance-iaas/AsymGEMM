#!/bin/bash
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_lib.sh
for t in s2q35sep768-c18_q3_5-35b-a3b__b1_s768000_ga1_drop000 s2q35sep768b-c18_q3_5-35b-a3b__b1_s768000_ga1_drop000 nonexistent; do
  printf '%s -> ' "$t"; verdict "$t" "$LOGD/r_s2q35sep768_b1.try1.log"
done
echo "harvest: $(harvest s2q35sep768-c18_q3_5-35b-a3b__b1_s768000_ga1_drop000 2)"
