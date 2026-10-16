#!/usr/bin/env python3
"""One-shot script: null out receptor layer prototypes so TID re-initializes to K=256 on next startup.
Run ONLY after TID is fully stopped and BEFORE it starts again.
"""
import json
import os
import time

STATE_FILE = "/home/josh/The-Inference-Difference/ng_lite_state.json"
BACKUP = f"{STATE_FILE}.bak.{int(time.time())}"

with open(STATE_FILE) as f:
    state = json.load(f)

# Backup first
import shutil
shutil.copy2(STATE_FILE, BACKUP)
print(f"Backup: {BACKUP}")

rl = state.get("receptor_layer", {})
before_protos = len(rl.get("prototypes") or [])
before_counts = rl.get("prototype_counts")

rl["prototypes"] = None
rl["prototype_counts"] = None
state["receptor_layer"] = rl

with open(STATE_FILE, "w") as f:
    json.dump(state, f)

print(f"Reset: {before_protos} prototypes → null")
print(f"Reset: prototype_counts {before_counts} → null")
print(f"input_count preserved: {rl.get('input_count')}")
print("TID will reinitialize to K=256 on next startup after 256 warmup requests.")
