#!/usr/bin/env python3
"""DECISIVE mind raw-amplitude probe (corrected: payload is a msgpack SOURCE
DICT, not f32). Samples the actual L/R inputs feeding inner_mind, reproduces the
raw 15D via the production collect_mind_15d, and compares raw-vs-output breath.
Answers H1 (L/R inputs themselves low-amplitude → producer) vs H2 (inputs vary
but §G5.2 spring over-damps the output → kernel)."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/antigravity/projects/titan")
import msgpack
from titan_hcl.core.state_registry import RegistrySpec, StateRegistryReader
from titan_hcl.api.shm_reader_bank import ShmReaderBank
import titan_hcl.logic.inner_mind_sensor_refresh as inn
from titan_hcl.logic.mind_tensor import collect_mind_15d

root = ShmReaderBank("T1").shm_root
spec = RegistrySpec(name=inn.SLOT_NAME, dtype=np.dtype("uint8"),
                    shape=(inn.MAX_PAYLOAD_BYTES,),
                    schema_version=inn.SCHEMA_VERSION, variable_size=True)
r = StateRegistryReader(spec, root)
bank = ShmReaderBank("T1")

# scalar input fields we track for variance
def flatten_inputs(d):
    return {
        "think0_memory": d["thinking_5d"][0], "think1_social": d["thinking_5d"][1],
        "think2_media": d["thinking_5d"][2], "think3_mood": d["thinking_5d"][3],
        "think4_knowledge": d["thinking_5d"][4],
        "audio_creates": d["audio_state"]["creates_recent"], "audio_ambient": d["audio_state"]["ambient"],
        "interaction_q": d["interaction_quality"],
        "visual_creates": d["visual_state"]["creates_recent"], "visual_ambient": d["visual_state"]["ambient"],
        "assessment_q": d["assessment_quality"], "ambient_change": d["ambient_change"],
        **{f"horm_{k}": v for k, v in d["hormone_levels"].items()},
    }

inputs_log, raw15_log, out_log = [], [], []
t_end = time.monotonic() + 70.0
last = None
while time.monotonic() < t_end:
    b = r.read_variable()
    if b:
        try:
            d = msgpack.unpackb(b, raw=False, strict_map_key=False)
            inputs_log.append(flatten_inputs(d))
            raw15 = collect_mind_15d(
                d["thinking_5d"], audio_state=d["audio_state"],
                interaction_quality=d["interaction_quality"], visual_state=d["visual_state"],
                assessment_quality=d["assessment_quality"], ambient_change=d["ambient_change"],
                hormone_levels=d["hormone_levels"])
            raw15_log.append(np.array(raw15, dtype=float))
        except Exception as e:
            pass
    tr = bank.read_trinity()
    if tr:
        out_log.append(np.array(tr["full_130dt"], dtype=float)[5:20])
    time.sleep(0.4)

print(f"\n=== INNER MIND INPUT-VARIANCE PROBE (T1, {len(inputs_log)} samples / ~70s) ===\n")
# 1) which L/R inputs actually move?
keys = list(inputs_log[0].keys())
print("L/R INPUT field std (over the window) — does the force vary?:")
for k in keys:
    vals = [float(s[k]) for s in inputs_log]
    print(f"  {k:20s} mean={np.mean(vals):.4f}  std={np.std(vals):.5f}  min={min(vals):.3f} max={max(vals):.3f}")

# 2) raw 15D vs output 15D breath
R = np.vstack(raw15_log); O = np.vstack(out_log)
block = [("Thinking[0:5]",0,5),("Feeling[5:10]",5,10),("Willing[10:15]",10,15)]
print("\nRAW 15D (reproduced via collect_mind_15d) vs OUTPUT 15D breath:")
print(f"  {'block':16s} {'raw_std':>8s} {'out_std':>8s}  ratio out/raw")
for name,a,c in block:
    rs=R[:,a:c].std(0).mean(); os=O[:,a:c].std(0).mean()
    print(f"  {name:16s} {rs:>8.4f} {os:>8.4f}  {('%.2f'%(os/rs)) if rs>1e-9 else 'n/a'}")
rs=R.std(0).mean(); os=O.std(0).mean()
print(f"  {'ALL 15D':16s} {rs:>8.4f} {os:>8.4f}  {('%.2f'%(os/rs)) if rs>1e-9 else 'n/a'}")
print("\nINTERPRETATION:")
print("  raw_std LOW (~out_std)  -> H1: L/R inputs themselves barely move (producer/force weakness)")
print("  raw_std HIGH, out_std LOW -> H2: inputs move but spring over-damps output (kernel)")
