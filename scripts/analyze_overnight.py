#!/usr/bin/env python3
"""Morning analysis of the 6h T3 overnight run (PROFILING.md leak/CPU/stability).

Pulls /tmp/overnight/{rss,box,cpu}.jsonl + /tmp/gc_leak_*.jsonl from T3 and emits:
  1. LEAK: per-worker gc object-type growth, split at the boot-plateau — growth
     AFTER plateau = a real slow leak (the 23-min window couldn't see this).
  2. RSS: per-worker PSS trajectory + slope + restart count (stability).
  3. CPU: per-worker cpu_pct trajectory + dominant on-CPU frame over the run.
  4. BOX: load/mem/swap/disk first→last (memory-pressure trajectory).
Usage: python scripts/analyze_overnight.py [--host root@10.135.0.6] [--pull]
"""
import json, subprocess, sys, glob, os

HOST = "root@10.135.0.6"
if "--host" in sys.argv: HOST = sys.argv[sys.argv.index("--host")+1]

def pull():
    subprocess.run(["ssh", HOST, "tar -czf /tmp/overnight_bundle.tgz -C /tmp overnight $(cd /tmp && ls gc_leak_*.jsonl 2>/dev/null)"], capture_output=True)
    subprocess.run(["scp", "-q", f"{HOST}:/tmp/overnight_bundle.tgz", "/tmp/"], capture_output=True)
    subprocess.run(["tar", "-xzf", "/tmp/overnight_bundle.tgz", "-C", "/tmp/"], capture_output=True)

def load(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError: return []

def hours(S): return (S[-1]["ts"]-S[0]["ts"])/3600 if len(S)>1 else 0

print("="*70)
print("1) LEAK — per-worker gc object growth (boot-ramp vs post-plateau slope)")
for f in sorted(glob.glob("/tmp/gc_leak_*.jsonl")):
    w=f.split("gc_leak_")[1].rsplit(".jsonl",1)[0]; S=load(f)
    if len(S)<6: print(f"  {w}: {len(S)} samples (too few)"); continue
    h=hours(S); plateau=next((i for i,s in enumerate(S) if s["ts"]-S[0]["ts"]>=300), len(S)//4)
    def fn(s): return s["top"].get("function",0)
    boot=fn(S[plateau])-fn(S[0])
    post=fn(S[-1])-fn(S[plateau]); post_min=(S[-1]["ts"]-S[plateau]["ts"])/60
    obj_post=S[-1]["total_objects"]-S[plateau]["total_objects"]
    flag=" 🔴 SLOW LEAK" if post>500 and post/max(post_min,1)>2 else " ✓ flat (no leak)"
    print(f"  {w:22} {h:.1f}h | boot-ramp fn +{boot:,} | POST-plateau fn {post:+,} ({post/max(post_min,1):+.1f}/min), objects {obj_post:+,}{flag}")

print("="*70)
print("2) RSS — per-worker PSS trajectory + restarts")
R=load("/tmp/overnight/rss.jsonl")
if R:
    h=hours(R); names=set()
    for s in R: names|=set(s["w"].keys())
    rows=[]
    for w in names:
        ser=[(s["ts"],s["w"][w]["pss"],s["w"][w]["start"]) for s in R if w in s["w"]]
        if len(ser)<3: continue
        starts={x[2] for x in ser}; d=ser[-1][1]-ser[0][1]
        rows.append((d,w,ser[0][1],ser[-1][1],d/max(h,0.1),len(starts)-1))
    print(f"  over {h:.1f}h:  {'WORKER':22}{'first':>7}{'last':>7}{'ΔMB':>7}{'MB/h':>7}{'restarts':>9}")
    for d,w,f0,fl,rate,rs in sorted(rows,reverse=True)[:14]:
        print(f"  {'':10}{w:22}{f0:7.0f}{fl:7.0f}{d:+7.0f}{rate:+7.1f}{rs:9d}")
print("="*70)
print("3) CPU — per-worker (cpu_pct trajectory + dominant frame)")
C=load("/tmp/overnight/cpu.jsonl")
if C:
    agg={}
    for s in C:
        for w,d in s["w"].items(): agg.setdefault(w,[]).append((d["cpu_pct"],d["top"][0] if d["top"] else None))
    for w,v in sorted(agg.items(),key=lambda x:-max(p for p,_ in x[1])):
        cps=[p for p,_ in v]; tops=[t[0] for _,t in v if t]
        from collections import Counter
        dom=Counter(tops).most_common(1)
        print(f"  {w:22} cpu%: min={min(cps):.1f} max={max(cps):.1f} avg={sum(cps)/len(cps):.1f} | dominant frame: {dom[0][0] if dom else '?'}")
print("="*70)
print("4) BOX — resource trajectory")
B=load("/tmp/overnight/box.jsonl")
if B:
    a,b=B[0],B[-1]
    print(f"  over {hours(B):.1f}h ({len(B)} samples):")
    print(f"    load(1m): {a['load'][0] if a['load'] else '?'} → {b['load'][0] if b['load'] else '?'}")
    print(f"    mem avail: {a['mem_mb'].get('MemAvailable','?')} → {b['mem_mb'].get('MemAvailable','?')} MB")
    sa=a['mem_mb'].get('SwapTotal',0)-a['mem_mb'].get('SwapFree',0); sb=b['mem_mb'].get('SwapTotal',0)-b['mem_mb'].get('SwapFree',0)
    print(f"    swap used: {sa} → {sb} MB ({sb-sa:+} MB)")
    print(f"    disk: {a['disk']} → {b['disk']}")
