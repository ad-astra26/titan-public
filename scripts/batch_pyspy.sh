#!/bin/bash
# Batch py-spy: for each worker name, resolve its pid (by cwd+setproctitle),
# sample on-CPU for DURATION, print top-6 self-time frames. Args: REPO DURATION WORKER...
PY=/home/antigravity/projects/titan/test_env/bin/py-spy
REPO="$1"; DUR="$2"; shift 2
for W in "$@"; do
  PID=$(for p in $(pgrep -f "titan_hcl:$W"); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$REPO" ] && echo "$p" && break; done)
  if [ -z "$PID" ]; then echo "### $W: NOT RUNNING ($REPO)"; continue; fi
  $PY record -f raw -o /tmp/ps.folded --pid "$PID" --duration "$DUR" --rate 50 2>/dev/null
  echo "### $W (pid=$PID)"
  python3 -c "
import collections
s=collections.Counter(); tot=0
try:
  for line in open('/tmp/ps.folded'):
    line=line.rstrip()
    if not line: continue
    try: st,c=line.rsplit(' ',1); c=int(c)
    except: continue
    fr=st.split(';'); tot+=c
    if fr: s[fr[-1]]+=c
except FileNotFoundError:
  print('  (no samples)'); raise SystemExit
if not tot: print('  (idle / no on-CPU samples)'); raise SystemExit
for f,c in s.most_common(6): print(f'  {100*c/tot:5.1f}%  {f[:88]}')
"
done
