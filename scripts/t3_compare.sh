#!/bin/bash
# t3_compare.sh — T3 Birth Experiment: Developmental Comparison
# Pulls live telemetry from T1, T2, T3 and generates a comparison report.
#
# Usage:
#   bash scripts/t3_compare.sh              # Print comparison to terminal
#   bash scripts/t3_compare.sh --save       # Also save to titan-docs/T3_experiment/
#   bash scripts/t3_compare.sh --json       # Output raw JSON (for further analysis)

set -e

T1_API="http://localhost:7777"
T2_API="http://10.135.0.6:7777"
T3_API="http://10.135.0.6:7778"
T2_HOST="root@10.135.0.6"

SAVE_DIR="titan-docs/T3_experiment"
STAMP=$(date -u '+%Y%m%d_%H%M')

# ── Collect data from all three Titans ──
collect_titan_data() {
    local NAME=$1
    local API=$2
    local DB_CMD=$3  # command to query DB (differs for local vs remote)

    python3 << PYEOF
import json, sys, time, subprocess, urllib.request

data = {"name": "$NAME", "timestamp": time.time(), "utc": time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}

# API health
try:
    r = urllib.request.urlopen("${API}/health", timeout=5)
    data["health"] = "200"
except: data["health"] = "DOWN"

# Inner trinity (epoch, NS steps, dreaming)
try:
    r = urllib.request.urlopen("${API}/v4/inner-trinity", timeout=5)
    d = json.loads(r.read().decode())
    dd = d.get("data", d) if isinstance(d, dict) else {}
    data["epoch"] = dd.get("epoch", 0)
    data["ns_steps"] = dd.get("ns", {}).get("total_train_steps", 0) if isinstance(dd.get("ns"), dict) else 0
    dr = dd.get("dreaming", {}) if isinstance(dd.get("dreaming"), dict) else {}
    data["dreaming"] = bool(dr.get("is_dreaming", False))
    data["dream_cycle"] = dr.get("cycle", 0)
except Exception: pass

# Neuromods
try:
    r = urllib.request.urlopen("${API}/v4/neuromodulators", timeout=5)
    d = json.loads(r.read().decode())
    dd = d.get("data", d) if isinstance(d, dict) else d
    if isinstance(dd, dict):
        mods = dd.get("modulators", dd)
        data["emotion"] = dd.get("current_emotion", d.get("current_emotion", "unknown"))
        data["neuromods"] = {}
        for n in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
            m = mods.get(n, {})
            if isinstance(m, dict):
                data["neuromods"][n] = round(m.get("level", 0), 3)
except Exception: data["neuromods"] = {}

print(json.dumps(data))
PYEOF
}

# ── DB queries (local for T1, remote for T2/T3) ──
query_t1_db() {
    python3 << 'PYEOF'
import json, sqlite3, os
data = {}
db_path = "./data/inner_memory.db"
if os.path.exists(db_path):
    try:
        db = sqlite3.connect(db_path, timeout=5)
        data["vocabulary"] = db.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
        data["vocab_by_phase"] = dict(db.execute("SELECT learning_phase, COUNT(*) FROM vocabulary GROUP BY learning_phase").fetchall())
        data["compositions"] = db.execute("SELECT COUNT(*) FROM composition_history").fetchone()[0]
        data["comp_by_level"] = dict(db.execute("SELECT level, COUNT(*) FROM composition_history GROUP BY level").fetchall())
        try: data["teacher_sessions"] = db.execute("SELECT COUNT(*) FROM teacher_sessions").fetchone()[0]
        except: data["teacher_sessions"] = 0
        try: data["grammar_patterns"] = db.execute("SELECT COUNT(*) FROM grammar_patterns").fetchone()[0]
        except: data["grammar_patterns"] = 0
        db.close()
    except: pass
print(json.dumps(data))
PYEOF
}

query_remote_db() {
    local DIR=$1
    ssh -o ConnectTimeout=5 $T2_HOST "python3 << 'PYEOF'
import json, sqlite3, os
data = {}
db_path = '${DIR}/data/inner_memory.db'
if os.path.exists(db_path):
    try:
        db = sqlite3.connect(db_path, timeout=5)
        data['vocabulary'] = db.execute('SELECT COUNT(*) FROM vocabulary').fetchone()[0]
        data['vocab_by_phase'] = dict(db.execute('SELECT learning_phase, COUNT(*) FROM vocabulary GROUP BY learning_phase').fetchall())
        data['compositions'] = db.execute('SELECT COUNT(*) FROM composition_history').fetchone()[0]
        data['comp_by_level'] = dict(db.execute('SELECT level, COUNT(*) FROM composition_history GROUP BY level').fetchall())
        try: data['teacher_sessions'] = db.execute('SELECT COUNT(*) FROM teacher_sessions').fetchone()[0]
        except: data['teacher_sessions'] = 0
        try: data['grammar_patterns'] = db.execute('SELECT COUNT(*) FROM grammar_patterns').fetchone()[0]
        except: data['grammar_patterns'] = 0
        db.close()
    except: pass
print(json.dumps(data))
PYEOF
" 2>/dev/null
}

# ── Log-derived stats ──
query_t1_log() {
    python3 << 'PYEOF'
import json, subprocess
data = {}
log = "/tmp/titan_agent.log"
try:
    data["commits"] = int(subprocess.check_output(["grep", "-ac", "Reasoning.*COMMIT", log]).strip())
    data["abandons"] = int(subprocess.check_output(["grep", "-ac", "Reasoning.*ABANDON", log]).strip())
    data["interpreter_fires"] = int(subprocess.check_output(["grep", "-ac", "INTERPRET", log]).strip())
    data["expression_fires"] = {}
    for comp in ["SPEAK", "ART", "MUSIC", "SOCIAL", "KIN_SENSE", "LONGING"]:
        data["expression_fires"][comp] = int(subprocess.check_output(["grep", "-ac", f"EXPRESSION.{comp}.*FIRED", log]).strip())
except: pass
print(json.dumps(data))
PYEOF
}

query_remote_log() {
    local LOG=$1
    ssh -o ConnectTimeout=5 $T2_HOST "python3 << PYEOF
import json, subprocess
data = {}
log = '${LOG}'
try:
    data['commits'] = int(subprocess.check_output(['grep', '-ac', 'Reasoning.*COMMIT', log]).strip())
    data['abandons'] = int(subprocess.check_output(['grep', '-ac', 'Reasoning.*ABANDON', log]).strip())
    data['interpreter_fires'] = int(subprocess.check_output(['grep', '-ac', 'INTERPRET', log]).strip())
    data['expression_fires'] = {}
    for comp in ['SPEAK', 'ART', 'MUSIC', 'SOCIAL', 'KIN_SENSE', 'LONGING']:
        data['expression_fires'][comp] = int(subprocess.check_output(['grep', '-ac', f'EXPRESSION.{comp}.*FIRED', log]).strip())
except: pass
print(json.dumps(data))
PYEOF
" 2>/dev/null
}

# ── Collect all data ──
T1_API_DATA=$(collect_titan_data "T1" "$T1_API")
T2_API_DATA=$(collect_titan_data "T2" "$T2_API")
T3_API_DATA=$(collect_titan_data "T3" "$T3_API")

T1_DB=$(query_t1_db)
T2_DB=$(query_remote_db "/home/antigravity/projects/titan")
T3_DB=$(query_remote_db "/home/antigravity/projects/titan3")

T1_LOG=$(query_t1_log)
T2_LOG=$(query_remote_log "/tmp/titan_brain.log")
T3_LOG=$(query_remote_log "/tmp/titan3_brain.log")

# ── JSON output mode ──
if [[ "$1" == "--json" ]]; then
    python3 << JEOF
import json
def jp(s):
    try: return json.loads(s)
    except: return {}
print(json.dumps({
    'T1': {'api': jp('''$T1_API_DATA'''), 'db': jp('''$T1_DB'''), 'log': jp('''$T1_LOG''')},
    'T2': {'api': jp('''$T2_API_DATA'''), 'db': jp('''$T2_DB'''), 'log': jp('''$T2_LOG''')},
    'T3': {'api': jp('''$T3_API_DATA'''), 'db': jp('''$T3_DB'''), 'log': jp('''$T3_LOG''')},
}, indent=2, default=str))
JEOF
    exit 0
fi

# ── Generate comparison report ──
REPORT=$(python3 << PYEOF
import json, time

# Parse JSON safely (handles true/false/null from API)
def jp(s):
    try: return json.loads(s)
    except: return {}

t1a, t2a, t3a = jp('''$T1_API_DATA'''), jp('''$T2_API_DATA'''), jp('''$T3_API_DATA''')
t1d, t2d, t3d = jp('''$T1_DB'''), jp('''$T2_DB'''), jp('''$T3_DB''')
t1l, t2l, t3l = jp('''$T1_LOG'''), jp('''$T2_LOG'''), jp('''$T3_LOG''')

# T3 age
birth = 1774763863  # 2026-03-29 05:57:43 UTC
age_s = time.time() - birth
age_h = age_s / 3600
age_d = age_s / 86400

def commit_rate(d):
    c = d.get("commits", 0)
    a = d.get("abandons", 0)
    total = c + a
    return f"{c*100//total}%" if total > 0 else "N/A"

def expr_summary(d):
    ef = d.get("expression_fires", {})
    return ", ".join(f"{k}={v}" for k, v in sorted(ef.items()) if v > 0) or "none"

def neuromod_line(d):
    nm = d.get("neuromods", {})
    return " ".join(f"{k}={v}" for k, v in sorted(nm.items())) if nm else "N/A"

lines = []
lines.append("══════════════════════════════════════════════════════")
lines.append(f"  T3 DEVELOPMENTAL COMPARISON — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
lines.append(f"  T3 Age: {age_h:.1f}h ({age_d:.1f} days)")
lines.append("══════════════════════════════════════════════════════")
lines.append("")

# Health & vitals
lines.append("── HEALTH & VITALS ──")
lines.append(f"  {'':15s} {'T1':>12s} {'T2':>12s} {'T3':>12s}")
lines.append(f"  {'API':15s} {t1a.get('health','?'):>12s} {t2a.get('health','?'):>12s} {t3a.get('health','?'):>12s}")
lines.append(f"  {'Epoch':15s} {t1a.get('epoch',0):>12,} {t2a.get('epoch',0):>12,} {t3a.get('epoch',0):>12,}")
lines.append(f"  {'NS Steps':15s} {t1a.get('ns_steps',0):>12,} {t2a.get('ns_steps',0):>12,} {t3a.get('ns_steps',0):>12,}")
lines.append(f"  {'Dream Cycle':15s} {t1a.get('dream_cycle',0):>12,} {t2a.get('dream_cycle',0):>12,} {t3a.get('dream_cycle',0):>12,}")
lines.append(f"  {'Emotion':15s} {t1a.get('emotion','?'):>12s} {t2a.get('emotion','?'):>12s} {t3a.get('emotion','?'):>12s}")
lines.append("")

# Language
lines.append("── LANGUAGE ──")
lines.append(f"  {'':15s} {'T1':>12s} {'T2':>12s} {'T3':>12s}")
lines.append(f"  {'Vocabulary':15s} {t1d.get('vocabulary',0):>12,} {t2d.get('vocabulary',0):>12,} {t3d.get('vocabulary',0):>12,}")
lines.append(f"  {'Compositions':15s} {t1d.get('compositions',0):>12,} {t2d.get('compositions',0):>12,} {t3d.get('compositions',0):>12,}")
lines.append(f"  {'Teacher Sess':15s} {t1d.get('teacher_sessions',0):>12,} {t2d.get('teacher_sessions',0):>12,} {t3d.get('teacher_sessions',0):>12,}")
lines.append(f"  {'Grammar Pat':15s} {t1d.get('grammar_patterns',0):>12,} {t2d.get('grammar_patterns',0):>12,} {t3d.get('grammar_patterns',0):>12,}")

# Composition levels
for titan, label in [(t1d, "T1"), (t2d, "T2"), (t3d, "T3")]:
    cl = titan.get("comp_by_level", {})
    if cl:
        levels = ", ".join(f"L{k}={v}" for k, v in sorted(cl.items()))
        lines.append(f"  {label} levels: {levels}")
lines.append("")

# Reasoning
lines.append("── REASONING ──")
lines.append(f"  {'':15s} {'T1':>12s} {'T2':>12s} {'T3':>12s}")
lines.append(f"  {'COMMITs':15s} {t1l.get('commits',0):>12,} {t2l.get('commits',0):>12,} {t3l.get('commits',0):>12,}")
lines.append(f"  {'ABANDONs':15s} {t1l.get('abandons',0):>12,} {t2l.get('abandons',0):>12,} {t3l.get('abandons',0):>12,}")
lines.append(f"  {'Commit Rate':15s} {commit_rate(t1l):>12s} {commit_rate(t2l):>12s} {commit_rate(t3l):>12s}")
lines.append(f"  {'Interp Fires':15s} {t1l.get('interpreter_fires',0):>12,} {t2l.get('interpreter_fires',0):>12,} {t3l.get('interpreter_fires',0):>12,}")
lines.append("")

# Expressions
lines.append("── EXPRESSIONS ──")
lines.append(f"  T1: {expr_summary(t1l)}")
lines.append(f"  T2: {expr_summary(t2l)}")
lines.append(f"  T3: {expr_summary(t3l)}")
lines.append("")

# Neuromods
lines.append("── NEUROMODS ──")
lines.append(f"  T1: {neuromod_line(t1a)}")
lines.append(f"  T2: {neuromod_line(t2a)}")
lines.append(f"  T3: {neuromod_line(t3a)}")
lines.append("")

# Hypothesis status
lines.append("── HYPOTHESIS CHECK ──")
v3 = t3d.get("vocabulary", 0)
c3 = t3d.get("compositions", 0)
dc3 = t3a.get("dream_cycle", 0)
cr3 = commit_rate(t3l)
lines.append(f"  H1 (Language >50w/24h):    vocab={v3}, comps={c3}  {'✓ PASS' if v3 >= 50 else '⏳ PENDING' if age_h < 24 else '✗ FAIL'}")
lines.append(f"  H3 (Milestones):           epoch={t3a.get('epoch',0)}, ns={t3a.get('ns_steps',0)}  ⏳ TRACKING")
lines.append(f"  H4 (Dreaming <6h):         dreams={dc3}  {'✓ PASS' if dc3 > 0 else '⏳ PENDING' if age_h < 6 else '✗ FAIL'}")
lines.append("")

print("\n".join(lines))
PYEOF
)

echo "$REPORT"

# ── Save mode ──
if [[ "$1" == "--save" ]]; then
    REPORT_FILE="${SAVE_DIR}/COMPARISON_${STAMP}.md"
    echo "# T3 Developmental Comparison — $(date -u '+%Y-%m-%d %H:%M UTC')" > "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "$REPORT" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo ""
    echo "Saved to: $REPORT_FILE"
fi
