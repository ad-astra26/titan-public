#!/bin/bash
# V6 Quick Status — one-command health overview for both Titans
# Usage: bash scripts/v6_status.sh

echo "════════════════════════════════════════════════════════"
echo "  TITAN V6 STATUS — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════════"

# T1
echo ""
echo "── T1 (localhost:7777) ──"
T1=$(curl -s http://localhost:7777/v4/inner-trinity 2>/dev/null)
if [ -z "$T1" ]; then
    echo "  API: UNREACHABLE"
else
    echo "$T1" | python3 -c "
import json,sys
d=json.load(sys.stdin).get('data',{})
r=d.get('reasoning',{})
pi=d.get('pi_heartbeat',{})
dr=d.get('dreaming',{})
ns=d.get('neural_nervous_system',{})
print(f'  Epoch: {d.get(\"tick_count\",\"?\")}  NS: {ns.get(\"total_train_steps\",\"?\")} steps')
print(f'  Reasoning: {r.get(\"total_conclusions\",0)}/{r.get(\"total_chains\",0)} commits ({r.get(\"total_conclusions\",0)*100//max(1,r.get(\"total_chains\",1))}%) avg_len={r.get(\"total_reasoning_steps\",0)/max(1,r.get(\"total_chains\",1)):.1f}')
print(f'  π-rate: {pi.get(\"heartbeat_ratio\",0)*100:.1f}% clusters={pi.get(\"cluster_count\",0)}')
print(f'  Dreaming: cycle={dr.get(\"cycle_count\",0)} is_dreaming={dr.get(\"is_dreaming\",\"?\")} since={dr.get(\"epochs_since_dream\",0)}')
" 2>/dev/null
    # Teacher + interpreter from log
    echo -n "  Teacher modes: "
    for m in modeling grammar creative meaning context; do
        c=$(grep -ac "mode=$m" /tmp/titan_brain.log 2>/dev/null)
        [ "$c" -gt 0 ] && echo -n "$m=$c "
    done
    echo ""
    IC=$(grep -ac "\[INTERPRET\]" /tmp/titan_brain.log 2>/dev/null)
    echo "  Interpreter fires: $IC"
    BT=$(tail -200 /tmp/titan_brain.log 2>/dev/null | grep -c "Request timed out")
    BD=$(tail -200 /tmp/titan_brain.log 2>/dev/null | grep -c "Queue full")
    echo "  Bus: $BT timeouts, $BD drops (last 200 lines)"
fi

# T2
echo ""
echo "── T2 (10.135.0.6:7777) ──"
T2=$(ssh -o ConnectTimeout=5 root@10.135.0.6 "curl -s http://localhost:7777/v4/inner-trinity" 2>/dev/null)
if [ -z "$T2" ]; then
    echo "  API: UNREACHABLE"
else
    echo "$T2" | python3 -c "
import json,sys
d=json.load(sys.stdin).get('data',{})
r=d.get('reasoning',{})
pi=d.get('pi_heartbeat',{})
dr=d.get('dreaming',{})
ns=d.get('neural_nervous_system',{})
print(f'  Epoch: {d.get(\"tick_count\",\"?\")}  NS: {ns.get(\"total_train_steps\",\"?\")} steps')
print(f'  Reasoning: {r.get(\"total_conclusions\",0)}/{r.get(\"total_chains\",0)} commits ({r.get(\"total_conclusions\",0)*100//max(1,r.get(\"total_chains\",1))}%) avg_len={r.get(\"total_reasoning_steps\",0)/max(1,r.get(\"total_chains\",1)):.1f}')
print(f'  π-rate: {pi.get(\"heartbeat_ratio\",0)*100:.1f}% clusters={pi.get(\"cluster_count\",0)}')
print(f'  Dreaming: cycle={dr.get(\"cycle_count\",0)} is_dreaming={dr.get(\"is_dreaming\",\"?\")} since={dr.get(\"epochs_since_dream\",0)}')
" 2>/dev/null
    # Teacher + interpreter from T2 log
    echo -n "  Teacher modes: "
    ssh -o ConnectTimeout=5 root@10.135.0.6 "
        for m in modeling grammar creative meaning context; do
            c=\$(grep -c \"mode=\$m\" /tmp/titan_brain.log 2>/dev/null)
            [ \"\$c\" -gt 0 ] && echo -n \"\$m=\$c \"
        done
    " 2>/dev/null
    echo ""
    IC=$(ssh -o ConnectTimeout=5 root@10.135.0.6 "grep -c '\[INTERPRET\]' /tmp/titan_brain.log 2>/dev/null" 2>/dev/null)
    echo "  Interpreter fires: ${IC:-0}"
fi

echo ""
echo "════════════════════════════════════════════════════════"
