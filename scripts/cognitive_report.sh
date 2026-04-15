#!/bin/bash
# cognitive_report.sh — Phase 3 Cognitive Loop Monitoring Report
# Usage: bash scripts/cognitive_report.sh [log_file]
#
# Produces a comprehensive report of Titan's cognitive architecture health:
# mini-reasoners, reasoning chains, interpreter, language, teacher, NN training

LOG="${1:-/tmp/titan_brain.log}"

echo "============================================"
echo "  TITAN COGNITIVE LOOP REPORT"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Log: $LOG"
echo "============================================"

echo ""
echo "=== MINI-REASONERS ==="
grep "Mini-reason" "$LOG" | tail -3
echo ""

echo "=== REASONING CHAINS ==="
COMMITS=$(grep -c "Reasoning.*COMMIT —" "$LOG" 2>/dev/null || echo 0)
ABANDONS=$(grep -c "Reasoning.*ABANDON —" "$LOG" 2>/dev/null || echo 0)
TOTAL=$((COMMITS + ABANDONS))
if [ "$TOTAL" -gt 0 ]; then
  RATE=$((COMMITS * 100 / TOTAL))
  echo "COMMITs: $COMMITS | ABANDONs: $ABANDONS | Rate: ${RATE}%"
else
  echo "No chains completed yet"
fi
grep "Reasoning.*reward=" "$LOG" | grep -oP "reward=\K[0-9.]+" | awk '{sum+=$1; count++} END {if(count>0) printf "Avg reward: %.3f over %d chains\n", sum/count, count}'
echo "Recent chains:"
grep "Reasoning.*COMMIT —\|Reasoning.*ABANDON —" "$LOG" | tail -5
echo ""

echo "=== ASSOCIATE (hierarchical + mini-experience) ==="
EUREKAS=$(grep -c "ASSOCIATE.*eureka=True" "$LOG" 2>/dev/null || echo 0)
echo "Eureka moments: $EUREKAS"
grep "ASSOCIATE.*eureka=True" "$LOG" | tail -3
echo ""

echo "=== INTERPRETER ACTIONS ==="
grep "INTERPRET" "$LOG" | tail -5
echo "Distribution:"
grep "INTERPRET" "$LOG" | grep -oP '→ \K\w+' | sort | uniq -c | sort -rn
echo ""

echo "=== COMPOSITIONS ==="
grep "SELF-COMPOSED" "$LOG" | tail -8
COMP_COUNT=$(grep -c "SELF-COMPOSED" "$LOG" 2>/dev/null || echo 0)
echo "Total: $COMP_COUNT compositions"
echo ""

echo "=== TEACHER SESSIONS ==="
grep "TEACHER.*Sent\|TEACHER.*Session\|TEACHER.*rule learned" "$LOG" | tail -8
echo "Mode distribution:"
grep "TEACHER.*Sent" "$LOG" | grep -oP "Sent \K\w+" | sort | uniq -c | sort -rn
echo ""

echo "=== NN TRAINING ==="
echo "FilterDown:"
grep "FilterDown.*Train step" "$LOG" | tail -1
echo "NeuralNS:"
grep "NeuralNS.*save_all\|NeuralNS.*Saved" "$LOG" | tail -1
echo "Reasoning buffer:"
grep "buffer=" "$LOG" | tail -1
echo "Mini-reasoner dream:"
grep "Mini-reasoner dream" "$LOG" | tail -1
echo ""

echo "=== EPOCH PERFORMANCE ==="
grep "\[PROFILE\] Epoch" "$LOG" | tail -5 | awk -F'[()]' '{print "  " $2}'
echo ""

echo "=== ERRORS (non-Solana) ==="
grep -i "error\|crash\|traceback" "$LOG" | grep -v "RPC\|Network\|SolanaClient\|get_balance\|Payment Required" | tail -5
echo ""
echo "============================================"
