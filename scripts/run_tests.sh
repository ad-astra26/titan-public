#!/usr/bin/env bash
# run_tests.sh — run the test suite (or a subset) under pytest-xdist.
#
# The historical "one test file per process" rule exists because multiple
# TitanHCL / SageRecorder instances in ONE process mmap the same
# LazyMemmapStorage scratch dir (titan_hcl/core/sage/recorder.py:149) → SIGBUS.
# It is a SAME-PROCESS collision, not a per-file law. pytest-xdist workers are
# separate OS processes, and `--dist loadfile` pins every test in a file to ONE
# worker — so the exact isolation of the old serial method is preserved while
# different files run in parallel. Verified 2026-05-31 (workflow audit): -n4 over
# 8 mmap-heavy files (meta_cgn, state_registry, v3_core, system_sensor,
# sage_recorder) produced NO SIGBUS and an identical pass/fail set to serial.
#
# Default -n 2: safe on this 4-core / 7GB box with T1 mainnet live (each torch
# worker ~860MB RSS; -n4 saturated with ~zero speedup + swap risk). Raise only on
# an idle box: `TITAN_TEST_JOBS=4 bash scripts/run_tests.sh`.
#
# Usage:
#   bash scripts/run_tests.sh                      # whole suite
#   bash scripts/run_tests.sh tests/test_backup_*.py   # a subset
set -euo pipefail
source /home/antigravity/projects/titan/test_env/bin/activate
N="${TITAN_TEST_JOBS:-2}"
exec python -m pytest "${@:-tests}" -p no:anchorpy --dist loadfile -n "$N" --tb=short
