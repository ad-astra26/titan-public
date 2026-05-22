# Retired Scripts

Scripts here have been **superseded** by Phase C-aware replacements and are kept
**for history only** — do not invoke them. Anything that referenced them in
docs / cron / runbooks should now use the new entry points listed below.

If you find a script invoked from here, file a bug — it means a caller wasn't
updated when the script moved.

## Inventory + Replacements

| Retired script | Date retired | Replaced by | Reason |
|---|---|---|---|
| `safe_restart.sh` | 2026-05-14 | `bash scripts/t1_manage.sh restart [--force]` | Dreaming-aware restart logic folded into the unified manage interface |
| `deploy_t2.sh` | 2026-05-14 | `bash scripts/t2_manage.sh deploy [--include-rust-binaries]` | Deploy is now a `deploy` subcommand of the unified manage script |
| `deploy_t3.sh` | 2026-05-14 | `bash scripts/t3_manage.sh deploy [--include-rust-binaries]` | Same |
| `t2` | 2026-05-14 | `bash scripts/t2_manage.sh <cmd>` | SSH wrapping is now transparent inside t2_manage.sh (no separate wrapper needed) |
| `t2_manage.sh.old` | 2026-05-14 | `bash scripts/t2_manage.sh` (new) | Restructured to use shared `lib/titan_common.sh` + `deploy` subcommand |
| `t3_manage.sh.old` | 2026-05-14 | `bash scripts/t3_manage.sh` (new) | Was Phase A/B legacy (`nohup python titan_main.py`); rewritten to systemd-aware |
| `titan_watchdog.sh` | 2026-05-14 | systemd `Restart=on-failure` + HCL Guardian | Phase C: kernel-rs is systemd-supervised; Guardian supervises L2/L3. External cron watchdog caused restart loops (see `feedback_phase_c_external_watchdog_retired.md`) |
| `services_watchdog.sh` | 2026-05-14 | Phase C SPEC-compliant supervision | Same as above |

## The new unified interface

```bash
# T1 (mainnet, local — runs on this VPS)
bash scripts/t1_manage.sh {status|health|start|stop|restart|logs|pid|help}

# T2 (devnet, remote 10.135.0.6)
bash scripts/t2_manage.sh {status|health|start|stop|restart|logs|pid|deploy|help}
bash scripts/t2_manage.sh deploy --include-rust-binaries

# T3 (devnet, remote 10.135.0.6, separate repo titan3/)
bash scripts/t3_manage.sh {status|health|start|stop|restart|logs|pid|deploy|help}
bash scripts/t3_manage.sh deploy --include-rust-binaries
```

All three scripts share the same command surface via `scripts/lib/titan_common.sh`
— one bug fix, one help text, fleet-wide consistency.

## Exit codes (uniform across all three)

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | systemctl / SSH command failed |
| 2 | post-restart health check failed (Titan didn't come up) |
| 3 | dreaming-wait timed out without natural wake (pass `--force` to override) |
| 4 | bad CLI arguments |
