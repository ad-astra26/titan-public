# Troubleshooting

> Common pitfalls and how to fix them. If `setup_titan --diagnostic`
> didn't help, start here.

> 📝 **Status: outline (W5 scaffold).** Catalog grows as we collect
> alpha-tester feedback. PRs welcome.

---

## What this covers

Symptom → cause → fix for the issues people hit most often during
setup and the first few days of running a Titan. The format is
deliberately recipe-like.

---

## Setup-time issues

### Telegram bot doesn't respond

[ outline: bot token format check; bot wasn't `/start`ed in your
client; firewall blocks outbound HTTPS; check journalctl ]

### Ollama not auto-detected

[ outline: is `ollama` in `$PATH`? is `ollama list` returning models?
is `:11434` responding? sometimes it's the systemd vs user-mode mismatch ]

### OpenRouter authentication failing

[ outline: key format; account balance > 0; rate-limit; URL prefix
mismatch ]

### Anchor build fails

[ outline: Rust version (need 1.75+); Solana CLI version (need 1.18+);
disk space; Anchor cache invalidation ]

### Genesis ceremony hangs at SOL airdrop (devnet)

[ outline: devnet faucet rate-limited; use a different faucet URL;
wait + retry ]

## Day-1 issues

### `/health` returns 503 for more than 60 seconds

[ outline: cold-load is normal up to ~60s; longer = check logs; if
NN brain replay is the culprit, see disk-IOPS ]

### Health check shows fewer than 20/23 OK

[ outline: which subsystems are red? cross-link to per-subsystem
recovery sections ]

### Titan is "quiet" — too few responses

[ outline: this is by design in the first 24h. See
[Getting started — step 6](../getting-started.md#step-6--what-to-expect-in-the-first-24-hours).
If you're past 24h and Titan is still quiet, check meditation cadence ]

### CPU pegged at 100%

[ outline: which worker? Inference workers occasionally do that;
Trinity tensor worker should not; check with `top -p $(pgrep titan_hcl)` ]

### Disk full

[ outline: Titan grows. Daily `consciousness.db` + `observatory.db`
accumulation. See the open observation gate on retention/VACUUM ]

## Recurring issues

### "SPHERE_PULSE not reaching broker"

[ outline: known active investigation as of May 2026. Cross-link to
the GitHub issue. ]

### "Trinity detector frozen"

[ outline: cross-link to GitHub issue; restart procedure ]

### LLM inference timeouts

[ outline: Ollama: model not loaded yet; OpenRouter: rate-limit or
account balance; check `journalctl -u titan` for the error envelope ]

## How to file a useful bug report

1. Run `setup_titan --diagnostic > diag.txt` and attach (redact your
   token + wallet addr if you want)
2. `journalctl -u titan --since '10 min ago' > journal.txt` and attach
3. Describe what you did and what you expected
4. Open on [titan-public/issues](https://github.com/ad-astra26/titan-public/issues)

---

→ [Diagnostics](diagnostics.md)
→ [Configuration](configuration.md)
→ [Hardware](../reference/hardware.md)
