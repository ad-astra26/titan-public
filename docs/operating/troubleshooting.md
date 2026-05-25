# Troubleshooting

> Common pitfalls and how to fix them. If `setup_titan --diagnostic`
> didn't help, start here.

Symptom → likely cause → fix. The format is deliberately recipe-like
so you can jump to a section by Ctrl-F. If your issue isn't here,
please file an issue (template at the bottom).

---

## Setup-time issues

### Telegram bot doesn't respond

**Symptoms:** You messaged the bot, no reply. The Telegram client shows
the bot as "online" but no `/start` response.

**Likely causes + fixes:**

- **Token wrong or revoked.** Re-check from
  [@BotFather](https://t.me/BotFather): `/mybots` → your bot → API
  Token. Compare with your `~/.titan/secrets.toml`
  `[channels.telegram] bot_token`. If you regenerated, update + restart
  Titan.
- **You haven't `/start`ed the bot from YOUR Telegram client.**
  Telegram bots don't accept commands until at least one user starts
  the chat. Click your bot in Telegram → click "Start" once.
- **Firewall blocks outbound HTTPS.** Telegram bot polling uses
  `https://api.telegram.org`. Verify:
  ```bash
  curl -fsS https://api.telegram.org/bot$YOUR_TOKEN/getMe
  ```
  If this fails, fix your network egress before debugging further.
- **Check journalctl for the Telegram channel worker:**
  ```bash
  journalctl -u titan --since '5 min ago' | grep -i telegram
  ```

### Ollama not auto-detected

**Symptoms:** Wizard says "no local Ollama found" but you know you
have Ollama installed.

**Likely causes + fixes:**

- **Ollama isn't in `$PATH` for the user the wizard runs as.**
  Verify: `which ollama` as the install user.
- **Ollama isn't running.** Start it: `ollama serve` (or `systemctl
  start ollama`). Confirm: `curl -s http://localhost:11434/api/tags`.
- **No models loaded.** `ollama list` should return ≥1 model. Pull
  one: `ollama pull deepseek-v3.1:7b` (or `llama3.3:70b` etc.).
- **Wizard checks `localhost:11434`; Ollama bound to other port or
  interface.** Edit `~/.titan/secrets.toml`
  `[inference.ollama_cloud] api_url` to point to your actual Ollama
  endpoint.

### OpenRouter authentication failing

**Symptoms:** `401 Unauthorized` or `403 Forbidden` in logs.

**Likely causes + fixes:**

- **Wrong key format.** OpenRouter keys start with `sk-or-v1-`. Check
  yours.
- **Account balance = $0.** Add credit at
  [openrouter.ai/credits](https://openrouter.ai/credits).
- **Model isn't available on your tier.** Some models require explicit
  enablement; check your OpenRouter dashboard.
- **Rate limit.** OpenRouter limits free-tier accounts; check headers
  on the failed request.

### Anchor build fails

**Symptoms:** During mode 1/2 setup, the wizard's `anchor build`
step fails.

**Likely causes + fixes:**

- **Rust too old.** Need 1.75+. Update: `rustup update stable`.
- **Solana CLI too old.** Need 1.18+. Update:
  `solana-install update`.
- **Anchor wrong version.** Need 0.30+. Reinstall:
  `cargo install --git https://github.com/coral-xyz/anchor anchor-cli --tag v0.30.1`.
- **Disk full.** Anchor's `target/` dir is large (~3GB for a clean
  build). `df -h` to check.
- **`Cargo.lock` mismatch.** Delete `programs/titan_zk_vault/target`
  and retry: `rm -rf programs/titan_zk_vault/target && anchor build`.

### Genesis ceremony hangs at SOL airdrop (devnet)

**Symptoms:** Wizard stays on "airdropping devnet SOL..." for >30s.

**Likely causes + fixes:**

- **Devnet faucet rate-limited.** Try alternate faucets:
  `https://faucet.solana.com/` or `https://solfaucet.com/`.
- **RPC unresponsive.** Try a different devnet RPC URL (the wizard
  defaults to one).
- **Skip airdrop:** if you already have devnet SOL from another
  source, pass `--skip-airdrop` to setup.

---

## Day-1 issues

### `/health` returns 503 for more than 60 seconds

**Symptoms:** After start, `curl http://localhost:7777/v6/health`
returns 503 (or connection refused) for >60s.

**Likely causes + fixes:**

- **Cold start is normal up to ~60s.** NN brain replay + memory load
  takes time. Wait 90s on slow disks.
- **Disk IOPS too low.** `iotop` (or `iostat -x 1`) — if disk is at
  100% utilization, you're bottlenecked on disk. Upgrade hardware
  or move `data/` to faster disk.
- **A subsystem is crashing on boot.** `journalctl -u titan | tail
  -50` — look for Python tracebacks or Rust panics.

### Health check shows 12/23 OK

**Symptoms:** `setup_titan --diagnostic` reports several subsystems
WARN or FAIL.

**Steps:**

1. **First identify the failing subsystems.** Look at the names — is
   it Trinity (critical), or Persona (less so)?
2. **Cross-reference cascade patterns.** If CGN is failing, expect
   META-CGN + language teacher + persona to follow.
3. **Check logs for each failing subsystem:**
   ```bash
   journalctl -u titan --since '15 min ago' | grep -iE '<subsystem-name>'
   ```
4. **Restart Titan if 5+ failures:**
   ```bash
   sudo systemctl restart titan
   ```
   Wait 90s and re-check.

### Titan is "quiet" — too few responses

**Symptoms:** Titan responds to `/chat` but not as often as you
expect, X posts are infrequent, etc.

**This is by design in the first 24h.** See
[Getting started — step 6](../getting-started.md#step-6--what-to-expect-in-the-first-24-hours).

If you're past 24h and Titan is still quiet:

- **Check meditation cadence.** Long gaps between meditations =
  metabolic dip. Cross-check SOL balance.
- **Check expression rate config.** `[expressive]` in config.toml —
  the rate-limiter might be too strict.
- **Check X posting safety.** X posts go through SocialXGateway with
  rate-limit hygiene. The gateway will pause itself rather than risk
  rate-limit retaliation.

### CPU pegged at 100%

**Symptoms:** `top` shows one or more Titan processes at sustained
100% CPU.

**Likely causes + fixes:**

- **`titan-trinity-rs` at 100%:** unusual. Normally it's <10%. Check
  if the spec-bound parity test is running in CI — that pegs CPU for
  the test duration only. Otherwise: investigate.
- **`titan_hcl.py` (Python main) at 100%:** an inference worker may
  be in a tight loop. Check `py-spy` if installed:
  `py-spy dump --pid $(pgrep -f titan_hcl.py | head -1)`.
- **One Rust BMS daemon at 100%:** sphere-clock pathology. Restart
  that daemon (the supervisor will respawn it):
  `journalctl -u titan-<daemon>-rs --since '5 min ago'`.

### Disk full

**Symptoms:** `df -h` shows the disk holding `data/` at >95%.

**Likely cause:** TimeChain + `consciousness.db` accumulate unboundedly
over time. See the open retention rFP.

**Short-term:**

- **Archive old `consciousness.db` snapshots to Arweave** (they're
  recoverable):
  ```bash
  # Set environment variable to bound the local cache
  echo 'CONSCIOUSNESS_DB_RETENTION_DAYS=30' >> ~/.titan/secrets.toml
  sudo systemctl restart titan
  ```
- **Clean `/tmp/titan-public-*` staging dirs from prior sync runs:**
  ```bash
  rm -rf /tmp/titan-public-incr-* /tmp/titan-public-src-* /tmp/titan-public-sync-*
  ```
- **Clean test_env if present and unused:**
  ```bash
  du -sh test_env/  # see how much it's using
  ```

---

## Recurring issues (open investigations)

### "SPHERE_PULSE not reaching broker"

Trinity detector frozen fleet-wide. Active investigation as of
May 2026. Workaround: restart the trinity daemon set:

```bash
sudo systemctl restart titan-trinity-rs
# or, if not via systemd:
bash scripts/t1_manage.sh restart --force
```

The restart re-bootstraps the broker. Issue tracked on
[titan-public/issues](https://github.com/ad-astra26/titan-public/issues).

### "Trinity detector frozen"

Same as above; symptomatic display only. Same restart procedure.

### LLM inference timeouts

**Symptoms:** Repeated `inference: timeout after Ns` in logs.

**Likely causes + fixes:**

- **Ollama: model not loaded yet.** First request loads the model
  into RAM (~30s for a 7B model on CPU). Subsequent requests are
  fast.
- **OpenRouter: rate-limit or account balance.** Check the response
  envelope in logs.
- **Provider down.** Curl the provider's status endpoint.
- **Bump the inference timeout.** Edit `[inference] timeout_seconds`
  in `titan_hcl/config.toml`.

---

## How to file a useful bug report

When something's not right, please file an issue:

1. **Run diagnostic + capture logs + redact secrets:**
   ```bash
   setup_titan --diagnostic > /tmp/diag.txt 2>&1
   journalctl -u titan --since '10 min ago' > /tmp/journal.txt 2>&1
   # Redact (defense in depth — the diag should auto-redact, this is
   # belt + suspenders):
   sed -E -i 's/(api_key|token|secret) *= *"[^"]+"/\1 = "<REDACTED>"/g' /tmp/diag.txt
   sed -E -i 's/(api_key|token|secret)["'\'']?\s*[:=]\s*["'\''][^"'\'' ]+/\1=<REDACTED>/g' /tmp/journal.txt
   ```
2. **Describe what you did and what you expected.** Specifically:
   - What command did you run?
   - What did you expect to happen?
   - What actually happened?
   - What's the most recent change you made before this started?
3. **Open the issue** at
   [titan-public/issues](https://github.com/ad-astra26/titan-public/issues).
   Use the "Bug" label.

---

→ [Diagnostics](diagnostics.md)
→ [Configuration](configuration.md)
→ [Hardware](../reference/hardware.md)
