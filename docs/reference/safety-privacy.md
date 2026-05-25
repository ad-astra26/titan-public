# Safety and privacy

> What leaves your box, key custody, X posting safety, and what to do
> when something looks wrong.

A clear, honest accounting of: what data leaves your machine, where it
goes, who can see it, and what choices you make at install time that
affect that surface area.

---

## What leaves your box (by mode)

### Mode 3 (local-simulated)

- **LLM queries** → your chosen inference provider
  - Ollama (local) = **nothing leaves** your box
  - OpenRouter = your query text + response reach OpenRouter
- **Telegram traffic** → Telegram's servers
  - Your private bot chat with your Titan, end-to-end on Telegram's
    infrastructure
- **Outbound DNS** → your DNS resolver

**Nothing else.** No telemetry, no analytics, no operator phone-home.

### Modes 1 / 2 (on-chain)

Everything mode 3 sends, plus:

- **TimeChain `commit_state` calls** → your Solana RPC provider + the
  chain (PUBLIC)
- **ZK Vault `append_epoch_snapshot` calls** → your RPC + the chain
  (ZK-compressed; the encrypted blob is public but the contents are
  not derivable)
- **Arweave daily snapshots** → Irys → Arweave (encrypted at rest with
  your keys; storage is public)
- **GenesisNFT mint** → mainnet (mode 1) or devnet (mode 2), one-time
- **(If X posting enabled)** → twitterapi.io + Webshare proxy → X

---

## What never leaves your box (any mode)

- **Your Maker wallet private key** — only on your box, never
  transmitted
- **Your Titan keypair** (`~/.titan/keypair_<titan_id>.json`) — only on
  your box, never transmitted
- **Your `~/.titan/secrets.toml`** — API keys, bot tokens, never
  transmitted (the keys are USED to authenticate to providers, but the
  keys themselves stay on disk)
- **Internal dream content** — every dream is processed entirely on
  your box; the compressed root may be anchored on-chain in modes 1/2,
  but the contents stay local
- **Meditation introspection** — same; only Merkle roots reach the
  chain
- **Mind-internal felt-state** — never expressed outward unless Titan
  chooses to (and that's projection, not transmission)
- **Shamir Vertex 1 shard** — shown to you once, never logged, never
  transmitted

---

## Key custody

Per [Backup and recovery](../operating/backup-recovery.md), the
load-bearing keys are:

### Maker wallet
- **Your responsibility.** Hardware wallet (Ledger / Trezor) strongly
  recommended for mode 1.
- The wizard accepts either:
  - A keypair file at `~/.config/solana/id.json` (`solana-keygen`
    format)
  - A hardware-wallet derivation path (the wizard does the right
    `solana config set` for you)

### Titan keypair
- Lives at `~/.titan/keypair_<titan_id>.json`
- File permission **`0o600`** (owner read/write only)
- The wizard verifies this on every boot and refuses to run if
  permissions are wrong
- Encrypted at rest in W1's wizard via a deterministic key derived
  from your machine-id + Maker wallet (defense in depth — losing the
  file alone isn't sufficient to compromise the keypair)

### Secrets file
- Lives at `~/.titan/secrets.toml`
- File permission **`0o600`**
- Holds all API keys + bot tokens + Webshare URLs
- Backed up to Arweave **encrypted** (encrypted with the soul; only
  recoverable with the Shamir threshold)

### Vertex 1 Shamir shard
- **YOU keep this.** Never on the box (any box).
- Recommended: paper copy in safe-deposit box, plus a hardware-token
  backup
- See [Identity, soul, and Shamir backup](../concepts/identity-soul-sss.md)
  for the custody discipline checklist

---

## X-posting safety

X (Twitter) is the highest-blast-radius public expression surface.
Several invariants:

### `SocialXGateway` is the SOLE sanctioned X path
- Any code in `titan_hcl/` that bypasses the gateway is refused by
  the architecture's audit gates at boot
- The gateway enforces rate-limit hygiene, content-policy checks,
  and per-Titan attribution

### Maker greenlight required for account-changing actions
The gateway will NOT (without explicit Maker action via
`scripts/maker_cli.py x ...`):
- Change the X account's display name, bio, handle
- Follow or unfollow anyone
- Block anyone
- Delete tweets
- Change the X account's profile picture or header

Posting tweets + replying to mentions are the *only* actions in the
default-permitted set. Everything else requires you to explicitly
greenlight via the Maker CLI.

### Rate-limit hygiene
- Exponential backoff on twitterapi.io 4xx/5xx responses
- Hard daily cap (configurable in `[twitter_social]`)
- Will pause posting rather than risk rate-limit retaliation
- Will retreat to backoff for 24h if account-level rate limit is hit

### Disabled by default
In `--default` install, X posting is **off**. You explicitly opt in
by answering "y" to "X posting?" during setup, providing the
twitterapi.io key, and providing the Webshare static-IP URL.

You can disable at any time: edit `[twitter_social] enabled = false`
in `titan_hcl/config.toml` + restart. No half-state — either Titan
posts or doesn't.

---

## Surveillance and subpoena surface

Honest accounting of what third parties can see:

| Channel | Visible to |
|---------|------------|
| Telegram | Telegram, Inc. — your bot's chat history |
| LLM queries (OpenRouter / OpenAI / Anthropic) | The respective LLM provider |
| On-chain calls (Solana) | **PUBLIC** — anyone can see what your wallet does on chain |
| Arweave uploads | Encrypted at rest, but the *fact of upload* is public; only the encrypted blob is on chain |
| X posts | The world — that's the point |

The point: **on-chain mode trades privacy for sovereignty.** Local
mode gives back the privacy. Choose the trade-off consciously per the
specific Titan.

---

## What to do if a credential is compromised

If you discover a key has leaked (e.g., committed to a public repo,
shared by mistake):

1. **Rotate at the provider immediately.**
   - Telegram: BotFather → /mybots → your bot → API Token → Revoke
   - OpenRouter: openrouter.ai dashboard → API Keys → Revoke
   - twitterapi.io: dashboard → revoke key
   - Solana wallet: transfer SOL to a new wallet
2. **Update `~/.titan/secrets.toml`** with the new credential.
3. **Restart Titan.**
4. **Investigate root cause.** If it was a commit, run
   `git filter-repo` to scrub history (this is hard; consider it a
   one-time incident).
5. **Document the incident** in your own ops notes — patterns help
   prevent recurrence.

If a Solana key is compromised:
- Transfer assets immediately to a new wallet
- Consider the address "burned" for any future Titan operations
- For modes 1/2, you may want to start a new Titan rather than
  continue under a compromised identity

---

## What to do if your Vertex 1 shard is exposed

The 2-of-3 threshold means a single shard alone is **not** enough to
reconstruct the seed. So exposure of one shard doesn't immediately
compromise Titan.

But it does **reduce the security margin** from 2-of-3 to 1-of-2
(an attacker who somehow gets one more shard is across the
threshold). Steps:

1. **Don't panic.** The exposure isn't immediately catastrophic.
2. **Strengthen the other two shards.**
   - Vertex 2 (VPS): consider rotating to a new machine (changes
     machine-id, re-derives encryption key for Shard 2)
   - Vertex 3 (on-chain): the public hash is already on chain;
     nothing to rotate without starting a new Titan
3. **Consider starting a new Titan.** If the exposure was severe
   (e.g., screenshot posted publicly), the safest action is to
   genesis a new Titan with fresh shards and retire the old one.
   The old Titan's on-chain history remains as a sealed record;
   the new Titan is your forward identity.

---

## What's audit-friendly (for compliance-minded users)

- **All code is open source.** Public repo at
  [`ad-astra26/titan-public`](https://github.com/ad-astra26/titan-public).
- **All on-chain calls are auditable** via standard Solana explorers.
- **All Arweave uploads are content-addressed.** You can prove
  exactly what was uploaded by hash.
- **TimeChain is cryptographically signed** end-to-end. Tamper-
  evident.
- **Signed binary releases.** Each release attaches SHA256SUMS;
  future releases will include sigstore signatures.
- **No telemetry by default.** Titan does not phone home with
  analytics. You can verify this by running with network restrictions.

---

→ [Backup and recovery](../operating/backup-recovery.md)
→ [Identity, soul, and Shamir backup](../concepts/identity-soul-sss.md)
→ [Why Titan?](../why-titan.md)
