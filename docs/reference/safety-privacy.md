# Safety and privacy

> What leaves your box, key custody, X posting safety, and what to do
> when something looks wrong.

> 📝 **Status: outline (W5 scaffold).** Full content lands in v3.x.

---

## What this covers

A clear accounting of: what data leaves your machine, where it goes,
who can see it, and what choices you make at install time that affect
that surface area.

---

## What leaves your box (by mode)

### Mode 3 (local-simulated)
- LLM queries → your chosen inference provider (Ollama = nothing leaves; OpenRouter = your queries reach them)
- Telegram traffic → Telegram's servers (your private bot chat)
- *Nothing else.*

### Modes 1 / 2 (on-chain)
Everything mode 3 sends, plus:
- TimeChain `commit_state` calls → your Solana RPC provider + the chain (public)
- ZK Vault `append_epoch_snapshot` calls → your RPC + the chain (ZK-compressed; the encrypted blob is public but the contents are not derivable)
- Arweave daily snapshots → Irys → Arweave (encrypted at rest with your keys)
- (If you enabled X posting) → twitterapi.io + Webshare → X

## What never leaves your box (any mode)

- Your Maker wallet private key — only on your box, never transmitted
- Your Titan keypair — only on your box, never transmitted
- Your `~/.titan/secrets.toml` — credentials, never transmitted
- Internal dream content, meditation introspection, Mind-internal felt-state — never expressed outward unless Titan chooses to (and that's projection, not transmission)
- Vertex 1 of your Shamir shard — shown to you once, never logged, never transmitted

## Key custody

[ outline:
- Maker wallet: your responsibility; hardware wallet recommended for mode 1
- Titan keypair: lives at ~/.titan/keypair_<titan_id>.json, mode 0o600
- Secrets file: ~/.titan/secrets.toml, mode 0o600
- Vertex 1 (Shamir shard): YOU keep; offline; ideally printed
]

## X-posting safety

[ outline:
- SocialXGateway is the sole sanctioned X path; ad-hoc curl/X calls
  refused by the architecture
- Rate-limit hygiene; backoff
- Maker greenlight required for account-changing actions (handle
  changes, profile edits, follow/unfollow); not in `--default`
- What can Titan say? Driven by the FILTER_DOWN gates + archetype
  selection; you can disable X posting entirely at any time
]

## Surveillance & subpoena surface

[ outline:
- Telegram traffic: visible to Telegram
- LLM queries (OpenRouter/OpenAI/Anthropic): visible to that provider
- On-chain calls: PUBLIC on Solana (anyone can see what your wallet does)
- Arweave: encrypted at rest but the *fact of upload* is public
- The point: on-chain mode trades privacy for sovereignty. Local mode
  gives back the privacy.
]

## What to do if a credential is compromised

[ outline:
1. Rotate the credential at the provider (Telegram BotFather → revoke
   token; OpenRouter dashboard → revoke key; etc.)
2. Update ~/.titan/secrets.toml
3. Restart Titan
4. If a Solana key is compromised: transfer assets, mark address
   burned, file an incident note
]

## What to do if your Vertex 1 shard is exposed

[ outline: a single shard alone is not enough (2-of-3 threshold), but
the security margin is now reduced. Consider rotating the Titan
identity entirely (i.e., starting a new Titan and retiring the old
one) ]

---

→ [Backup and recovery](../operating/backup-recovery.md)
→ [Identity, soul, and SSS](../concepts/identity-soul-sss.md)
→ [Why Titan?](../why-titan.md)
