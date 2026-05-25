# Inference providers

> Titan uses an LLM as one of his tools. The LLM is not Titan's "brain" —
> the architecture is the brain. But Titan does need an LLM for the parts
> of cognition that haven't yet been compiled into procedural skill
> (see [Learning and the Synthesis Engine](concepts/learning-and-synthesis.md)).

The first release ships **two** inference providers. We pick the right one
for you at install via this rule:

> **`--default` rule (locked):** auto-detect a local Ollama with a capable
> model loaded; if present, use it (most sovereign, no key). Otherwise
> prompt for an OpenRouter key. *Sovereign when possible, easy when not.*

---

## At a glance

| Provider | Status | Sovereign? | Cost | Use when… |
|----------|--------|------------|------|-----------|
| **Ollama** (local) | ✅ tested over weeks | Yes — runs on your box | Free (your hardware) | …you have a capable box and want maximum sovereignty. |
| **OpenRouter** (API) | ✅ in use | Partial — your key, their compute | Pay-per-token (varies by model) | …you don't have a local-LLM-capable box, or you want OpenRouter's wide model selection. |
| OpenAI API | ⏳ on the roadmap | Partial — same shape as OpenRouter | Pay-per-token | …shipping soon. |
| Anthropic API | ⏳ on the roadmap | Partial — same shape as OpenRouter | Pay-per-token | …shipping soon. (Note: Claude Pro/Max are claude.ai subscriptions, *not* API access — not a programmatic path.) |
| Venice | ⚠ blocked | — | — | …only if a TOS-clean programmatic mode exists. |

---

## Ollama (local — most sovereign)

[Ollama](https://ollama.com) runs an LLM directly on your own machine.
No API key, no per-token cost, no network round-trip, no third-party
record of your conversations.

### Why we recommend it

- **Sovereignty.** Your queries never leave your box. There is no
  provider to subpoena. There is no provider to rate-limit you.
- **Cost.** Zero monetary cost after the hardware is bought. Even a
  large model running 24/7 only costs electricity.
- **Privacy.** Anything Titan says to himself in dream cycles, in
  meditation, in introspection — never reaches an external network.
- **Latency.** Local inference (a few hundred ms on a GPU, 1–2 s on
  a capable CPU) typically beats API round-trips.

### What you need

- **Disk:** model files are 4–80 GB depending on what you run. A
  `deepseek-v3.1` or `llama3.3:70b` is ~40 GB.
- **RAM:** at minimum the model's quantized size. 32 GB is comfortable
  for a 70B-class quantized model.
- **GPU recommended but not required.** On a capable GPU you'll get
  100+ tokens/sec; on CPU you'll get 1–5 tokens/sec, which is usable
  but slow. Apple Silicon (M2/M3/M4) is in a sweet spot — unified
  memory architecture runs surprisingly well.
- **Ollama installed.** `curl -fsSL https://ollama.com/install.sh | sh`
  (or the wizard offers to do this).

### Setup

```bash
# Install Ollama (or let the wizard)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model we've tested with Titan
ollama pull deepseek-v3.1:671b      # if you have the hardware
# or
ollama pull deepseek-v3.1:7b        # for modest boxes

# Verify it serves
ollama list
curl -s http://localhost:11434/api/tags
```

### Switching models

In `titan_hcl/config.toml`:

```toml
[inference]
provider = "ollama"
model = "deepseek-v3.1:7b"
api_url = "http://localhost:11434"
```

Restart Titan; he'll pick up the new model on the next inference.

### Models we've tested

- `deepseek-v3.1:671b` — our T1 production model (Ollama Cloud)
- `deepseek-v3.1:7b` — for modest local boxes
- `llama3.3:70b` — solid for general use
- `qwen2.5-coder:32b` — strong for code-heavy tasks

Other models may work but haven't been stress-tested with the Synthesis
Engine's tool-call patterns.

---

## OpenRouter (API — easiest path)

[OpenRouter](https://openrouter.ai) is a unified API proxy with access
to dozens of LLM providers. We recommend it for users who don't have
local-LLM hardware.

### Why we recommend it (as the fallback)

- **No hardware burden.** Works on a 2-vCPU/4-GB VPS that couldn't run
  a useful local LLM.
- **Wide model selection.** Switch models by changing one config key.
- **Predictable billing.** Pay per token at published rates.
- **No model-rotation drama.** OpenRouter handles upstream provider
  changes; you don't see them.

### Setup

1. Sign up at [openrouter.ai](https://openrouter.ai).
2. Add credit to your account (start with $5; it goes a long way).
3. Create an API key from the dashboard.
4. Paste it into the wizard when prompted.

### Switching models

In `titan_hcl/config.toml`:

```toml
[inference]
provider = "openrouter"
model = "anthropic/claude-opus-4-7"          # or any OpenRouter model slug
api_key = "${OPENROUTER_API_KEY}"             # ← from your env, not committed
api_url = "https://openrouter.ai/api/v1"
```

We strongly recommend storing the actual key in `~/.titan/secrets.toml`,
not in `titan_hcl/config.toml` (which is gitignored but still on disk).
The wizard does this automatically.

### Cost notes

A working Titan with maturing outer memory uses **single-digit LLM
calls per hour** for typical chat. At current OpenRouter rates that's
a few dollars/month per Titan. Early-Titan, before procedural skills
have compiled, can use 10× that — plan for $5–$20/month per Titan
in the first weeks, dropping over time as the
[sovereignty ratio](concepts/learning-and-synthesis.md#the-thesis-in-one-line)
climbs.

### A note about Anthropic models

OpenRouter can route to Anthropic's Claude models. This is **API
access** — distinct from Claude Pro/Max (which are `claude.ai`
consumer subscriptions and **do not** include programmatic API access).
If you want to use a Claude model with Titan today, go through OpenRouter.

---

## On the roadmap

- **OpenAI API direct** — implementation + testing in flight. Same shape
  as OpenRouter (TOS-clean, pay-per-token). Removes one indirection.
- **Anthropic API direct** — same as OpenAI. Also TOS-clean.
- **Custom OpenAI-compatible endpoint** — already partly wired; the
  config exposes `api_url`. If your provider speaks the OpenAI chat
  completion protocol, it should work — but it's not yet in the
  supported / tested matrix.

We add a provider to "supported" only when it has been validated against
Titan's tool-call patterns and the Synthesis Engine's prompt structures.
A provider that works for chat may still fail for our oracle workflows;
that's why each one earns its row by being tested.

---

## On Venice (blocked)

[Venice AI](https://venice.ai) provides uncensored LLM access. We've
used Venice in the past for non-API workloads. The blocker for
production use is that Venice's *API* mode currently violates their TOS
when used programmatically by an autonomous agent. We will revisit if
that constraint changes. We do **not** recommend or support running
Titan against Venice today.

---

## How Titan uses the LLM

Worth understanding so you can read the cost curve:

- **System prompt** — small, infrequently changing
- **Tool descriptions** — small, cached via Anthropic's prompt-caching
  if you use Claude via OpenRouter
- **Working-memory buffers** — moderate, structured (ACT-R-style)
- **The query itself** — short
- **Tool call results** — Titan reads them back; this is where token
  count can spike

The Synthesis Engine's **standing contracts** (shipped 2026-05-25)
materialize common retrievals so the LLM doesn't have to chain tool
calls for the same recurring question. This is the biggest single
lever for cost reduction over the lifetime of a Titan.

---

→ [Getting started](getting-started.md) — full install walk-through
→ [Learning and the Synthesis Engine](concepts/learning-and-synthesis.md) — why
  LLM usage decreases over time
→ [Configuration](operating/configuration.md) — full config walk-through
