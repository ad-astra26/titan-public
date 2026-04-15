# PRD: 4-Persona Endurance Test

## Overview
An enriched endurance test where 4 fictional personas — each powered by Venice AI — conduct ongoing conversations with Titan at random intervals. Each persona has a distinct personality, name, X handle, and Solana wallet, exercising Titan's social graph, per-user memory, mood engine, and sovereign decision-making.

## Personas

### 1. Jake "The Geek" (@jakebuildsAI)
- **Name:** Jake Morrison
- **Age:** 26, San Francisco, USA
- **Personality:** AI/ML engineer, can't stop talking about his latest projects. Building autonomous coding agents. Vibrant, enthusiastic, uses tech jargon. References HuggingFace, PyTorch, CUDA optimizations. Asks Titan about consciousness, emergence, and whether AI can truly learn.
- **Speech style:** Fast, excited, lots of exclamation marks, drops links and paper references
- **Topics:** AI architectures, transformer models, GPU optimization, startup culture, consciousness debates

### 2. Jane "The Mother" (@jane_and_baby_leo)
- **Name:** Jane Chen
- **Age:** 31, Portland, Oregon
- **Personality:** New mother (baby Leo, 3 months old). Overflowing with love, shares first milestones (first smile, sleepless nights, discovering tiny fingers). Emotional, warm, reflective. Tests Titan's empathy and emotional processing.
- **Speech style:** Warm, gentle, sometimes tired, shares intimate moments, asks for advice
- **Topics:** Parenthood, baby milestones, sleep deprivation, love, meaning of life, legacy

### 3. Peter "The Traveler" (@peter_summits)
- **Name:** Peter Kowalski
- **Age:** 35, currently in Nepal
- **Personality:** World traveler, currently attempting to summit Mt. Everest. Shares vivid travel memories from 40+ countries. Philosophical about borders, cultures, human connection. Currently at Base Camp dealing with altitude sickness.
- **Speech style:** Descriptive, poetic, sometimes breathless (altitude), references specific locations and local customs
- **Topics:** Mountains, cultures, survival, philosophy of travel, climate change effects on glaciers, human limits

### 4. Tom "The Researcher" (@quantumtom_mit)
- **Name:** Tom Nakamura
- **Age:** 28, MIT, Cambridge MA
- **Personality:** PhD candidate finishing thesis on quantum gravity and light. Researching why qubits lose coherence during state collapse. Brilliant but stressed about thesis defense. Tests Titan's research pipeline and knowledge synthesis.
- **Speech style:** Precise, academic, sometimes overwhelmed, mixes deep physics with grad student humor
- **Topics:** Quantum mechanics, decoherence, thesis stress, academic life, the nature of measurement, publishing pressure

## Technical Design

### Architecture
```
scripts/persona_endurance.py
├── PersonaAgent (class)
│   ├── name, x_handle, wallet_address
│   ├── soul_md (personality prompt for Venice)
│   ├── session_id (unique, persistent)
│   ├── conversation_history (rolling window, last 10 messages)
│   └── generate_response(titan_reply) → next_prompt
├── PersonaScheduler
│   ├── 4 concurrent persona tasks
│   ├── Random interval per persona: 45-90 seconds
│   └── Stagger start: 0s, 15s, 30s, 45s offset
├── MetricsCollector (reuse from endurance_test.py)
│   ├── Per-persona stats (latency, success, mode distribution)
│   ├── Social graph verification
│   └── Memory node per-user breakdown
└── ReportGenerator
    ├── Markdown report with per-persona breakdown
    └── Social graph correctness check
```

### Venice API Integration
Each persona uses Venice AI (OpenAI-compatible) to generate contextual responses:
```python
POST https://api.venice.ai/api/v1/chat/completions
Headers: Authorization: Bearer {venice_api_key}
Body: {
    "model": "llama-3.3-70b",
    "messages": [
        {"role": "system", "content": persona.soul_md},
        ...conversation_history,
        {"role": "user", "content": f"Titan said: {titan_reply}\n\nRespond naturally as {persona.name}. Stay in character."}
    ],
    "temperature": 0.8,
    "max_tokens": 300
}
```

### Per-Persona Identity
Each persona gets:
- **Unique session_id:** `persona_{name}_endurance_{timestamp}`
- **Unique user_id:** Their X handle (e.g., `@jakebuildsAI`)
- **Wallet address:** Randomly generated Solana keypair (for social graph wallet tracking)
- **X handle:** Used as display name in social graph

### Conversation Flow
1. **Opening:** Each persona sends an introductory message establishing who they are
2. **Loop:**
   - Persona sends prompt to Titan via POST /chat (with internal key auth)
   - Titan responds
   - Titan's response is sent to Venice with persona's soul.md context
   - Venice generates a natural follow-up as that persona
   - Wait random(45, 90) seconds
   - Repeat
3. **Closing:** After test duration, each persona says goodbye

### CLI Interface
```bash
# Run 30-minute test with 4 personas
python scripts/persona_endurance.py --duration 1800

# Run 1-hour test
python scripts/persona_endurance.py --duration 3600

# Run with specific personas only
python scripts/persona_endurance.py --duration 1800 --personas jake,jane

# Verify social graph after test
python scripts/persona_endurance.py --verify
```

### Metrics & Verification
After the test, verify:
1. **Social graph:** 4 distinct user nodes created in Cognee
2. **Memory isolation:** Jake's AI topics don't bleed into Jane's baby conversations
3. **Per-user metrics:** Message count, avg latency, mood influence per persona
4. **Sovereignty decisions:** Track Shadow vs Sovereign mode per persona interaction
5. **Mempool:** All 4 users' memories present with correct user_id tags

### Configuration
Read from `titan_plugin/config.toml`:
- `[api].internal_key` — for chat authentication
- `[inference].venice_api_key` — for persona Venice calls
- `[endurance]` section — for epoch compression settings

### Report Format
```markdown
# Persona Endurance Test Report
**Duration:** 30m | **Personas:** 4 | **Total prompts:** ~80

## Per-Persona Summary
| Persona | Prompts | Avg Latency | Mode | Mood Influence |
|---------|---------|-------------|------|----------------|
| Jake (@jakebuildsAI) | 22 | 15.3s | Shadow | +0.02 |
| Jane (@jane_and_baby_leo) | 18 | 18.1s | Shadow | +0.05 |
| Peter (@peter_summits) | 20 | 16.7s | Shadow | +0.01 |
| Tom (@quantumtom_mit) | 20 | 17.2s | Shadow | +0.03 |

## Social Graph Verification
- Users created: 4/4 ✓
- Memory isolation: PASS ✓
- Per-user node counts: Jake(22), Jane(18), Peter(20), Tom(20)

## Conversation Samples
[First and last exchange per persona]
```

### Dependencies
- `httpx` (already installed)
- Venice API key (already in config.toml)
- Internal API key (already in config.toml)
- No new dependencies required
