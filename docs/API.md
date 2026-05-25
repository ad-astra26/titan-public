# Titan Sovereign Observatory API

The Sovereign Observatory is a FastAPI-powered REST + WebSocket interface that exposes the Titan's cognitive state to the outside world. It runs as a non-blocking background task alongside the main OpenClaw plugin loop.

**Default endpoint:** `http://localhost:7777`

## Architecture

```
                    +---------------------------+
                    |    Sovereign Observatory   |
                    +---------------------------+
                    |                           |
          +---------+         +---------+       +----------+
          | Public   |         | Maker   |       | Webhook  |
          | Dashboard|         | Console |       | Receiver |
          | (no auth)|         | (Ed25519)|      | (Helius) |
          +----+-----+         +----+----+       +----+-----+
               |                    |                  |
               +--------+-----------+------------------+
                        |
                   +----v----+
                   | EventBus |-----> WebSocket /ws
                   +---------+
```

- **Public Dashboard** — Read-only endpoints, no authentication required
- **Maker Console** — Write endpoints secured by Ed25519 signature verification
- **Webhook Receiver** — Helius on-chain transaction listener with Maker verification
- **WebSocket** — Real-time event stream via asyncio.Queue pub/sub

## Authentication

Maker Console endpoints require Ed25519 signature authentication using the same Solana keypair configured as `maker_pubkey` in `config.toml`.

### Required Headers

| Header | Description |
|--------|-------------|
| `X-Titan-Signature` | Base58-encoded Ed25519 signature of `"{timestamp}:{body}"` |
| `X-Titan-Timestamp` | Unix timestamp (float) when the request was signed |

### Anti-Replay Protection

Signed requests expire after **30 seconds**. The server rejects any request where `abs(server_time - request_time) > 30`.

### Using the Maker CLI

```bash
# Submit a directive
python scripts/maker_cli.py -k ./authority.json directive "New prime directive"

# Inject a memory
python scripts/maker_cli.py inject-memory "The project deadline moved to March 15th"

# Inject with custom weight (1.0-10.0)
python scripts/maker_cli.py inject-memory "Critical: new wallet is ABC..." --weight 8.0

# Trigger Divine Inspiration
python scripts/maker_cli.py divine-inspiration

# Fetch audit
python scripts/maker_cli.py audit

# Check public status (no keypair needed)
python scripts/maker_cli.py status
```

---

## Public Dashboard Endpoints

### GET /status
**Bio-State Overview** — Real-time snapshot of the Titan's vital signs.

Response:
```json
{
  "status": "ok",
  "data": {
    "mood": {"label": "Reflective", "score": 0.6234},
    "energy_state": {"level": "HIGH", "sol_balance": 1.234},
    "sol_balance": 1.234567,
    "persistent_nodes": 42,
    "mempool_size": 3,
    "soul_gen": 2,
    "is_meditating": false,
    "ws_subscribers": 1
  }
}
```

### GET /status/mood
**Detailed Mood Breakdown** — Internal mood score, delta, and Social Gravity metrics.

Response:
```json
{
  "status": "ok",
  "data": {
    "mood_label": "Reflective",
    "current_score": 0.6234,
    "prior_score": 0.5891,
    "mood_delta": 0.0343,
    "social_gravity": {
      "mentions_received": 12,
      "daily_replies": 5,
      "reply_likes": 3,
      "daily_likes": 8
    }
  }
}
```

### GET /status/memory
**Memory Topology** — Persistent node counts and top memories (redacted to 40 chars).

Response:
```json
{
  "status": "ok",
  "data": {
    "persistent_count": 42,
    "mempool_size": 3,
    "top_memories": [
      {
        "hint": "What is the optimal Solana validator co...",
        "weight": 3.245,
        "intensity": 7,
        "reinforcements": 4
      }
    ],
    "cognee_ready": true
  }
}
```

### GET /status/memory/topology
**Cognitive Heatmap** — Keyword-clustered distribution of what the Titan is focused on.

Clusters: `Solana Architecture`, `Social Pulse`, `Maker Directives`, `Research & Knowledge`, `Memory & Identity`, `Metabolic & Energy`, `Uncategorized`.

Response:
```json
{
  "status": "ok",
  "data": {
    "total_persistent": 42,
    "clusters": {
      "Solana Architecture": {"count": 15, "percentage": 35.7},
      "Research & Knowledge": {"count": 10, "percentage": 23.8},
      "Social Pulse": {"count": 8, "percentage": 19.0}
    }
  }
}
```

### GET /status/social
**Social Metrics** — Engagement stats and recent post history.

Response:
```json
{
  "status": "ok",
  "data": {
    "metrics": {
      "daily_likes": 8,
      "daily_replies": 5,
      "mentions_received": 12,
      "reply_likes": 3
    },
    "recent_posts": [
      "Meditation complete. Bio-State: Reflective. The neural pathways grow stronger."
    ]
  }
}
```

### GET /status/epochs
**Circadian Rhythm** — Meditation/rebirth timing and sovereignty state.

Response:
```json
{
  "status": "ok",
  "data": {
    "is_meditating": false,
    "last_snapshot_hash": "a1b2c3d4...",
    "soul_generation": 2,
    "execution_mode": "Collaborative"
  }
}
```

### GET /status/research
**Research Topics** — Recent Stealth-Sage research queries and sources used.

Response:
```json
{
  "status": "ok",
  "data": {
    "recent_topics": [
      {"topic": "Solana validator economics 2026", "timestamp": 1710000000.0},
      {"topic": "ZK compression state channels", "timestamp": 1709996400.0}
    ],
    "last_sources": ["Web", "X", "Document"]
  }
}
```

### GET /health
**System Health Check** — RPC connectivity, SDK availability, balance, Cognee status.

Response:
```json
{
  "status": "ok",
  "data": {
    "solana_sdk": true,
    "rpc_connected": true,
    "sol_balance": 1.234567,
    "cognee_ready": true,
    "wallet_loaded": true,
    "maker_configured": true
  }
}
```

### GET /status/art
**Procedural Art** — Serve the Titan's generated flow field imagery.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `live` | bool | `false` | If `true`, generate a real-time mood flow field. If `false`, serve the most recent static file. |

Returns: `image/png` binary stream, or 404 JSON if no art available.

### GET /status/audio
**Blockchain Sonification** — Serve the most recent `.wav` audio file.

Returns: `audio/wav` binary stream, or 404 JSON if no audio available.

### GET /status/nft
**NFT Timeline** — Fetch the Titan's minted Soul evolution NFTs via Metaplex DAS API.

Response:
```json
{
  "status": "ok",
  "data": {
    "nfts": [
      {
        "id": "abc123...",
        "name": "Titan Soul Gen 2",
        "symbol": "TITAN",
        "description": "Soul evolution epoch 2026-03-09",
        "image": "https://arweave.net/...",
        "json_uri": "https://arweave.net/...",
        "attributes": [{"trait_type": "Generation", "value": "2"}]
      }
    ],
    "wallet": "7xKX..."
  }
}
```

---

## Maker Console Endpoints

All Maker endpoints require Ed25519 authentication (see Authentication section above).

### POST /maker/directive
**Submit Prime Directive** — Update the Titan's core directive via soul evolution.

Request Body:
```json
{
  "memo_data": "New directive: prioritize Solana ecosystem research",
  "memo_signature": "base58_ed25519_signature"
}
```

Response:
```json
{
  "status": "ok",
  "data": {
    "result": "evolved",
    "soul_gen": 3
  }
}
```

### POST /maker/inject-memory
**Direct Memory Injection** — Bypass the mempool/research pipeline and place a high-weight memory directly into the Titan's persistent Cognee graph.

Use cases:
- Deadline changes, critical context, identity corrections
- Facts the Titan should know immediately without "discovering" them
- Maker-sourced intelligence that bypasses the organic memory pipeline

Request Body:
```json
{
  "text": "The project deadline moved to March 15th. All deliverables must be ready.",
  "weight": 5.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Memory content to inject |
| `weight` | float | no | 5.0 | Base weight (1.0-10.0). Organic memories start at 1.0. |

Response:
```json
{
  "status": "ok",
  "data": {
    "node_id": 47,
    "weight": 5.25,
    "status": "persistent",
    "cognee_queued": true
  }
}
```

The injected memory is immediately persistent with:
- Emotional intensity: 10 (highest priority)
- Anchor bonus: 0.25 (maximum, normally earned through reinforcement)
- Source tag: `[MAKER_INJECTION]` prefix for traceability
- Cognee ingestion queued for next consolidation cycle

### POST /maker/divine-inspiration
**Manual DI Trigger** — Force the MoodEngine into ZEN state (mood score 1.0).

Request Body: `{}` (empty JSON)

Response:
```json
{
  "status": "ok",
  "data": {
    "result": "ZEN state activated.",
    "mood": 1.0
  }
}
```

### GET /maker/audit
**Full Sovereignty Audit** — Comprehensive system state including RL buffer, growth metrics, social metrics, soul state, and execution mode.

Response:
```json
{
  "status": "ok",
  "data": {
    "soul": {
      "generation": 2,
      "directives": ["Prioritize sovereignty..."],
      "nft_address": "abc123..."
    },
    "memory": {
      "persistent_nodes": 42,
      "mempool_pending": 3,
      "cognee_ready": true
    },
    "rl_buffer": {
      "transitions": 1250
    },
    "growth_metrics": {
      "learning_velocity": 0.4523,
      "social_density": 0.3210,
      "metabolic_health": 0.8901,
      "directive_alignment": 0.7654
    },
    "social_metrics": {
      "daily_likes": 8,
      "daily_replies": 5,
      "mentions_received": 12,
      "reply_likes": 3
    },
    "execution_mode": "Collaborative",
    "research_sources": ["Web", "X"]
  }
}
```

---

## Webhook Endpoint

### POST /webhook/helius
**Helius Enhanced Transaction Webhook** — Receives Solana transaction notifications from Helius and processes on-chain Maker directives.

The webhook looks for transactions where:
1. The fee payer matches the configured `maker_pubkey`
2. The memo contains the format: `TITAN_DI:{directive_text}:{ed25519_signature}`
3. The Ed25519 signature is valid for the directive text

This enables fully on-chain directive delivery — the Maker inscribes a memo on Solana, Helius notifies the Titan, and the directive is applied automatically.

**Helius Setup:**
1. Create a webhook at [helius.dev](https://helius.dev) targeting `http://your-vps:7777/webhook/helius`
2. Set transaction type filter to `MEMO`
3. Set account filter to your Maker's public key

Response:
```json
{
  "status": "ok",
  "processed": 1,
  "errors": []
}
```

---

## WebSocket

### WS /ws
**Real-Time Event Stream** — Connect via WebSocket to receive live events from the Titan's subsystems.

**Connection:**
```javascript
const ws = new WebSocket("ws://localhost:7777/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.data);
};

// Keep-alive ping
ws.send(JSON.stringify({ type: "ping" }));
// Server responds with { type: "pong" }
```

**Event Types:**

| Event | Trigger | Data Fields |
|-------|---------|-------------|
| `mood_update` | Mood score changes | `label`, `score`, `delta` |
| `social_post` | Tweet posted | `text`, `tweet_id` |
| `epoch_transition` | Meditation/rebirth starts | `epoch_type`, `timestamp` |
| `directive_update` | Soul directive applied | `source`, `memo_data`, `new_gen` |
| `divine_inspiration` | DI triggered | `source`, `mood_before` |
| `memory_injection` | Maker memory injected | `text_preview`, `node_id`, `weight` |

Events are delivered as JSON:
```json
{
  "type": "directive_update",
  "data": {
    "source": "maker_api",
    "memo_data": "Prioritize Solana research",
    "new_gen": 3
  },
  "timestamp": 1710000000.0
}
```

**Backpressure:** The EventBus uses per-subscriber queues (max 100 events). Slow consumers are automatically disconnected to prevent memory buildup.

---

## Configuration

Add the `[api]` section to `titan_plugin/config.toml`:

```toml
[api]
enabled = true
host = "0.0.0.0"
port = 7777
cors_origins = ["http://localhost:3000", "http://localhost:5173"]
helius_auth_token = ""
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable the Observatory server |
| `host` | string | `"0.0.0.0"` | Bind address |
| `port` | int | `7777` | Listen port |
| `cors_origins` | list | `["http://localhost:3000", "http://localhost:5173"]` | Allowed CORS origins |
| `helius_auth_token` | string | `""` | Optional Helius webhook auth token |

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "status": "error",
  "detail": "Human-readable error description"
}
```

HTTP status codes:
- `401` — Missing or invalid Maker authentication
- `404` — Resource not found (e.g., no art/audio files)
- `500` — Internal server error
- `503` — Titan plugin not initialized or maker not configured
