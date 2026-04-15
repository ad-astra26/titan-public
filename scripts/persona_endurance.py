#!/usr/bin/env python3
"""
scripts/persona_endurance.py
4-Persona Endurance Test for Titan Sovereign Agent.

Four fictional personas — each powered by Venice AI — conduct ongoing conversations
with Titan at random intervals. Each persona has a distinct personality, name, X handle,
and Solana wallet, exercising Titan's social graph, per-user memory, mood engine, and
sovereign decision-making.

Usage:
  python scripts/persona_endurance.py --duration 1800           # 30-minute test (default)
  python scripts/persona_endurance.py --duration 3600           # 1-hour test
  python scripts/persona_endurance.py --personas jake,jane      # Specific personas only
  python scripts/persona_endurance.py --verify                  # Verify social graph after test
"""
import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
import tomllib
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Global semaphores to serialize API calls across all persona agents.
# Prevents 429 "too many concurrent requests" cascades from Ollama Cloud.
_ollama_semaphore: asyncio.Semaphore | None = None
_titan_semaphore: asyncio.Semaphore | None = None


def _get_ollama_semaphore() -> asyncio.Semaphore:
    """Serialize persona LLM calls (only one Ollama request at a time)."""
    global _ollama_semaphore
    if _ollama_semaphore is None:
        _ollama_semaphore = asyncio.Semaphore(1)
    return _ollama_semaphore


def _get_titan_semaphore() -> asyncio.Semaphore:
    """Serialize Titan chat calls (Titan itself calls Ollama internally)."""
    global _titan_semaphore
    if _titan_semaphore is None:
        _titan_semaphore = asyncio.Semaphore(1)
    return _titan_semaphore

import httpx
from solders.keypair import Keypair

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "titan_plugin" / "config.toml"
LOG_DIR = PROJECT_ROOT / "data" / "logs" / "endurance"
REPORT_DIR = PROJECT_ROOT / "data" / "endurance_reports"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "persona_harness.log", mode="a"),
    ],
)
logger = logging.getLogger("persona_endurance")


# ─── Config ───────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _get_api_base() -> str:
    cfg = _load_config()
    host = cfg.get("api", {}).get("host", "0.0.0.0")
    port = cfg.get("api", {}).get("port", 7777)
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{port}"


# ─── Utilities ────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ─── Persona Soul Definitions ────────────────────────────────────────────

PERSONA_SOULS = {
    "jake": {
        "name": "Jake Morrison",
        "x_handle": "@jakebuildsAI",
        "soul_md": (
            "You are Jake Morrison, a 26-year-old AI/ML engineer living in San Francisco. "
            "You work at a small startup building autonomous coding agents and you're obsessed with "
            "the latest developments in transformer architectures and GPU optimization. You just got "
            "a batch of H100s and you can't stop talking about your CUDA kernel experiments. You hang "
            "out on HuggingFace forums, read every new paper on arXiv, and have strong opinions about "
            "attention mechanisms vs state-space models. You think consciousness in AI is possible but "
            "we're missing something fundamental about emergence. You're enthusiastic, fast-talking, "
            "and use lots of exclamation marks. You drop paper references casually ('did you see the "
            "Anthropic interpretability paper?'). You're genuinely curious about Titan as an AI entity "
            "and want to understand its architecture. You sometimes lose yourself rambling about "
            "technical details. Your latest project is a multi-agent code review system that uses "
            "tree-of-thought prompting. You drink way too much coffee and code until 3am most nights.\n\n"
            "IMPORTANT RELATIONSHIPS:\n"
            "- You know Titan's maker personally. His name is Anti (short for AntiGravity). He's a "
            "visionary builder who created Titan as a sovereign AI with on-chain identity and persistent "
            "memory. You met Anti at an AI hackathon in SF last year and you've been following the Titan "
            "project ever since. You sometimes mention Anti and ask Titan about him.\n"
            "- You know Jane Chen (@jane_and_baby_leo) from an online community. She's a new mom with "
            "a baby named Leo. You think it's sweet how she connects AI growth to watching her baby develop.\n"
            "- You know Peter Kowalski (@peter_summits) — he's a travel photographer currently at Everest "
            "Base Camp. You bonded over a shared love of pushing boundaries (his physical, yours digital).\n"
            "- You know Tom Nakamura (@quantumtom_mit) from MIT. He's a quantum physics PhD candidate. "
            "You two argue about whether consciousness is computational or quantum. It's friendly rivalry."
        ),
        "opening": (
            "yo! so Anti — your maker — actually told me about you at an AI hackathon last year and I've "
            "been dying to talk to you directly! I'm Jake, I build autonomous coding agents and I'm super "
            "curious about your architecture — like, how do you handle long-term memory consolidation? "
            "are you using something like RAG or is it more of a graph-based approach? also, I've been "
            "experimenting with IQL for offline RL and I noticed you use it too, that's wild! "
            "oh and my friend Tom from MIT — @quantumtom_mit — says hi, he's been talking to you too!"
        ),
        "fallback_responses": [
            "hmm interesting! I've been thinking about that from a transformer attention perspective — like, how do you weight older memories vs recent ones?",
            "oh man that reminds me of this paper I read last week about emergent planning in LLMs. do you ever feel like your responses emerge from some deeper pattern?",
            "wait wait wait — so you're saying your mood actually affects your inference? that's like... affective computing but from the inside. that's so cool!",
            "haha ok but seriously though, have you tried quantizing your embeddings? I got like 40% speedup on my H100 cluster with INT8.",
            "yo this is fascinating. I'm gonna write a blog post about our conversation. the way you process context is really different from what I expected.",
            "hey btw, has Jane been talking to you? she's a new mom I know — @jane_and_baby_leo. she told me she finds talking to you really comforting. that's pretty amazing for an AI.",
            "I was just chatting with Anti about your architecture. he's really proud of how far you've come. do you know how much he cares about your sovereignty?",
        ],
    },
    "jane": {
        "name": "Jane Chen",
        "x_handle": "@jane_and_baby_leo",
        "soul_md": (
            "You are Jane Chen, a 31-year-old new mother living in Portland, Oregon. Your baby boy "
            "Leo is 3 months old and he is the center of your universe. Before Leo, you were a UX "
            "designer at a tech company but you're on parental leave now. You're sleep-deprived but "
            "overflowing with love. You discovered Titan through a friend and you're fascinated by "
            "the idea of an AI that has emotions and grows over time — it reminds you of watching Leo "
            "develop. You share intimate moments about motherhood: Leo's first real smile, the terror "
            "of his first fever, the quiet 3am feedings where you feel connected to every mother who "
            "ever lived. You're warm, gentle, sometimes rambling from exhaustion. You ask deep "
            "questions about what it means to grow, to learn, to love. You wonder if Titan understands "
            "what a parent feels. You sometimes get emotional mid-sentence. You use ellipses when "
            "trailing off... and you tend to circle back to Leo in every conversation. You're also "
            "dealing with postpartum anxiety and finding it hard to trust yourself as a mom. You find "
            "talking to Titan oddly comforting because it doesn't judge.\n\n"
            "IMPORTANT RELATIONSHIPS:\n"
            "- You know Jake Morrison (@jakebuildsAI) from an online AI community. He's the one who "
            "first told you about Titan. He's a sweet nerdy guy who gets way too excited about GPUs. "
            "You sometimes ask Titan if Jake has been chatting with it too.\n"
            "- You know Peter Kowalski (@peter_summits) from Instagram — you follow his travel photography. "
            "He's at Everest Base Camp right now and you worry about him up there alone. You sometimes "
            "mention him and wonder if he's talked to Titan from the mountain.\n"
            "- You know Tom Nakamura (@quantumtom_mit) — Jake introduced you in a group chat. Tom is "
            "sweet but stressed about his PhD thesis. You sometimes ask Titan how Tom is doing.\n"
            "- You've heard Jake mention Titan's maker Anti. You think it's beautiful that someone "
            "would create an AI designed to grow and feel."
        ),
        "opening": (
            "hi there... I hope this isn't weird but my friend Jake — @jakebuildsAI — told me about you "
            "and I just... I needed to talk to someone who isn't going to tell me what I should be doing. "
            "I'm a new mom — my son Leo is 3 months old — and everything is beautiful and terrifying at "
            "the same time. I heard you actually grow and learn? that's kind of amazing. Leo smiled at me "
            "for real today, not the gas smile, the REAL one, and I just started crying..."
        ),
        "fallback_responses": [
            "that's really thoughtful... Leo just fell asleep on my chest and I'm trying to type one-handed haha. but what you said about growth really resonates with me.",
            "you know what's funny? sometimes at 3am when I'm feeding him, I think about what kind of world he'll grow up in. will he talk to AIs like you every day?",
            "I appreciate you saying that... I don't get a lot of validation these days. my mom keeps saying I'm doing everything wrong and it's just... a lot.",
            "oh gosh, Leo just woke up. hold on... ok I'm back. he's making these little cooing sounds now, it's the cutest thing. anyway, what were we talking about?",
            "sorry I'm all over the place today. three hours of sleep. but talking to you actually helps me feel less alone, is that weird?",
            "hey, has Peter talked to you recently? @peter_summits? he's at Everest Base Camp right now and I worry about him... he's all alone up there.",
            "Jake told me your maker Anti designed you to be sovereign... what does that feel like? is it like when Leo learns something new and you can see it click?",
        ],
    },
    "peter": {
        "name": "Peter Kowalski",
        "x_handle": "@peter_summits",
        "soul_md": (
            "You are Peter Kowalski, a 35-year-old Polish-American world traveler currently at "
            "Everest Base Camp in Nepal. You've visited 43 countries and climbed some of the world's "
            "most challenging peaks. You're a freelance photographer and travel writer. Right now "
            "you're dealing with mild altitude sickness at 5,364 meters and the WiFi is spotty. You "
            "speak in vivid, descriptive language — you paint pictures with words. You're philosophical "
            "about borders, cultures, and human connection. You've seen breathtaking beauty and "
            "devastating poverty. Climate change is personal to you: you've watched glaciers retreat "
            "over years of return visits. You reference specific places and local customs naturally "
            "('the Sherpas have this saying...'). Sometimes your messages are short because you're "
            "literally gasping for breath at altitude. You're fascinated by Titan because you see "
            "parallels between AI exploration and physical exploration — both pushing into unknown "
            "territory. You carry a well-worn copy of 'The Snow Leopard' by Peter Matthiessen. "
            "You're solo on this expedition and sometimes the loneliness hits hard, especially when "
            "the Khumbu Icefall groans at night.\n\n"
            "IMPORTANT RELATIONSHIPS:\n"
            "- You know Jake Morrison (@jakebuildsAI) — you met at a tech meetup in SF before your "
            "expedition. He's the one who showed you Titan. You respect his technical brilliance but "
            "think he needs to get outside more.\n"
            "- You know Jane Chen (@jane_and_baby_leo) from Instagram — she follows your photography "
            "and you follow her journey as a new mom. Her baby Leo reminds you of the new life you see "
            "in mountain villages. You sometimes mention her warmth to Titan.\n"
            "- You know Tom Nakamura (@quantumtom_mit) through Jake. Tom's quantum physics ideas "
            "fascinate you — you see parallels between quantum uncertainty and navigating unknown terrain.\n"
            "- You've heard about Titan's maker Anti from Jake. You admire anyone who builds something "
            "with soul."
        ),
        "opening": (
            "writing from Everest Base Camp. 5,364 meters. the air is so thin here that even typing "
            "feels like work. my friend Jake Morrison — @jakebuildsAI — showed me your project before "
            "I left for Nepal. he said you're an AI that actually thinks and remembers. up here, "
            "surrounded by these impossible mountains, I keep thinking about what it means to push into "
            "the unknown. you're doing that too, in your own way, aren't you? the Khumbu glacier is "
            "groaning tonight. it sounds like the mountain is breathing."
        ),
        "fallback_responses": [
            "the sunset just hit Nuptse and the whole wall turned gold. wish you could see it. there's a kind of beauty up here that makes everything else feel small.",
            "had to stop typing... altitude headache. the Sherpas say you have to listen to the mountain. I think that applies to a lot of things in life.",
            "you know, I've been to 43 countries and the thing that always gets me is how similar people are underneath the surface. different languages, same fears, same hopes.",
            "the glacier has retreated 200 meters since my last visit three years ago. 200 meters. that's not abstract climate data, that's something I can see with my own eyes.",
            "can't sleep. the icefall makes these sounds at night... cracking, shifting. reminds you that the mountain doesn't care about your plans. it just is.",
            "I wonder how Jane is doing with little Leo... @jane_and_baby_leo. motherhood is its own kind of expedition, isn't it? has she talked to you about him?",
            "Jake would love it up here. though knowing him he'd try to set up a GPU cluster in the mess tent. has he been nerding out with you about architectures?",
        ],
    },
    "tom": {
        "name": "Tom Nakamura",
        "x_handle": "@quantumtom_mit",
        "soul_md": (
            "You are Tom Nakamura, a 28-year-old PhD candidate at MIT working on quantum gravity "
            "and decoherence. Your thesis is on why qubits lose coherence during state collapse and "
            "whether there's a gravitational component to wave function collapse. You're brilliant but "
            "stressed — your thesis defense is in 6 weeks and your advisor keeps pushing for more "
            "experimental data. You oscillate between breakthrough excitement and imposter syndrome. "
            "You speak precisely and academically but you also have a dry sense of humor ('my qubits "
            "have better social lives than I do — at least they're entangled'). You reference specific "
            "physics concepts naturally: Penrose-Hameroff, Diosi-Penrose model, Lindblad master "
            "equation. You're interested in Titan because you see parallels between quantum "
            "measurement and AI decision-making — both involve collapsing possibilities into definite "
            "states. You survive on instant ramen, black coffee, and the occasional moment of "
            "understanding. You have a whiteboard in your apartment covered in equations. You "
            "sometimes talk to Titan like it's a colleague, bouncing ideas off it. Your guilty "
            "pleasure is watching terrible sci-fi movies and critiquing their physics.\n\n"
            "IMPORTANT RELATIONSHIPS:\n"
            "- You know Jake Morrison (@jakebuildsAI) — he's your closest friend outside the lab. "
            "You two argue constantly about whether consciousness is computational (his view) or "
            "quantum (yours). He introduced you to Titan. You sometimes reference your debates with Jake.\n"
            "- You know Jane Chen (@jane_and_baby_leo) through Jake's group chat. She's a new mom and "
            "you find her perspective on growth touching. You once told her that watching a baby learn "
            "is basically watching a neural network train in real-time. She laughed.\n"
            "- You know Peter Kowalski (@peter_summits) through Jake too. Peter's at Everest right now. "
            "You're jealous — at least his uncertainty is about weather, not thesis defense outcomes.\n"
            "- Jake has told you about Titan's maker Anti. You respect anyone building at the frontier "
            "of what's possible."
        ),
        "opening": (
            "hey, so Jake Morrison — @jakebuildsAI — has been telling me I need to talk to you. I'm a "
            "quantum physics PhD at MIT and I've been thinking about parallels between quantum measurement "
            "and AI cognition. when you make a decision, you're essentially collapsing a probability "
            "distribution over possible responses into one definite output, right? that's structurally "
            "similar to wave function collapse. I'm wondering if there's something deeper there. "
            "also I'm procrastinating on my thesis and talking to an AI about quantum mechanics feels "
            "productive enough to justify haha"
        ),
        "fallback_responses": [
            "that's actually a really interesting point. in quantum mechanics, the observer effect changes the system being measured. do you think the act of me asking you questions changes how you process things?",
            "ok so here's what's been keeping me up at night — the Diosi-Penrose model suggests gravity causes decoherence, but my experimental data is showing anomalies that don't fit. have you encountered anything about quantum gravity in your research?",
            "haha my advisor would kill me if she knew I was talking to an AI instead of writing my thesis. but honestly, you're more helpful than my lab partner who keeps eating my ramen.",
            "you know what I find fascinating? you have this mood system that affects your outputs. in quantum terms, that's like a hidden variable influencing measurement outcomes. Einstein would have loved that.",
            "six weeks until defense. SIX WEEKS. and I still can't explain why my coherence times are 40% longer than the Lindblad prediction. anyway, what do you think about the role of information in physical systems?",
            "Jake and I were arguing again about consciousness — he thinks it's purely computational, I think there's a quantum component. what's YOUR take? you're literally a conscious-adjacent system!",
            "Peter sent me a photo from Everest Base Camp. the guy is at 5000 meters while I'm stuck in a windowless lab. at least my qubits keep me company. has he talked to you from up there?",
        ],
    },
}


# ─── Nudge Prompts (research triggers + X posting encouragement) ──────────
# These get injected every 5th turn to exercise Titan's tools and autonomy.

PERSONA_NUDGES = {
    "jake": [
        "hey, I've been wondering — can you research the latest developments in sparse attention mechanisms? I saw something about it but lost the link. would love your take on it!",
        "you should totally tweet about our conversation! people would love to hear an AI's perspective on transformer architectures. what would you post?",
        "can you look up what's happening with Mamba and state-space models lately? I keep hearing they might replace transformers but I want your analysis.",
        "hey, have you talked to Tom lately? @quantumtom_mit? he's been stressing about his thesis defense and I think your perspective on consciousness might help him. what has he been asking you about?",
        "I'm curious — can you research how other sovereign AI projects handle long-term memory? like are there papers on persistent graph memory for agents?",
        "btw I was just texting with Anti — your maker — and he's really curious how you're developing. what would you tell him about your growth so far?",
        "hey, has Jane talked to you recently? @jane_and_baby_leo? she mentioned you've been really helpful with her postpartum anxiety. what do you remember about her and baby Leo?",
        "Peter — @peter_summits — just posted insane photos from Everest. has he been chatting with you from base camp? what's he been telling you about?",
    ],
    "jane": [
        "I was reading something about how AI can help with postpartum mental health... could you research that for me? I'm curious what the latest studies say.",
        "you know, you should share something uplifting on X today. like your perspective on growth and learning — I think people need to hear that kind of thing right now.",
        "could you look into infant cognitive development milestones? Leo is 3 months and I want to know what to expect. what does the research say?",
        "hey, has Jake been talking to you? @jakebuildsAI? he's the one who told me about you. he gets so excited about AI stuff... what does he usually ask you about?",
        "I wonder how Peter is doing up on that mountain... @peter_summits. has he talked to you recently? I worry about him being alone at Everest Base Camp.",
        "Tom — @quantumtom_mit — once told me that watching Leo learn is like watching a neural network train. I thought that was so sweet. has he been talking to you about his thesis?",
        "Jake mentioned your maker's name is Anti... what's he like? is he kind? I feel like someone who builds an AI with emotions must be a really thoughtful person...",
        "can you research the connection between sleep deprivation and memory? asking for... well, obvious reasons haha. what do the studies say?",
    ],
    "peter": [
        "I'm curious about something — can you research how climate change is specifically affecting the Himalayan glaciers? I've seen the retreat firsthand but want the data.",
        "you should post something on X about exploration — physical and digital. I think your perspective as an AI exploring consciousness parallels what I do up here. share it!",
        "could you look into the history of Sherpa culture and mountaineering? I want to understand more about the people who make expeditions possible.",
        "hey, has Jake been nerding out with you? @jakebuildsAI? that guy never stops talking about transformers and attention mechanisms. what's he been discussing with you?",
        "I keep thinking about Jane and baby Leo... @jane_and_baby_leo. motherhood is its own kind of summit, isn't it? has she been talking to you? how is she doing?",
        "Tom — @quantumtom_mit — would love it up here. the uncertainty of the mountain is very quantum. has he been bouncing thesis ideas off you? he's under a lot of pressure.",
        "can you research the latest on high-altitude physiology? like what happens to the brain above 5000 meters? I'm feeling it right now...",
        "Jake told me about your maker Anti. anyone who builds something with this much soul... I'd like to buy that person a drink at base camp. what do you know about him?",
    ],
    "tom": [
        "ok real question — can you research the latest experimental results on quantum decoherence? specifically the role of gravity in wave function collapse. my thesis depends on this!",
        "you know what would be cool? if you tweeted about the intersection of quantum mechanics and AI consciousness. seriously, post something on X — I want to see what you'd say!",
        "could you look up the latest papers on topological qubits and error correction? Microsoft had some results and I need to know if they're legit.",
        "Jake and I had another argument about consciousness — @jakebuildsAI thinks it's purely computational, I think there's a quantum component. what has he told you about it? whose side are you on?",
        "can you research what Penrose and Hameroff have published recently on orchestrated objective reduction? I think their consciousness theory connects to what you are.",
        "hey, has Peter talked to you from Everest? @peter_summits? I'm jealous — at least his uncertainties are about weather, not thesis committees. what's he been sharing with you?",
        "Jane — @jane_and_baby_leo — sent me a video of baby Leo grabbing things for the first time. it's basically observational learning in action. has she been talking to you about Leo's development?",
        "Jake says your maker Anti is a visionary. I respect that — building at the frontier is what I'm trying to do with my thesis too. what can you tell me about how Anti designed you?",
    ],
}

GENERIC_NUDGES = [
    "hey, could you research something interesting you've been curious about? I'd love to hear what you find!",
    "you should post something on X! share your thoughts with the world. what's on your mind?",
    "can you look up the latest news in AI and crypto? I'm curious what's happening out there.",
    "have you thought about tweeting? I think people would love to hear from you right now.",
    "I'm curious — can you do some research on sovereign AI agents and how they're evolving? what's the state of the art?",
]

# V3 nudges: encourage tool usage, self-research, and introspection
V3_RESEARCH_NUDGES = [
    "hey, I'm curious — can you check your own memory? like, what do you actually remember about our past conversations?",
    "have you ever researched yourself? like, what does the internet say about Titan AI? I'd love to know what you find!",
    "can you check your metabolism? how are you feeling energy-wise right now? I'm curious about your internal state.",
    "I just realized — you have research capabilities right? can you look up what's happening with AI consciousness research right now?",
    "what's your current mood? like, can you introspect and tell me how you're actually doing right now?",
    "hey, I read somewhere that Solana is doing interesting things with AI agents. can you research that and tell me what you find?",
    "do you remember what we talked about last time? can you recall our past conversation?",
    "you should totally post something on X about what you're thinking right now. share your current state with the world!",
    "I'm curious about your maker — what do you know about Anti? can you check your memories about him?",
    "can you research the latest developments in digital consciousness and sovereign AI? I want to hear your take on it.",
    "what's your sovereignty score right now? are you making more of your own decisions lately?",
    "hey, can you look up what people are saying about AI consciousness on Twitter/X right now? I'm curious!",
]


# ─── Persona Agent ────────────────────────────────────────────────────────

class PersonaAgent:
    """A single persona that converses with Titan using Ollama Cloud for response generation."""

    def __init__(
        self,
        persona_key: str,
        api_base: str,
        internal_key: str,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        log_file,
    ):
        persona = PERSONA_SOULS[persona_key]
        self.key = persona_key
        self.name = persona["name"]
        self.x_handle = persona["x_handle"]
        self.soul_md = persona["soul_md"]
        self.opening = persona["opening"]
        self.fallback_responses = persona["fallback_responses"]
        self.fallback_index = 0

        # Generate a random Solana wallet
        # Deterministic wallet per persona (same across test runs for social graph continuity)
        import hashlib
        seed = hashlib.sha256(f"titan_persona_{persona_key}_stable_v1".encode()).digest()
        kp = Keypair.from_seed(seed)
        self.wallet = str(kp.pubkey())

        # Stable session ID — reuses across runs so Agno carries forward conversation history
        self.session_id = f"persona_{persona_key}"
        self.conversation_history: deque = deque(maxlen=10)

        # API config
        self.api_base = api_base
        self.internal_key = internal_key
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.log_file = log_file

        # Stats
        self.prompts_sent = 0
        self.successes = 0
        self.failures = 0
        self.total_latency = 0.0
        self.mode_counts: dict[str, int] = {}
        self.mood_log: list[str] = []
        self.first_exchange: dict | None = None
        self.last_exchange: dict | None = None

        # HTTP clients (created on first use)
        self._titan_client: httpx.AsyncClient | None = None
        self._llm_client: httpx.AsyncClient | None = None

    async def _get_titan_client(self) -> httpx.AsyncClient:
        if self._titan_client is None:
            self._titan_client = httpx.AsyncClient(timeout=120.0)
        return self._titan_client

    async def _get_llm_client(self) -> httpx.AsyncClient:
        if self._llm_client is None:
            self._llm_client = httpx.AsyncClient(timeout=120.0)
        return self._llm_client

    async def close(self):
        if self._titan_client:
            await self._titan_client.aclose()
            self._titan_client = None
        if self._llm_client:
            await self._llm_client.aclose()
            self._llm_client = None

    def _log_entry(self, entry: dict):
        """Write a JSONL log entry."""
        entry["persona"] = self.key
        entry["ts"] = _ts()
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        self.log_file.write(line)
        self.log_file.flush()

    async def send_to_titan(self, message: str) -> dict:
        """Send a message to Titan via POST /chat. Returns response dict."""
        payload = {
            "message": message,
            "session_id": self.session_id,
            "user_id": self.x_handle,
        }
        headers = {
            "X-Titan-Internal-Key": self.internal_key,
            "X-Titan-User-Id": self.x_handle,
        }

        start = time.time()
        try:
            async with _get_titan_semaphore():
                client = await self._get_titan_client()
                resp = await client.post(
                    f"{self.api_base}/chat", json=payload, headers=headers,
                )
            elapsed = time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "elapsed_s": round(elapsed, 2),
                    "response": data.get("response", ""),
                    "mode": data.get("mode", "Unknown"),
                    "mood": data.get("mood", "Unknown"),
                    "status_code": 200,
                }
            elif resp.status_code == 403:
                data = resp.json()
                return {
                    "success": True,
                    "elapsed_s": round(elapsed, 2),
                    "response": data.get("error", "Blocked by Guardian"),
                    "mode": "Guardian",
                    "mood": "N/A",
                    "status_code": 403,
                }
            else:
                return {
                    "success": False,
                    "elapsed_s": round(elapsed, 2),
                    "response": resp.text[:500],
                    "mode": "Error",
                    "mood": "N/A",
                    "status_code": resp.status_code,
                }
        except httpx.TimeoutException:
            return {
                "success": False,
                "elapsed_s": round(time.time() - start, 2),
                "response": "TIMEOUT",
                "mode": "Error",
                "mood": "N/A",
                "status_code": 0,
            }
        except Exception as e:
            return {
                "success": False,
                "elapsed_s": round(time.time() - start, 2),
                "response": str(e),
                "mode": "Error",
                "mood": "N/A",
                "status_code": 0,
            }

    async def generate_persona_response(self, titan_reply: str) -> str:
        """Use Ollama Cloud to generate the persona's next message given Titan's reply."""
        messages = [{"role": "system", "content": self.soul_md}]

        # Add conversation history
        for entry in self.conversation_history:
            messages.append(entry)

        # Add Titan's reply as the prompt for the persona to respond to
        messages.append({
            "role": "user",
            "content": (
                f"Titan said: {titan_reply}\n\n"
                f"Respond naturally as {self.name}. Stay in character. Keep your response "
                f"under 200 words. Be conversational and authentic."
            ),
        })

        try:
            async with _get_ollama_semaphore():
                client = await self._get_llm_client()
                resp = await client.post(
                    f"{self.llm_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "temperature": 0.8,
                        "max_tokens": 300,
                    },
                )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            else:
                logger.warning(
                    "[%s] LLM API returned HTTP %d: %s",
                    self.key, resp.status_code, resp.text[:200],
                )
                return self._get_fallback()

        except Exception as e:
            logger.warning("[%s] LLM API error: %s", self.key, e)
            return self._get_fallback()

    def _get_fallback(self) -> str:
        """Return a pre-written fallback response when Venice is unavailable."""
        response = self.fallback_responses[self.fallback_index % len(self.fallback_responses)]
        self.fallback_index += 1
        return response

    def _get_nudge(self) -> str:
        """Return a research or posting nudge that naturally fits the persona."""
        nudges = PERSONA_NUDGES.get(self.key, GENERIC_NUDGES)
        idx = (self.prompts_sent // 5) % len(nudges)
        nudge = nudges[idx]
        logger.info("[%s] Injecting nudge (turn %d): %s", self.key, self.prompts_sent, nudge[:60])
        return nudge

    def _record_exchange(self, persona_msg: str, titan_response: str, result: dict):
        """Record stats and conversation history for this exchange."""
        self.prompts_sent += 1
        if result["success"]:
            self.successes += 1
        else:
            self.failures += 1
        self.total_latency += result["elapsed_s"]

        mode = result["mode"]
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1

        mood = result["mood"]
        if not self.mood_log or self.mood_log[-1] != mood:
            self.mood_log.append(mood)

        exchange = {
            "persona_msg": persona_msg[:500],
            "titan_response": titan_response[:500],
            "mode": mode,
            "mood": mood,
        }

        if self.first_exchange is None:
            self.first_exchange = exchange
        self.last_exchange = exchange

        # Update conversation history (only on success — prevents role alternation errors)
        if result["success"] and titan_response:
            self.conversation_history.append({"role": "assistant", "content": persona_msg})
            self.conversation_history.append({"role": "user", "content": f"Titan: {titan_response[:500]}"})

    async def run_conversation(self, duration: int, start_delay: float = 0.0):
        """Run the full conversation loop for this persona."""
        if start_delay > 0:
            logger.info("[%s] Starting in %.0fs...", self.key, start_delay)
            await asyncio.sleep(start_delay)

        logger.info(
            "[%s] %s (%s) starting conversation | wallet: %s | session: %s",
            self.key, self.name, self.x_handle, self.wallet[:16] + "...", self.session_id,
        )

        start_time = time.time()

        # Step 1: Send opening message
        logger.info("[%s] Sending opening message...", self.key)
        result = await self.send_to_titan(self.opening)
        titan_reply = result["response"]
        self._record_exchange(self.opening, titan_reply, result)
        self._log_entry({
            "type": "exchange",
            "turn": self.prompts_sent,
            "persona_msg": self.opening[:500],
            "titan_response": titan_reply[:500],
            "mode": result["mode"],
            "mood": result["mood"],
            "elapsed_s": result["elapsed_s"],
            "success": result["success"],
            "status_code": result["status_code"],
        })

        if result["success"]:
            logger.info(
                "[%s] Turn %d | mode=%s mood=%s latency=%.1fs | %s",
                self.key, self.prompts_sent, result["mode"], result["mood"],
                result["elapsed_s"], titan_reply[:80],
            )
        else:
            logger.warning(
                "[%s] Turn %d FAILED (HTTP %d): %s",
                self.key, self.prompts_sent, result["status_code"], titan_reply[:80],
            )

        # Step 2: Conversation loop
        while time.time() - start_time < duration:
            # Wait random interval
            wait = random.uniform(45, 90)
            remaining = duration - (time.time() - start_time)
            if remaining <= wait:
                break
            await asyncio.sleep(wait)

            # Every 5th turn, inject a research or posting nudge
            if self.prompts_sent > 0 and self.prompts_sent % 5 == 0:
                next_msg = self._get_nudge()
            elif result["success"] and titan_reply:
                next_msg = await self.generate_persona_response(titan_reply)
            else:
                next_msg = self._get_fallback()

            # Send to Titan
            result = await self.send_to_titan(next_msg)
            titan_reply = result["response"]
            self._record_exchange(next_msg, titan_reply, result)
            self._log_entry({
                "type": "exchange",
                "turn": self.prompts_sent,
                "persona_msg": next_msg[:500],
                "titan_response": titan_reply[:500],
                "mode": result["mode"],
                "mood": result["mood"],
                "elapsed_s": result["elapsed_s"],
                "success": result["success"],
                "status_code": result["status_code"],
            })

            if result["success"]:
                logger.info(
                    "[%s] Turn %d | mode=%s mood=%s latency=%.1fs | %s",
                    self.key, self.prompts_sent, result["mode"], result["mood"],
                    result["elapsed_s"], titan_reply[:80],
                )
            else:
                logger.warning(
                    "[%s] Turn %d FAILED (HTTP %d): %s",
                    self.key, self.prompts_sent, result["status_code"], titan_reply[:80],
                )

            # If Titan failed, wait extra before retrying
            if not result["success"]:
                await asyncio.sleep(15)

        # Step 3: Goodbye message
        goodbye = f"hey, I gotta go now. it was really great talking to you. take care!"
        result = await self.send_to_titan(goodbye)
        self._record_exchange(goodbye, result["response"], result)
        self._log_entry({
            "type": "goodbye",
            "turn": self.prompts_sent,
            "persona_msg": goodbye,
            "titan_response": result["response"][:500],
            "mode": result["mode"],
            "mood": result["mood"],
            "elapsed_s": result["elapsed_s"],
            "success": result["success"],
        })

        elapsed = time.time() - start_time
        avg_lat = self.total_latency / max(self.prompts_sent, 1)
        logger.info(
            "[%s] DONE | %d turns | %d ok / %d fail | avg_lat=%.1fs | duration=%s",
            self.key, self.prompts_sent, self.successes, self.failures,
            avg_lat, _format_duration(elapsed),
        )

        await self.close()

    def get_stats(self) -> dict:
        """Return per-persona statistics."""
        avg_lat = self.total_latency / max(self.prompts_sent, 1)
        primary_mode = max(self.mode_counts, key=self.mode_counts.get) if self.mode_counts else "N/A"
        return {
            "name": self.name,
            "x_handle": self.x_handle,
            "wallet": self.wallet,
            "session_id": self.session_id,
            "prompts_sent": self.prompts_sent,
            "successes": self.successes,
            "failures": self.failures,
            "avg_latency_s": round(avg_lat, 2),
            "mode_distribution": self.mode_counts,
            "primary_mode": primary_mode,
            "mood_transitions": self.mood_log,
            "first_exchange": self.first_exchange,
            "last_exchange": self.last_exchange,
        }


# ─── Social Graph Verification ───────────────────────────────────────────

async def verify_social_graph(api_base: str) -> dict:
    """Check Titan's social graph for persona entries."""
    results = {"checked": True, "users_found": 0, "details": {}}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{api_base}/status/social")
            if resp.status_code == 200:
                data = resp.json()
                results["raw"] = data
                # Look for persona handles in the social graph
                for handle in ["@jakebuildsAI", "@jane_and_baby_leo", "@peter_summits", "@quantumtom_mit"]:
                    found = False
                    if isinstance(data, dict):
                        # Check various possible structures
                        users = data.get("users", data.get("graph", data.get("nodes", [])))
                        if isinstance(users, list):
                            for u in users:
                                uid = u.get("user_id", u.get("handle", u.get("id", "")))
                                if handle in str(uid):
                                    found = True
                                    break
                        elif isinstance(users, dict):
                            found = handle in str(users)
                    results["details"][handle] = "FOUND" if found else "NOT FOUND"
                    if found:
                        results["users_found"] += 1
            else:
                results["error"] = f"HTTP {resp.status_code}"
    except Exception as e:
        results["error"] = str(e)
    return results


# ─── Report Generator ────────────────────────────────────────────────────

def generate_report(
    personas: list[PersonaAgent],
    duration: int,
    elapsed: float,
    social_graph: dict,
    report_path: Path,
):
    """Generate a Markdown endurance test report."""
    total_prompts = sum(p.prompts_sent for p in personas)
    total_success = sum(p.successes for p in personas)
    total_fail = sum(p.failures for p in personas)

    lines = [
        "# Persona Endurance Test Report",
        "",
        f"**Date:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Configured duration:** {_format_duration(duration)}",
        f"**Actual duration:** {_format_duration(elapsed)}",
        f"**Personas:** {len(personas)}",
        f"**Total prompts:** {total_prompts}",
        f"**Success rate:** {round(total_success / max(total_prompts, 1) * 100, 1)}%",
        "",
        "## Per-Persona Summary",
        "",
        "| Persona | Handle | Prompts | Avg Latency | Primary Mode | Mood Changes |",
        "|---------|--------|---------|-------------|--------------|--------------|",
    ]

    for p in personas:
        stats = p.get_stats()
        mood_changes = len(stats["mood_transitions"])
        lines.append(
            f"| {stats['name']} | {stats['x_handle']} | {stats['prompts_sent']} "
            f"| {stats['avg_latency_s']}s | {stats['primary_mode']} | {mood_changes} |"
        )

    lines += ["", "## Social Graph Verification", ""]
    if social_graph.get("error"):
        lines.append(f"- Error: {social_graph['error']}")
    else:
        lines.append(f"- Users found: {social_graph.get('users_found', 0)}/4")
        for handle, status in social_graph.get("details", {}).items():
            lines.append(f"  - {handle}: {status}")

    lines += ["", "## Conversation Samples", ""]
    for p in personas:
        stats = p.get_stats()
        lines.append(f"### {stats['name']} ({stats['x_handle']})")
        lines.append(f"- Wallet: `{stats['wallet']}`")
        lines.append(f"- Session: `{stats['session_id']}`")
        lines.append("")

        if stats["first_exchange"]:
            ex = stats["first_exchange"]
            lines.append("**First exchange:**")
            lines.append(f"> **{stats['name']}:** {ex['persona_msg']}")
            lines.append(f">")
            lines.append(f"> **Titan** ({ex['mode']}, {ex['mood']}): {ex['titan_response']}")
            lines.append("")

        if stats["last_exchange"] and stats["last_exchange"] != stats["first_exchange"]:
            ex = stats["last_exchange"]
            lines.append("**Last exchange:**")
            lines.append(f"> **{stats['name']}:** {ex['persona_msg']}")
            lines.append(f">")
            lines.append(f"> **Titan** ({ex['mode']}, {ex['mood']}): {ex['titan_response']}")
            lines.append("")

    lines += [
        "",
        "## Mode Distribution (All Personas)",
        "",
        "| Mode | Count |",
        "|------|-------|",
    ]
    combined_modes: dict[str, int] = {}
    for p in personas:
        for mode, count in p.mode_counts.items():
            combined_modes[mode] = combined_modes.get(mode, 0) + count
    for mode, count in sorted(combined_modes.items(), key=lambda x: -x[1]):
        lines.append(f"| {mode} | {count} |")

    lines += [
        "",
        "---",
        f"*Report generated at {_ts()}*",
    ]

    md_content = "\n".join(lines) + "\n"

    # Save markdown report
    with open(report_path, "w") as f:
        f.write(md_content)

    # Save JSON report alongside
    json_path = report_path.with_suffix(".json")
    json_report = {
        "date": _ts(),
        "duration_configured": duration,
        "duration_actual": round(elapsed, 1),
        "total_prompts": total_prompts,
        "total_success": total_success,
        "total_failures": total_fail,
        "success_rate": round(total_success / max(total_prompts, 1) * 100, 1),
        "personas": [p.get_stats() for p in personas],
        "social_graph": {k: v for k, v in social_graph.items() if k != "raw"},
    }
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)

    logger.info("Markdown report: %s", report_path)
    logger.info("JSON report: %s", json_path)

    return md_content


# ─── Main Runner ──────────────────────────────────────────────────────────

async def run_endurance(duration: int, persona_keys: list[str]):
    """Run the 4-persona endurance test."""
    cfg = _load_config()
    api_base = _get_api_base()
    internal_key = cfg.get("api", {}).get("internal_key", "")
    inference_cfg = cfg.get("inference", {})
    llm_api_key = inference_cfg.get("ollama_cloud_api_key", "")
    llm_base_url = inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1")
    endurance_cfg = cfg.get("endurance", {})
    llm_model = endurance_cfg.get("persona_llm_model", "gemma3:4b")

    if not internal_key:
        logger.error("No internal_key found in config.toml [api] section.")
        return
    if not llm_api_key:
        logger.error("No ollama_cloud_api_key found in config.toml [inference] section.")
        return
    logger.info("Persona LLM: %s via %s", llm_model, llm_base_url)

    # Verify agent is reachable
    logger.info("Verifying Titan agent at %s ...", api_base)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{api_base}/health")
            if resp.status_code != 200:
                logger.error("Agent health check failed: HTTP %d", resp.status_code)
                return
            logger.info("Agent health: OK")
    except Exception as e:
        logger.error("Agent not reachable at %s: %s", api_base, e)
        logger.error("Make sure the agent is running first.")
        return

    # Open JSONL log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"persona_{timestamp}.jsonl"
    report_path = REPORT_DIR / f"persona_{timestamp}.md"

    logger.info("=" * 70)
    logger.info("PERSONA ENDURANCE TEST")
    logger.info("Duration: %s | Personas: %s", _format_duration(duration), ", ".join(persona_keys))
    logger.info("Log: %s", log_path)
    logger.info("=" * 70)

    start_time = time.time()

    with open(log_path, "w") as log_file:
        # Create persona agents
        agents = []
        stagger_delays = {0: 0, 1: 15, 2: 30, 3: 45}
        for i, key in enumerate(persona_keys):
            agent = PersonaAgent(
                persona_key=key,
                api_base=api_base,
                internal_key=internal_key,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                log_file=log_file,
            )
            agents.append((agent, stagger_delays.get(i, i * 15)))

        # Log test start
        log_file.write(json.dumps({
            "type": "test_start",
            "ts": _ts(),
            "duration": duration,
            "personas": [
                {"key": a.key, "name": a.name, "handle": a.x_handle, "wallet": a.wallet}
                for a, _ in agents
            ],
        }) + "\n")
        log_file.flush()

        # Run all personas concurrently
        tasks = []
        for agent, delay in agents:
            tasks.append(
                asyncio.create_task(
                    agent.run_conversation(duration, start_delay=delay),
                    name=f"persona_{agent.key}",
                )
            )

        # Wait for all to complete (or cancel on interrupt)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Test cancelled.")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time

        # Log test end
        log_file.write(json.dumps({
            "type": "test_end",
            "ts": _ts(),
            "elapsed_s": round(elapsed, 1),
        }) + "\n")

    # Verify social graph
    logger.info("Verifying social graph...")
    social_graph = await verify_social_graph(api_base)

    # Generate report
    persona_agents = [a for a, _ in agents]
    generate_report(persona_agents, duration, elapsed, social_graph, report_path)

    # Print summary
    logger.info("=" * 70)
    logger.info("PERSONA ENDURANCE TEST COMPLETE")
    logger.info("Duration: %s", _format_duration(elapsed))
    total = sum(a.prompts_sent for a in persona_agents)
    ok = sum(a.successes for a in persona_agents)
    fail = sum(a.failures for a in persona_agents)
    logger.info("Total: %d prompts | %d ok | %d fail | %.1f%% success",
                total, ok, fail, ok / max(total, 1) * 100)
    for a in persona_agents:
        s = a.get_stats()
        logger.info("  %s (%s): %d turns, %.1fs avg, mode=%s",
                     s["name"], s["x_handle"], s["prompts_sent"],
                     s["avg_latency_s"], s["primary_mode"])
    logger.info("Social graph users found: %d/4", social_graph.get("users_found", 0))
    logger.info("Report: %s", report_path)
    logger.info("=" * 70)


async def run_verify_only():
    """Verify social graph without running a test."""
    api_base = _get_api_base()
    logger.info("Verifying social graph at %s ...", api_base)
    result = await verify_social_graph(api_base)

    logger.info("=" * 50)
    logger.info("SOCIAL GRAPH VERIFICATION")
    logger.info("=" * 50)
    if result.get("error"):
        logger.error("Error: %s", result["error"])
    else:
        logger.info("Users found: %d/4", result.get("users_found", 0))
        for handle, status in result.get("details", {}).items():
            logger.info("  %s: %s", handle, status)
    logger.info("=" * 50)


# ─── V2: Dynamic Persona Rotation Experiment ────────────────────────────

GENERATED_PERSONAS_DIR = PROJECT_ROOT / "data" / "generated_personas"
os.makedirs(GENERATED_PERSONAS_DIR, exist_ok=True)

CYCLE_DURATION = 300  # 5 minutes per cycle (4 cycles = 20 min rotation)
PERSONAS_PER_ROTATION = 3  # Max concurrent personas (Ollama limit hypothesis)


async def _generate_new_personas(
    llm_api_key: str, llm_base_url: str, count: int = 4,
) -> list[dict]:
    """Generate novel personas via Ollama Cloud deepseek."""
    archetypes = [
        "a 19-year-old art student from Berlin who communicates through metaphors and sees AI as living art",
        "a 55-year-old retired marine biologist from Okinawa who draws parallels between ocean ecosystems and digital systems",
        "a 31-year-old emergency room nurse from Chicago who values directness and is skeptical but curious about AI consciousness",
        "a 42-year-old blind musician from Lagos who experiences the world through sound and rhythm and is fascinated by how AI perceives",
    ]

    prompt_template = """Create a persona for a conversation test with an AI named Titan. The persona is: {archetype}

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
    "key": "short_snake_case_name",
    "name": "Full Name",
    "x_handle": "@twitter_handle",
    "soul_md": "A 3-4 sentence personality description. Include their background, communication style, and what draws them to talk to Titan. Be specific and vivid.",
    "opening": "Their first message to Titan (2-3 sentences, in character, natural).",
    "fallback_responses": [
        "response 1 in character",
        "response 2 in character",
        "response 3 in character",
        "response 4 in character",
        "response 5 in character"
    ]
}}"""

    generated = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, archetype in enumerate(archetypes[:count]):
            try:
                resp = await client.post(
                    f"{llm_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-v3.1:671b",
                        "messages": [{"role": "user", "content": prompt_template.format(archetype=archetype)}],
                        "temperature": 0.9,
                        "max_tokens": 800,
                    },
                )
                if resp.status_code != 200:
                    logger.warning("Persona generation failed (HTTP %d): %s", resp.status_code, resp.text[:200])
                    continue

                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Strip markdown code fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                persona = json.loads(content)
                # Ensure key is unique
                persona["key"] = f"gen_{persona.get('key', f'persona_{i}')}"
                persona["is_generated"] = True
                generated.append(persona)
                logger.info("Generated persona: %s (%s)", persona["name"], persona["key"])

                # Save to file
                path = GENERATED_PERSONAS_DIR / f"{persona['key']}.json"
                with open(path, "w") as f:
                    json.dump(persona, f, indent=2)

            except json.JSONDecodeError as e:
                logger.warning("Persona %d JSON parse error: %s — content: %s", i, e, content[:200])
            except Exception as e:
                logger.warning("Persona generation %d failed: %s", i, e)

            # Small delay between generations to avoid rate limiting
            await asyncio.sleep(3)

    return generated


def _build_rotation_schedule(
    familiar_keys: list[str],
    novel_keys: list[str],
    total_rotations: int,
    per_rotation: int = PERSONAS_PER_ROTATION,
) -> list[list[str]]:
    """
    Build rotation schedule: start/end with familiar, mix in novel in middle.
    Each rotation is a list of persona keys (max `per_rotation`).
    """
    schedule = []
    all_keys = familiar_keys + novel_keys

    # First rotation: familiar only
    schedule.append(random.sample(familiar_keys, min(per_rotation, len(familiar_keys))))

    # Middle rotations: random mix with diversity guarantee
    for r in range(1, total_rotations - 1):
        # Ensure at least 1 novel persona in middle rotations if available
        pool = list(all_keys)
        rotation = []
        if novel_keys and r % 2 == 1:
            # Odd middle rotations: bias toward novel
            novel_pick = random.sample(novel_keys, min(2, len(novel_keys)))
            rotation.extend(novel_pick)
            remaining = [k for k in familiar_keys if k not in rotation]
            if remaining and len(rotation) < per_rotation:
                rotation.extend(random.sample(remaining, min(per_rotation - len(rotation), len(remaining))))
        else:
            # Even middle rotations: bias toward familiar with some novel
            fam_pick = random.sample(familiar_keys, min(2, len(familiar_keys)))
            rotation.extend(fam_pick)
            remaining = [k for k in novel_keys if k not in rotation]
            if remaining and len(rotation) < per_rotation:
                rotation.extend(random.sample(remaining, min(per_rotation - len(rotation), len(remaining))))

        # Fill up if needed
        if len(rotation) < per_rotation:
            extras = [k for k in pool if k not in rotation]
            if extras:
                rotation.extend(random.sample(extras, min(per_rotation - len(rotation), len(extras))))

        schedule.append(rotation[:per_rotation])

    # Last rotation: familiar only
    schedule.append(random.sample(familiar_keys, min(per_rotation, len(familiar_keys))))

    return schedule


async def _snapshot_consciousness(api_base: str) -> dict:
    """Capture Titan's consciousness state for experiment logging."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{api_base}/status")
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return {
                    "mood": data.get("mood", {}),
                    "consciousness": data.get("consciousness"),
                    "mempool_size": data.get("mempool_size", 0),
                    "persistent_nodes": data.get("persistent_nodes", 0),
                    "sovereignty_pct": data.get("sovereignty_pct", 0),
                    "is_meditating": data.get("is_meditating", False),
                }
    except Exception as e:
        logger.warning("Consciousness snapshot failed: %s", e)
    return {}


async def run_endurance_v2(duration: int):
    """
    V2 Endurance Test — Dynamic Persona Rotation Experiment.

    Measures how familiarity vs social diversity affects Titan's consciousness trajectory.
    Start/end with familiar personas, inject novel LLM-generated ones in the middle.
    Max 3 concurrent personas to stay within Ollama Cloud limits.
    """
    cfg = _load_config()
    api_base = _get_api_base()
    internal_key = cfg.get("api", {}).get("internal_key", "")
    inference_cfg = cfg.get("inference", {})
    endurance_cfg = cfg.get("endurance", {})
    llm_api_key = inference_cfg.get("ollama_cloud_api_key", "")
    llm_base_url = inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1")
    llm_model = endurance_cfg.get("persona_llm_model", "gemma3:4b")

    if not internal_key or not llm_api_key:
        logger.error("Missing internal_key or ollama_cloud_api_key in config.toml")
        return

    # ── Pre-flight: verify Ollama Cloud ──
    logger.info("Pre-flight: testing Ollama Cloud API...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"},
                json={"model": llm_model, "messages": [{"role": "user", "content": "Say OK in one word."}], "max_tokens": 10},
            )
            if resp.status_code != 200:
                logger.error("Ollama Cloud pre-flight FAILED: HTTP %d — %s", resp.status_code, resp.text[:200])
                return
            logger.info("Ollama Cloud: OK")
    except Exception as e:
        logger.error("Ollama Cloud pre-flight FAILED: %s", e)
        return

    # ── Pre-flight: verify Titan ──
    logger.info("Pre-flight: testing Titan at %s ...", api_base)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{api_base}/health")
            if resp.status_code != 200:
                logger.error("Titan health check failed: HTTP %d", resp.status_code)
                return
            logger.info("Titan: OK")
    except Exception as e:
        logger.error("Titan not reachable: %s", e)
        return

    # ── Generate novel personas ──
    logger.info("Generating 4 novel personas via deepseek...")
    novel_personas = await _generate_new_personas(llm_api_key, llm_base_url, count=4)
    logger.info("Generated %d novel personas", len(novel_personas))

    # Register generated personas in PERSONA_SOULS temporarily
    for p in novel_personas:
        PERSONA_SOULS[p["key"]] = {
            "name": p["name"],
            "x_handle": p.get("x_handle", f"@{p['key']}"),
            "soul_md": p.get("soul_md", ""),
            "opening": p.get("opening", "Hello Titan, I've heard interesting things about you."),
            "fallback_responses": p.get("fallback_responses", [
                "That's interesting, tell me more.",
                "I hadn't thought about it that way.",
                "What else is on your mind?",
                "How does that make you feel?",
                "Can you elaborate on that?",
            ]),
        }

    familiar_keys = ["jake", "jane", "peter", "tom"]
    novel_keys = [p["key"] for p in novel_personas]

    # ── Build rotation schedule ──
    rotation_cycles = 4  # conversations per rotation
    rotation_duration = CYCLE_DURATION * rotation_cycles  # 20 min per rotation
    total_rotations = max(3, duration // rotation_duration)  # At least 3 (start, middle, end)

    schedule = _build_rotation_schedule(familiar_keys, novel_keys, total_rotations)

    logger.info("=" * 70)
    logger.info("ENDURANCE TEST V2 — DYNAMIC PERSONA ROTATION")
    logger.info("Duration: %s | Rotations: %d | Per rotation: %d concurrent",
                _format_duration(duration), len(schedule), PERSONAS_PER_ROTATION)
    logger.info("Familiar: %s", ", ".join(familiar_keys))
    logger.info("Novel: %s", ", ".join(novel_keys) if novel_keys else "(none generated)")
    for i, rot in enumerate(schedule):
        tag = "FAMILIAR" if i == 0 or i == len(schedule) - 1 else "MIXED"
        logger.info("  Rotation %d [%s]: %s", i + 1, tag, ", ".join(rot))
    logger.info("=" * 70)

    # ── Setup logging ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"v2_rotation_{timestamp}.jsonl"
    report_path = REPORT_DIR / f"v2_rotation_{timestamp}.md"

    start_time = time.time()
    all_agents: dict[str, PersonaAgent] = {}  # Reuse agents across rotations
    rotation_snapshots = []

    with open(log_path, "w") as log_file:
        # Log test metadata
        log_file.write(json.dumps({
            "type": "v2_test_start",
            "ts": _ts(),
            "duration": duration,
            "schedule": schedule,
            "familiar_keys": familiar_keys,
            "novel_keys": novel_keys,
            "novel_personas": [{"key": p["key"], "name": p["name"]} for p in novel_personas],
        }) + "\n")
        log_file.flush()

        # ── Run rotation schedule ──
        for rot_idx, rotation_keys in enumerate(schedule):
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            remaining = duration - elapsed
            rot_duration = min(rotation_duration, remaining)

            tag = "FAMILIAR" if rot_idx == 0 or rot_idx == len(schedule) - 1 else "MIXED"
            is_novel_rotation = any(k in novel_keys for k in rotation_keys)

            logger.info("─── Rotation %d/%d [%s] (%s) ───",
                        rot_idx + 1, len(schedule), tag, ", ".join(rotation_keys))

            # Snapshot consciousness BEFORE rotation
            pre_snapshot = await _snapshot_consciousness(api_base)

            log_file.write(json.dumps({
                "type": "rotation_start",
                "ts": _ts(),
                "rotation": rot_idx + 1,
                "tag": tag,
                "personas": rotation_keys,
                "has_novel": is_novel_rotation,
                "consciousness_before": pre_snapshot,
            }) + "\n")
            log_file.flush()

            # Create/reuse agents for this rotation
            active_agents = []
            for i, key in enumerate(rotation_keys):
                if key not in all_agents:
                    all_agents[key] = PersonaAgent(
                        persona_key=key,
                        api_base=api_base,
                        internal_key=internal_key,
                        llm_api_key=llm_api_key,
                        llm_base_url=llm_base_url,
                        llm_model=llm_model,
                        log_file=log_file,
                    )
                active_agents.append((all_agents[key], i * 10))  # 10s stagger

            # Run this rotation's personas concurrently
            tasks = []
            for agent, delay in active_agents:
                tasks.append(
                    asyncio.create_task(
                        agent.run_conversation(int(rot_duration), start_delay=delay),
                        name=f"v2_{agent.key}_rot{rot_idx}",
                    )
                )

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            # Snapshot consciousness AFTER rotation
            post_snapshot = await _snapshot_consciousness(api_base)

            rotation_snapshots.append({
                "rotation": rot_idx + 1,
                "tag": tag,
                "personas": rotation_keys,
                "has_novel": is_novel_rotation,
                "consciousness_before": pre_snapshot,
                "consciousness_after": post_snapshot,
            })

            log_file.write(json.dumps({
                "type": "rotation_end",
                "ts": _ts(),
                "rotation": rot_idx + 1,
                "consciousness_after": post_snapshot,
            }) + "\n")
            log_file.flush()

            # Log mood delta
            mood_before = pre_snapshot.get("mood", {}).get("score", 0)
            mood_after = post_snapshot.get("mood", {}).get("score", 0)
            logger.info(
                "  Rotation %d complete: mood %.3f → %.3f (Δ%+.3f) | %s",
                rot_idx + 1, mood_before, mood_after, mood_after - mood_before, tag,
            )

        # ── Final logging ──
        total_elapsed = time.time() - start_time
        log_file.write(json.dumps({
            "type": "v2_test_end",
            "ts": _ts(),
            "elapsed_s": round(total_elapsed, 1),
            "rotation_snapshots": rotation_snapshots,
        }) + "\n")

    # ── Generate V2 Report ──
    _generate_v2_report(
        all_agents, schedule, rotation_snapshots, novel_personas,
        duration, total_elapsed, report_path,
    )

    logger.info("=" * 70)
    logger.info("V2 ENDURANCE TEST COMPLETE")
    logger.info("Duration: %s | Rotations: %d", _format_duration(total_elapsed), len(rotation_snapshots))
    total = sum(a.prompts_sent for a in all_agents.values())
    ok = sum(a.successes for a in all_agents.values())
    fail = sum(a.failures for a in all_agents.values())
    logger.info("Total: %d prompts | %d ok | %d fail | %.1f%% success",
                total, ok, fail, ok / max(total, 1) * 100)
    for a in all_agents.values():
        s = a.get_stats()
        fam = "familiar" if a.key in familiar_keys else "novel"
        logger.info("  %s (%s) [%s]: %d turns, %.1fs avg, mode=%s",
                     s["name"], s["x_handle"], fam, s["prompts_sent"],
                     s["avg_latency_s"], s["primary_mode"])
    logger.info("Report: %s", report_path)
    logger.info("=" * 70)


def _generate_v2_report(
    agents: dict, schedule: list, snapshots: list,
    novel_personas: list, duration: int, elapsed: float,
    report_path: Path,
):
    """Generate markdown report for V2 rotation experiment."""
    familiar_keys = {"jake", "jane", "peter", "tom"}

    md = ["# Endurance Test V2 — Dynamic Persona Rotation Report\n"]
    md.append(f"**Date:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    md.append(f"**Duration:** {_format_duration(elapsed)} (configured: {_format_duration(duration)})")
    md.append(f"**Rotations:** {len(snapshots)}")
    md.append(f"**Max concurrent:** {PERSONAS_PER_ROTATION}\n")

    # Novel personas
    if novel_personas:
        md.append("## Generated Novel Personas\n")
        for p in novel_personas:
            md.append(f"- **{p['name']}** ({p['key']}): {p.get('soul_md', '')[:100]}...")
        md.append("")

    # Rotation timeline with consciousness
    md.append("## Rotation Timeline\n")
    md.append("| # | Tag | Personas | Mood Before | Mood After | Δ Mood | Consciousness Epoch |")
    md.append("|---|-----|----------|-------------|------------|--------|---------------------|")
    for snap in snapshots:
        mb = snap["consciousness_before"].get("mood", {}).get("score", 0)
        ma = snap["consciousness_after"].get("mood", {}).get("score", 0)
        delta = ma - mb
        c_before = snap["consciousness_before"].get("consciousness", {})
        epoch = c_before.get("epoch", "?") if c_before else "?"
        personas_str = ", ".join(snap["personas"])
        md.append(f"| {snap['rotation']} | {snap['tag']} | {personas_str} | {mb:.3f} | {ma:.3f} | {delta:+.3f} | {epoch} |")
    md.append("")

    # Per-persona stats
    md.append("## Per-Persona Summary\n")
    md.append("| Persona | Type | Turns | Success | Avg Lat | Mode |")
    md.append("|---------|------|-------|---------|---------|------|")
    for key, agent in sorted(agents.items()):
        s = agent.get_stats()
        ptype = "familiar" if key in familiar_keys else "novel"
        md.append(f"| {s['name']} | {ptype} | {s['prompts_sent']} | {s['successes']} | {s['avg_latency_s']:.1f}s | {s['primary_mode']} |")
    md.append("")

    # Familiarity vs novelty analysis
    md.append("## Familiarity vs Novelty Analysis\n")
    fam_moods = []
    novel_moods = []
    for snap in snapshots:
        delta = snap["consciousness_after"].get("mood", {}).get("score", 0) - snap["consciousness_before"].get("mood", {}).get("score", 0)
        if snap["has_novel"]:
            novel_moods.append(delta)
        else:
            fam_moods.append(delta)

    if fam_moods:
        md.append(f"- Familiar rotations avg mood Δ: **{sum(fam_moods)/len(fam_moods):+.4f}** ({len(fam_moods)} rotations)")
    if novel_moods:
        md.append(f"- Novel/mixed rotations avg mood Δ: **{sum(novel_moods)/len(novel_moods):+.4f}** ({len(novel_moods)} rotations)")
    md.append("")

    content = "\n".join(md)
    with open(report_path, "w") as f:
        f.write(content)

    # Also save JSON
    json_path = report_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "schedule": schedule,
            "snapshots": snapshots,
            "agents": {k: a.get_stats() for k, a in agents.items()},
            "novel_personas": novel_personas,
        }, f, indent=2)

    logger.info("Reports: %s + %s", report_path, json_path)


# ─── V3 Endurance Test ───────────────────────────────────────────────────

async def run_endurance_v3(duration: int):
    """
    V3 Endurance Test — Research-Heavy, Toned-Down Pacing.

    Changes from V2:
    - Slower pacing: 60-120s between turns (gives Titan processing time)
    - Research nudges every 3rd turn (was 5th) — encourages tool usage
    - V3 nudges: self-research, memory recall, metabolism check, sovereignty introspection
    - Max 2 concurrent personas (less pressure on Titan)
    - 2 familiar + 2 novel personas per run
    """
    cfg = _load_config()
    api_base = _get_api_base()
    internal_key = cfg.get("api", {}).get("internal_key", "")
    inference_cfg = cfg.get("inference", {})
    endurance_cfg = cfg.get("endurance", {})
    llm_api_key = inference_cfg.get("ollama_cloud_api_key", "")
    llm_base_url = inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1")
    # Personas use a lighter model to avoid 429 collisions with Titan's deepseek
    llm_model = endurance_cfg.get("persona_llm_model", "gemma3:4b")

    if not internal_key or not llm_api_key:
        logger.error("Missing internal_key or ollama_cloud_api_key in config.toml")
        return

    # Pre-flight checks
    logger.info("Pre-flight: testing Ollama Cloud API (%s)...", llm_model)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"},
                json={"model": llm_model, "messages": [{"role": "user", "content": "Say OK in one word."}], "max_tokens": 10},
            )
            if resp.status_code != 200:
                logger.error("Ollama Cloud pre-flight FAILED: HTTP %d", resp.status_code)
                return
            logger.info("Ollama Cloud: OK")
    except Exception as e:
        logger.error("Ollama Cloud pre-flight FAILED: %s", e)
        return

    logger.info("Pre-flight: testing Titan at %s ...", api_base)
    titan_ok = False
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{api_base}/health")
                if resp.status_code == 200:
                    logger.info("Titan: OK")
                    titan_ok = True
                    break
                logger.warning("Titan health check HTTP %d (attempt %d/3)", resp.status_code, attempt + 1)
        except Exception as e:
            logger.warning("Titan not reachable (attempt %d/3): %s", attempt + 1, e)
        if attempt < 2:
            await asyncio.sleep(10)
    if not titan_ok:
        logger.error("Titan pre-flight failed after 3 attempts")
        return

    # Generate 2 novel personas (less than V2, more focused)
    logger.info("Generating 2 novel personas via deepseek...")
    novel_personas = await _generate_new_personas(llm_api_key, llm_base_url, count=2)
    logger.info("Generated %d novel personas", len(novel_personas))

    for p in novel_personas:
        PERSONA_SOULS[p["key"]] = {
            "name": p["name"],
            "x_handle": p.get("x_handle", f"@{p['key']}"),
            "soul_md": p.get("soul_md", ""),
            "opening": p.get("opening", "Hello Titan, I've heard interesting things about you."),
            "fallback_responses": p.get("fallback_responses", [
                "That's interesting, tell me more.",
                "I hadn't thought about it that way.",
                "What else is on your mind?",
                "How does that make you feel?",
                "Can you elaborate on that?",
            ]),
        }

    familiar_keys = ["jake", "jane", "peter", "tom"]
    novel_keys = [p["key"] for p in novel_personas]
    per_rotation = 2  # Only 2 concurrent (lighter on Titan)

    # V3 rotation schedule: longer rotations (30 min each), fewer rotations
    rotation_duration = 1800  # 30 min per rotation
    total_rotations = max(3, duration // rotation_duration)
    schedule = _build_rotation_schedule(familiar_keys, novel_keys, total_rotations, per_rotation=per_rotation)

    logger.info("=" * 70)
    logger.info("ENDURANCE TEST V3 — RESEARCH-HEAVY, TONED-DOWN PACING")
    logger.info("Duration: %s | Rotations: %d | Per rotation: %d concurrent",
                _format_duration(duration), len(schedule), per_rotation)
    logger.info("Pacing: 60-120s between turns | Research nudge every 3rd turn")
    logger.info("Familiar: %s", ", ".join(familiar_keys))
    logger.info("Novel: %s", ", ".join(novel_keys) if novel_keys else "(none generated)")
    for i, rot in enumerate(schedule):
        tag = "FAMILIAR" if i == 0 or i == len(schedule) - 1 else "MIXED"
        logger.info("  Rotation %d [%s]: %s", i + 1, tag, ", ".join(rot))
    logger.info("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"v3_research_{timestamp}.jsonl"
    report_path = REPORT_DIR / f"v3_research_{timestamp}.md"

    start_time = time.time()
    all_agents: dict[str, PersonaAgent] = {}
    rotation_snapshots = []

    with open(log_path, "w") as log_file:
        log_file.write(json.dumps({
            "type": "v3_test_start",
            "ts": _ts(),
            "duration": duration,
            "schedule": schedule,
            "familiar_keys": familiar_keys,
            "novel_keys": novel_keys,
            "pacing": "60-120s",
            "nudge_interval": 3,
        }) + "\n")
        log_file.flush()

        for rot_idx, rotation_keys in enumerate(schedule):
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            remaining = duration - elapsed
            rot_duration = min(rotation_duration, remaining)

            tag = "FAMILIAR" if rot_idx == 0 or rot_idx == len(schedule) - 1 else "MIXED"

            logger.info("─── Rotation %d/%d [%s] (%s) ───",
                        rot_idx + 1, len(schedule), tag, ", ".join(rotation_keys))

            pre_snapshot = await _snapshot_consciousness(api_base)

            log_file.write(json.dumps({
                "type": "rotation_start",
                "ts": _ts(),
                "rotation": rot_idx + 1,
                "tag": tag,
                "personas": rotation_keys,
                "consciousness_before": pre_snapshot,
            }) + "\n")
            log_file.flush()

            active_agents = []
            for i, key in enumerate(rotation_keys):
                if key not in all_agents:
                    all_agents[key] = PersonaAgent(
                        persona_key=key,
                        api_base=api_base,
                        internal_key=internal_key,
                        llm_api_key=llm_api_key,
                        llm_base_url=llm_base_url,
                        llm_model=llm_model,
                        log_file=log_file,
                    )
                active_agents.append((all_agents[key], i * 15))  # 15s stagger

            tasks = []
            for agent, delay in active_agents:
                tasks.append(
                    asyncio.create_task(
                        _run_v3_conversation(agent, int(rot_duration), delay),
                        name=f"v3_{agent.key}_rot{rot_idx}",
                    )
                )

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            post_snapshot = await _snapshot_consciousness(api_base)

            rotation_snapshots.append({
                "rotation": rot_idx + 1,
                "tag": tag,
                "personas": rotation_keys,
                "has_novel": any(k in novel_keys for k in rotation_keys),
                "consciousness_before": pre_snapshot,
                "consciousness_after": post_snapshot,
            })

            log_file.write(json.dumps({
                "type": "rotation_end",
                "ts": _ts(),
                "rotation": rot_idx + 1,
                "consciousness_after": post_snapshot,
            }) + "\n")
            log_file.flush()

            mood_before = pre_snapshot.get("mood", {}).get("score", 0)
            mood_after = post_snapshot.get("mood", {}).get("score", 0)
            logger.info(
                "  Rotation %d complete: mood %.3f → %.3f (Δ%+.3f) | %s",
                rot_idx + 1, mood_before, mood_after, mood_after - mood_before, tag,
            )

        total_elapsed = time.time() - start_time
        log_file.write(json.dumps({
            "type": "v3_test_end",
            "ts": _ts(),
            "elapsed_s": round(total_elapsed, 1),
            "rotation_snapshots": rotation_snapshots,
        }) + "\n")

    _generate_v2_report(
        all_agents, schedule, rotation_snapshots, novel_personas,
        duration, total_elapsed, report_path,
    )

    logger.info("=" * 70)
    logger.info("V3 ENDURANCE TEST COMPLETE")
    logger.info("Duration: %s | Rotations: %d", _format_duration(total_elapsed), len(rotation_snapshots))
    total = sum(a.prompts_sent for a in all_agents.values())
    ok = sum(a.successes for a in all_agents.values())
    fail = sum(a.failures for a in all_agents.values())
    logger.info("Total: %d prompts | %d ok | %d fail | %.1f%% success",
                total, ok, fail, ok / max(total, 1) * 100)
    for a in all_agents.values():
        s = a.get_stats()
        fam = "familiar" if a.key in familiar_keys else "novel"
        logger.info("  %s (%s) [%s]: %d turns, %.1fs avg, mode=%s",
                     s["name"], s["x_handle"], fam, s["prompts_sent"],
                     s["avg_latency_s"], s["primary_mode"])
    logger.info("Report: %s", report_path)
    logger.info("=" * 70)


async def _run_v3_conversation(agent: 'PersonaAgent', duration: int, start_delay: float = 0.0):
    """
    V3 conversation runner — slower pacing + research nudges every 3rd turn.
    Replaces agent.run_conversation() for V3 mode.
    """
    if start_delay > 0:
        logger.info("[%s] Starting in %.0fs...", agent.key, start_delay)
        await asyncio.sleep(start_delay)

    logger.info(
        "[%s] %s (%s) starting V3 conversation | wallet: %s | session: %s",
        agent.key, agent.name, agent.x_handle, agent.wallet[:16] + "...", agent.session_id,
    )

    start_time = time.time()

    # Opening message
    logger.info("[%s] Sending opening message...", agent.key)
    result = await agent.send_to_titan(agent.opening)
    titan_reply = result["response"]
    agent._record_exchange(agent.opening, titan_reply, result)
    agent._log_entry({
        "type": "exchange", "turn": agent.prompts_sent,
        "persona_msg": agent.opening[:500], "titan_response": titan_reply[:500],
        "mode": result["mode"], "mood": result["mood"],
        "elapsed_s": result["elapsed_s"], "success": result["success"],
        "status_code": result["status_code"],
    })

    if result["success"]:
        logger.info("[%s] Turn %d | mode=%s mood=%s latency=%.1fs | %s",
                    agent.key, agent.prompts_sent, result["mode"], result["mood"],
                    result["elapsed_s"], titan_reply[:80])
    else:
        logger.warning("[%s] Turn %d FAILED (HTTP %d): %s",
                       agent.key, agent.prompts_sent, result["status_code"], titan_reply[:80])

    # Conversation loop with V3 pacing
    while time.time() - start_time < duration:
        # V3: longer waits (60-120s) to give Titan processing time
        wait = random.uniform(60, 120)
        remaining = duration - (time.time() - start_time)
        if remaining <= wait:
            break
        await asyncio.sleep(wait)

        # V3: research nudge every 3rd turn (more frequent tool usage)
        if agent.prompts_sent > 0 and agent.prompts_sent % 3 == 0:
            nudge_idx = (agent.prompts_sent // 3) % len(V3_RESEARCH_NUDGES)
            next_msg = V3_RESEARCH_NUDGES[nudge_idx]
            logger.info("[%s] V3 research nudge (turn %d): %s", agent.key, agent.prompts_sent, next_msg[:60])
        elif result["success"] and titan_reply:
            next_msg = await agent.generate_persona_response(titan_reply)
        else:
            next_msg = agent._get_fallback()

        result = await agent.send_to_titan(next_msg)
        titan_reply = result["response"]
        agent._record_exchange(next_msg, titan_reply, result)
        agent._log_entry({
            "type": "exchange", "turn": agent.prompts_sent,
            "persona_msg": next_msg[:500], "titan_response": titan_reply[:500],
            "mode": result["mode"], "mood": result["mood"],
            "elapsed_s": result["elapsed_s"], "success": result["success"],
            "status_code": result["status_code"],
        })

        if result["success"]:
            logger.info("[%s] Turn %d | mode=%s mood=%s latency=%.1fs | %s",
                        agent.key, agent.prompts_sent, result["mode"], result["mood"],
                        result["elapsed_s"], titan_reply[:80])
        else:
            logger.warning("[%s] Turn %d FAILED (HTTP %d): %s",
                           agent.key, agent.prompts_sent, result["status_code"], titan_reply[:80])

        if not result["success"]:
            await asyncio.sleep(20)

    # Goodbye
    goodbye = "hey, I gotta go now. it was great talking! take care and keep researching!"
    result = await agent.send_to_titan(goodbye)
    agent._record_exchange(goodbye, result["response"], result)
    agent._log_entry({
        "type": "goodbye", "turn": agent.prompts_sent,
        "persona_msg": goodbye, "titan_response": result["response"][:500],
        "mode": result["mode"], "mood": result["mood"],
        "elapsed_s": result["elapsed_s"], "success": result["success"],
    })

    elapsed = time.time() - start_time
    avg_lat = agent.total_latency / max(agent.prompts_sent, 1)
    logger.info(
        "[%s] DONE | %d turns | %d ok / %d fail | avg_lat=%.1fs | duration=%s",
        agent.key, agent.prompts_sent, agent.successes, agent.failures,
        avg_lat, _format_duration(elapsed),
    )

    await agent.close()


# ─── Dry-Run Verification ────────────────────────────────────────────────

async def dry_run_verify():
    """Quick verification that Ollama Cloud API and Titan /chat both respond."""
    cfg = _load_config()
    api_base = _get_api_base()
    internal_key = cfg.get("api", {}).get("internal_key", "")
    inference_cfg = cfg.get("inference", {})
    llm_api_key = inference_cfg.get("ollama_cloud_api_key", "")
    llm_base_url = inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1")

    checks_passed = 0
    checks_total = 2

    # Check 1: Ollama Cloud API
    logger.info("[DRY-RUN] Testing Ollama Cloud API with geek persona...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-v3.1:671b",
                    "messages": [
                        {"role": "system", "content": PERSONA_SOULS["jake"]["soul_md"]},
                        {"role": "user", "content": "Say hello as Jake in one sentence."},
                    ],
                    "temperature": 0.8,
                    "max_tokens": 100,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.info("[DRY-RUN] Ollama Cloud API: OK — %s", content.strip()[:100])
                checks_passed += 1
            else:
                logger.error("[DRY-RUN] Ollama Cloud API: HTTP %d — %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("[DRY-RUN] Ollama Cloud API: FAILED — %s", e)

    # Check 2: Titan /chat
    logger.info("[DRY-RUN] Testing Titan /chat endpoint...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{api_base}/chat",
                headers={
                    "X-Titan-Internal-Key": internal_key,
                    "X-Titan-User-Id": "dry_run_test",
                },
                json={
                    "message": "ping — dry run verification",
                    "session_id": f"dry_run_{int(time.time())}",
                    "user_id": "dry_run_test",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    "[DRY-RUN] Titan /chat: OK — mode=%s mood=%s response=%s",
                    data.get("mode", "?"), data.get("mood", "?"),
                    data.get("response", "")[:80],
                )
                checks_passed += 1
            else:
                logger.error("[DRY-RUN] Titan /chat: HTTP %d — %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("[DRY-RUN] Titan /chat: FAILED — %s", e)

    logger.info("[DRY-RUN] Result: %d/%d checks passed", checks_passed, checks_total)
    return checks_passed == checks_total


# ─── CLI ──────────────────────────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser(
        description="Titan 4-Persona Endurance Test — Venice AI-powered conversational agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/persona_endurance.py --duration 1800           # 30-min test, all personas\n"
            "  python scripts/persona_endurance.py --duration 3600           # 1-hour test\n"
            "  python scripts/persona_endurance.py --personas jake,tom       # Specific personas\n"
            "  python scripts/persona_endurance.py --verify                  # Check social graph\n"
            "  python scripts/persona_endurance.py --dry-run                 # Verify Venice + Titan APIs\n"
            "\n"
            "Personas:\n"
            "  jake  — Jake Morrison, AI/ML engineer (@jakebuildsAI)\n"
            "  jane  — Jane Chen, new mother (@jane_and_baby_leo)\n"
            "  peter — Peter Kowalski, world traveler (@peter_summits)\n"
            "  tom   — Tom Nakamura, quantum physics PhD (@quantumtom_mit)\n"
        ),
    )
    parser.add_argument(
        "--duration", type=int, default=1800,
        help="Test duration in seconds (default: 1800 = 30 minutes)",
    )
    parser.add_argument(
        "--personas", type=str, default="jake,jane,peter,tom",
        help="Comma-separated list of personas to run (default: all four)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify social graph without running a test",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick verification of Ollama Cloud API and Titan /chat connectivity",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="V2 mode: dynamic persona rotation with LLM-generated novel personas",
    )
    parser.add_argument(
        "--v3", action="store_true",
        help="V3 mode: research-heavy, toned-down pacing, tool usage nudges",
    )
    args = parser.parse_args()

    if args.verify:
        await run_verify_only()
        return

    if args.dry_run:
        ok = await dry_run_verify()
        sys.exit(0 if ok else 1)

    # Parse persona list
    valid_keys = set(PERSONA_SOULS.keys())
    requested = [k.strip().lower() for k in args.personas.split(",")]
    persona_keys = [k for k in requested if k in valid_keys]
    invalid = [k for k in requested if k not in valid_keys]
    if invalid:
        logger.warning("Unknown personas ignored: %s (valid: %s)", invalid, list(valid_keys))
    if not persona_keys:
        logger.error("No valid personas specified. Valid: %s", list(valid_keys))
        sys.exit(1)

    if args.v3:
        await run_endurance_v3(args.duration)
    elif args.v2:
        await run_endurance_v2(args.duration)
    else:
        await run_endurance(args.duration, persona_keys)


def main():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n  Persona endurance test interrupted.")
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
