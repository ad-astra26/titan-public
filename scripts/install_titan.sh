#!/bin/bash
# install_titan.sh
# Installation script for Titan Memory Architecture Plugin for OpenClaw 2026

set -e

echo "==============================================="
echo "  Installing Titan Memory Architecture Plugin  "
echo "==============================================="

# 1. Ensure Python dependencies and venv
if [ ! -d ".venv" ]; then
    echo "Creating Python Virtual Environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Titan Python Dependencies..."
pip install -e .

# 2. Check for Solana Toolchain
if ! command -v solana &> /dev/null; then
    echo "⚠️ Solana CLI not found. Please install Agave Solana CLI v2.x+"
    echo "Run: sh -c \"\$(curl -sSfL https://release.anza.xyz/v2.2.14/install)\""
    exit 1
fi

# 3. Check for Rust and Anchor (Optional for plugin users, required for devs)
if ! command -v anchor &> /dev/null; then
    echo "ℹ️ Anchor CLI not found. (Not required just to run the agent, but required for ZK-Vault compilation)."
fi

# 4. Set up environment file
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cp .env.example .env
    echo "✅ .env created. Please configure your RPC URLs and Inference API keys."
fi

# 5. Initialize Cognee directories
mkdir -p ./cognee_data

echo "==============================================="
echo "  Titan-Class Plugin Installed Successfully!   "
echo "==============================================="
echo ""

# ============================================================
# Phase 2: Stealth-Sage Infrastructure (V2.0 Step 5)
# ============================================================
echo ""
echo "==============================================="
echo "  Phase 2: Stealth-Sage Infrastructure         "
echo "==============================================="

# 6. Docker — install if absent
if ! command -v docker &>/dev/null; then
    echo "Installing Docker (required for SearXNG)..."
    sudo apt-get update -qq
    sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
    # Add current user to docker group (takes effect on next login)
    sudo usermod -aG docker "$USER"
    echo "✅ Docker installed. You may need to log out and back in for group changes."
else
    echo "ℹ️ Docker already installed ($(docker --version | head -1))."
fi

# 7. SearXNG — idempotent Docker container
if ! docker ps -a --format '{{.Names}}' | grep -q "^searxng$"; then
    echo "Starting SearXNG search engine (Docker)..."
    mkdir -p "$(pwd)/searxng"

    # Write a minimal SearXNG settings.yml to allow the JSON format API
    cat > "$(pwd)/searxng/settings.yml" <<'SEARXNG_CONFIG'
use_default_settings: true
server:
  secret_key: "titan_sage_secret_key_change_me"
  base_url: http://localhost:8080/
search:
  safe_search: 0
  default_lang: en
  formats:
    - html
    - json
SEARXNG_CONFIG

    docker run -d \
        --name searxng \
        --restart=unless-stopped \
        -p 8080:8080 \
        -v "$(pwd)/searxng:/etc/searxng" \
        searxng/searxng
    echo "✅ SearXNG container started on http://localhost:8080"
    echo "   Waiting 5 seconds for container initialization..."
    sleep 5
else
    echo "ℹ️ SearXNG container already exists."
    if ! docker ps --format '{{.Names}}' | grep -q "^searxng$"; then
        echo "   Restarting stopped SearXNG container..."
        docker start searxng
    fi
fi

# 8. Crawl4AI — stealth web scraper with Playwright/Chromium
echo "Installing Crawl4AI and Playwright Chromium..."
pip install "crawl4ai>=0.4.0"
crawl4ai-setup || true   # sets up Playwright browsers
python -m playwright install --with-deps chromium
echo "✅ Crawl4AI + Playwright/Chromium installed."

# 9. Unstructured — document parsing (PDF, DOCX, PPTX, XLSX)
echo "Installing Unstructured document parsing dependencies..."
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr
pip install "unstructured[pdf,docx,pptx,xlsx]"
echo "✅ Unstructured + system OCR dependencies installed."

# 10. Ollama — local LLM for distillation
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed."
else
    echo "ℹ️ Ollama already installed ($(ollama --version 2>/dev/null || echo 'version unknown'))."
fi

# Pull phi3:mini model (~2.3GB) — this is required for research distillation
echo "Pulling Ollama model: phi3:mini (~2.3GB, please wait)..."
ollama pull phi3:mini
echo "✅ phi3:mini model ready."

# 11. Dependency Audit
echo ""
echo "--- Dependency Audit ---"

# Crawl4AI doctor
python -c "
import asyncio
async def check():
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        print('✅ crawl4ai: importable')
    except ImportError as e:
        print(f'⚠️ crawl4ai: import failed — {e}')
asyncio.run(check())
"

# Unstructured
python -c "
try:
    from unstructured.partition.auto import partition
    print('✅ unstructured: importable')
except ImportError as e:
    print(f'⚠️ unstructured: import failed — {e}')
"

# Ollama liveness
curl -sf http://localhost:11434/api/tags | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    models = [m['name'] for m in d.get('models', [])]
    print(f'✅ Ollama: reachable — models: {models}')
except Exception as e:
    print(f'⚠️ Ollama: could not parse response — {e}')
" 2>/dev/null || echo "⚠️ Ollama: not reachable on :11434 — ensure ollama serve is running."

# SearXNG liveness
curl -sf "http://localhost:8080/search?q=test&format=json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'✅ SearXNG: reachable — {len(d.get(\"results\", []))} results for test query.')
except Exception as e:
    print(f'⚠️ SearXNG: could not parse response — {e}')
" 2>/dev/null || echo "⚠️ SearXNG: not reachable on :8080 — Docker container may still be initializing."

# 12. Advisory (non-fatal) — optional external service credentials
echo ""
echo "--- Optional Service Configuration Advisory ---"
python3 - <<'ADVISORY'
import sys, os
sys.path.insert(0, os.getcwd())
try:
    try:
        import tomllib
    except ImportError:
        import toml as tomllib
    with open("titan_plugin/config.toml", "rb") as f:
        cfg = tomllib.load(f)
    ss = cfg.get("stealth_sage", {})
    missing = []
    if not ss.get("twitterapi_io_key", "").strip():
        missing.append("twitterapi_io_key  (X/Twitter search will be disabled)")
    if not ss.get("webshare_rotating_url", "").strip():
        missing.append("webshare_rotating_url  (proxy will be disabled; VPS IP used directly)")
    if missing:
        print("ℹ️ Advisory: The following optional stealth_sage credentials are not configured:")
        for m in missing:
            print(f"   • {m}")
        print("   Edit titan_plugin/config.toml to enable these features.")
        print("   The Titan will operate gracefully without them.")
    else:
        print("✅ All optional stealth_sage credentials are configured.")
except Exception as e:
    print(f"ℹ️ Could not read config.toml for advisory check: {e}")
ADVISORY

# Create required data directories
mkdir -p ./data/logs ./data/history /tmp/titan_sage_docs

echo ""
echo "==============================================="
echo "  Phase 2 Complete: Stealth-Sage is Ready!    "
echo "==============================================="
echo ""
echo "Next Steps:"
echo "1. Configure .env with PREMIUM_RPC_URL and INFERENCE_API_KEY."
echo "2. (Optional) Add twitterapi_io_key and webshare_rotating_url to titan_plugin/config.toml."
echo "3. Generate your sovereign wallet: 'solana-keygen new --outfile ./authority.json'"
echo "4. Restart your OpenClaw agent instance."
echo "5. Verify with: python -m pytest tests/test_sage_v2.py -v"
