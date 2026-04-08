#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo ""
echo "=== MCP - Ollama - LangChain Agent ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------

if [ ! -d ".venv" ]; then
  echo "[..] Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[..] Installing dependencies..."
pip install -r requirements.txt
echo "[ok] Dependencies ready"

# ---------------------------------------------------------------------------
# 2. Ollama
# ---------------------------------------------------------------------------

if ! command -v ollama &>/dev/null; then
  echo "[ERROR] Ollama is not installed."
  echo "        Run: curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi

if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "[ok] Ollama already running"
else
  echo "[..] Starting Ollama..."
  ollama serve &>/dev/null &
  sleep 3
fi

for model in llama3.2 nomic-embed-text; do
  if ollama list 2>/dev/null | grep -q "$model"; then
    echo "[ok] $model already pulled"
  else
    echo "[..] Pulling $model (first run, this may take a while)..."
    ollama pull "$model"
  fi
done

# ---------------------------------------------------------------------------
# 3. Docker check
# ---------------------------------------------------------------------------

SKIP_DOCKER=false
if ! command -v docker &>/dev/null; then
  echo "[WARN] Docker not found -- Open WebUI will be skipped."
  SKIP_DOCKER=true
elif ! docker info &>/dev/null 2>&1; then
  echo "[WARN] Docker daemon not running -- Open WebUI will be skipped."
  SKIP_DOCKER=true
fi

# ---------------------------------------------------------------------------
# 4. Start services
# ---------------------------------------------------------------------------

echo ""
echo "[..] Starting MCP server on :8001"
python -m mcp_server.main &
MCP_PID=$!

sleep 2

echo "[..] Starting Agent API on :8000"
python -m agent.main &
AGENT_PID=$!

sleep 2

if [ "$SKIP_DOCKER" = false ]; then
  echo "[..] Starting Open WebUI on :3000"
  docker run -d --rm \
    --name open-webui \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
    -e OPENAI_API_KEY=local \
    ghcr.io/open-webui/open-webui:main
fi

echo ""
echo "All services running"
[ "$SKIP_DOCKER" = false ] && echo "  Open WebUI  -> http://localhost:3000"
echo "  Agent API   -> http://localhost:8000/docs"
echo "  MCP server  -> http://localhost:8001/health"
echo ""
echo "Press Ctrl+C to stop"

cleanup() {
  echo ""
  echo "Stopping..."
  kill "$MCP_PID" "$AGENT_PID" 2>/dev/null
  [ "$SKIP_DOCKER" = false ] && docker stop open-webui 2>/dev/null
  exit 0
}

trap cleanup INT TERM
wait
