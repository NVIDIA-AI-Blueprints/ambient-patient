#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./2_stop_ambient_patient.sh

Stops the current branch's voice app:
  - voice-agents-webrtc-ui-app
  - voice-agents-webrtc-python-app
  - app-server-healthcare-assistant
  - turn-server
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
AGENT_COMPOSE_FILE="$REPO_ROOT/agent/docker-compose.yaml"
ACE_COMPOSE_FILE="$REPO_ROOT/ace-controller-voice-interface/docker-compose.yml"
RUNTIME_CONFIG_FILE="$REPO_ROOT/ace-controller-voice-interface/configs/config_riva_runtime.yaml"

for file in "$AGENT_COMPOSE_FILE" "$ACE_COMPOSE_FILE"; do
  if [[ ! -f "$file" ]]; then
    echo "Error: required file not found: $file"
    exit 1
  fi
done

echo "[1/3] Stopping voice services..."
docker compose -f "$ACE_COMPOSE_FILE" --profile ace-controller stop ui-app python-app || true

echo "[2/3] Stopping app-server-healthcare-assistant..."
docker compose -f "$AGENT_COMPOSE_FILE" stop app-server || true

echo "[3/3] Removing turn-server..."
docker rm -f turn-server >/dev/null 2>&1 || true

rm -f "$RUNTIME_CONFIG_FILE"

echo
echo "Voice app stopped."
