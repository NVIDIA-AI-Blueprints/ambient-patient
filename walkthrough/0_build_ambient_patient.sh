#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./0_build_ambient_patient.sh

Builds the images needed for the current branch's voice app:
  - app-server-healthcare-assistant
  - voice-agents-webrtc-python-app
  - voice-agents-webrtc-ui-app
EOF
}

prepare_healthcare_network() {
  local network_name="healthcare-agent"
  local expected_label="healthcare-agent-network"

  if ! docker network inspect "$network_name" >/dev/null 2>&1; then
    echo "No existing $network_name network found."
    echo "Docker Compose will create it during startup."
    return 0
  fi

  local actual_label
  actual_label="$(docker network inspect -f '{{ index .Labels "com.docker.compose.network" }}' "$network_name" 2>/dev/null || true)"

  if [[ "$actual_label" == "$expected_label" ]]; then
    echo "Found Compose-managed $network_name network."
    return 0
  fi

  local attached_containers
  attached_containers="$(docker network inspect -f '{{ len .Containers }}' "$network_name" 2>/dev/null || echo 0)"

  if [[ "$attached_containers" != "0" ]]; then
    echo "Error: $network_name exists with the wrong Compose label and still has attached containers."
    echo "Stop the running stack first, then remove the network and retry."
    exit 1
  fi

  echo "Removing stale unmanaged $network_name network so Compose can recreate it correctly..."
  docker network rm "$network_name" >/dev/null
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
AGENT_COMPOSE_FILE="$REPO_ROOT/agent/docker-compose.yaml"
ACE_COMPOSE_FILE="$REPO_ROOT/ace-controller-voice-interface/docker-compose.yml"

for file in "$AGENT_COMPOSE_FILE" "$ACE_COMPOSE_FILE"; do
  if [[ ! -f "$file" ]]; then
    echo "Error: required file not found: $file"
    exit 1
  fi
done

export USERID="$(id -u):$(id -g)"

echo "[0/2] Checking Docker network..."
prepare_healthcare_network

echo "[1/2] Building app-server-healthcare-assistant..."
docker compose -f "$AGENT_COMPOSE_FILE" build app-server

echo "[2/2] Building voice-agents-webrtc-python-app and voice-agents-webrtc-ui-app..."
docker compose -f "$ACE_COMPOSE_FILE" --profile ace-controller build python-app ui-app

echo
echo "Build complete."
echo "Next: ./1_start_ambient_patient.sh <SERVER_IP>"
