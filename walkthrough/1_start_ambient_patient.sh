#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./1_start_ambient_patient.sh <SERVER_IP>

Starts the current branch's voice app:
  - turn-server
  - app-server-healthcare-assistant
  - voice-agents-webrtc-python-app
  - voice-agents-webrtc-ui-app

Example:
  ./1_start_ambient_patient.sh 160.211.47.251
EOF
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local attempts="${3:-24}"
  local sleep_seconds="${4:-5}"

  for i in $(seq 1 "$attempts"); do
    if curl -fsS -m 5 "$url" >/dev/null 2>&1; then
      echo "$name is ready."
      return 0
    fi

    echo "Waiting for $name ($i/$attempts)..."
    sleep "$sleep_seconds"
  done

  echo "Error: $name did not become ready: $url"
  return 1
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

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

SERVER_IP="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
AGENT_COMPOSE_FILE="$REPO_ROOT/agent/docker-compose.yaml"
ACE_DIR="$REPO_ROOT/ace-controller-voice-interface"
ACE_COMPOSE_FILE="$ACE_DIR/docker-compose.yml"
ACE_ENV_FILE="$ACE_DIR/ace_controller.local.env"
ACE_ENV_TEMPLATE="$ACE_DIR/ace_controller.env"
RUNTIME_CONFIG_FILE="$ACE_DIR/configs/config_riva_runtime.yaml"
RUNTIME_CONFIG_REL="./configs/config_riva_runtime.yaml"
RUNTIME_CONFIG_TEMPLATE="$ACE_DIR/configs/config_riva_hybrid.yaml"

for file in "$AGENT_COMPOSE_FILE" "$ACE_COMPOSE_FILE" "$RUNTIME_CONFIG_TEMPLATE"; do
  if [[ ! -f "$file" ]]; then
    echo "Error: required file not found: $file"
    exit 1
  fi
done

if [[ ! -f "$ACE_ENV_FILE" ]]; then
  if [[ ! -f "$ACE_ENV_TEMPLATE" ]]; then
    echo "Error: missing both $ACE_ENV_FILE and $ACE_ENV_TEMPLATE"
    exit 1
  fi

  cp "$ACE_ENV_TEMPLATE" "$ACE_ENV_FILE"
  echo "Created $ACE_ENV_FILE from template."
fi

python3 - "$RUNTIME_CONFIG_TEMPLATE" "$RUNTIME_CONFIG_FILE" "$SERVER_IP" <<'PY'
from pathlib import Path
import sys

template = Path(sys.argv[1]).read_text()
runtime_file = Path(sys.argv[2])
server_ip = sys.argv[3]

runtime_file.write_text(template.replace("${SERVER_IP}", server_ip))
PY

python3 - "$ACE_ENV_FILE" "$RUNTIME_CONFIG_REL" "$SERVER_IP" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
config_path = sys.argv[2]
server_ip = sys.argv[3]

lines = env_path.read_text().splitlines()
updates = {
    "CONFIG_PATH": config_path,
    "TURN_SERVER_URL": f"turn:{server_ip}:3478",
}

seen = set()
result = []

for line in lines:
    if "=" not in line or line.lstrip().startswith("#"):
        result.append(line)
        continue

    key, _, _ = line.partition("=")
    if key in updates:
      result.append(f"{key}={updates[key]}")
      seen.add(key)
    else:
      result.append(line)

for key, value in updates.items():
    if key not in seen:
        result.append(f"{key}={value}")

env_path.write_text("\n".join(result) + "\n")
PY

export USERID="$(id -u):$(id -g)"
export SERVER_IP

prepare_healthcare_network

echo "[1/3] Restarting turn-server..."
docker rm -f turn-server >/dev/null 2>&1 || true
docker run -d \
  --name turn-server \
  --network host \
  instrumentisto/coturn \
  -n \
  --log-file=stdout \
  --external-ip="$SERVER_IP" \
  --listening-ip=0.0.0.0 \
  --listening-port=3478 \
  --lt-cred-mech \
  --fingerprint \
  --user=admin:admin \
  --no-multicast-peers \
  --realm=tokkio.realm.org \
  --min-port=50000 \
  --max-port=52000 >/dev/null

echo "[2/3] Starting app-server-healthcare-assistant..."
docker compose -f "$AGENT_COMPOSE_FILE" up -d app-server
wait_for_url "http://localhost:8081/health" "app-server-healthcare-assistant"

echo "[3/3] Starting voice-agents-webrtc-python-app and voice-agents-webrtc-ui-app..."
docker compose -f "$ACE_COMPOSE_FILE" --profile ace-controller up -d python-app ui-app
wait_for_url "http://localhost:7860/get_prompt" "voice-agents-webrtc-python-app"

echo
echo "Voice UI: http://localhost:4400"
echo "Runtime config: $RUNTIME_CONFIG_FILE"
