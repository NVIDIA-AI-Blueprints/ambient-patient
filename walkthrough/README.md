# Walkthrough

This README describes the current manual startup flow for the voice app.

## 1. Create local .env files

First, copy and rename the example environment variable files to create your local overrides.

From the repo root, run:

```bash
# For the ACE Controller (voice UI frontend)
cp ace-controller-voice-interface/ace_controller.env ace-controller-voice-interface/ace_controller.local.env

# For the agent server
cp agent/vars.local.env.example agent/vars.local.env
```

You now have two local files to customize:

- `ace-controller-voice-interface/ace_controller.local.env`
- `agent/vars.local.env`

**Edit the values in these files to configure API keys and endpoints as needed for your environment.** Your local changes are ignored by git.
  
See the comments in those files for guidance on individual settings.

## 2. Update the Brev server IP

When the Brev server IP changes, update it in these files.

### `ace-controller-voice-interface/configs/config_riva_hybrid.yaml`

Set the remote Riva endpoints:

- `RivaASRService.server` -> `<BREV_SERVER_IP>:50052`
- `RivaTTSService.server` -> `<BREV_SERVER_IP>:50051`

Example:

```yaml
RivaASRService:
    server: "160.211.45.53:50052"

RivaTTSService:
    server: "160.211.45.53:50051"
```

### `ace-controller-voice-interface/ace_controller.local.env`

Make sure these values are set:

```env
CONFIG_PATH=./configs/config_riva_hybrid.yaml
TURN_SERVER_URL=turn:<BREV_SERVER_IP>:3478
```

## 3. Create Tunnel for neo4j

Neo4j runs inside the remote Brev instance and its HTTP (7474) and Bolt (7687) ports are **not exposed publicly** by Brev for security reasons. To interact with Neo4j from your local machine (for example, with Neo4j Browser or the agent), you need to create local port-forwards to those ports.

Use the following commands in a separate terminal to forward the necessary ports:

`brev port-forward mmse-l40 -p 7474:7474`
`brev port-forward mmse-l40 -p 7687:7687`

This will make the Neo4j service accessible locally at `localhost:7474` (HTTP) and `localhost:7687` (Bolt/RPC).

## 4. Start the app

Run these commands from the repo root:

```bash
docker rm -f turn-server 2>/dev/null || true

docker run -d \
  --name turn-server \
  --network host \
  instrumentisto/coturn \
  -n \
  --log-file=stdout \
  --external-ip="160.211.45.53" \
  --listening-ip=0.0.0.0 \
  --listening-port=3478 \
  --lt-cred-mech \
  --fingerprint \
  --user=admin:admin \
  --no-multicast-peers \
  --realm=tokkio.realm.org \
  --min-port=50000 \
  --max-port=52000

docker compose -f agent/docker-compose.yaml up --build -d app-server

docker compose --profile ace-controller -f ace-controller-voice-interface/docker-compose.yml up --build -d
```

## 5. Open the UI

Open:

```text
http://localhost:4400
```

## 6. Stop the app

```bash
docker compose --profile ace-controller -f ace-controller-voice-interface/docker-compose.yml down
docker compose -f agent/docker-compose.yaml down
docker rm -f turn-server 2>/dev/null || true
```

## 7. Langfuse

You can access langfuse at <BREV_SERVER_IP>:3000