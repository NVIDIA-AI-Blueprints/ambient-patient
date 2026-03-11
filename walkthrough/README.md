# Walkthrough

This README describes the current manual startup flow for the voice app.

## 1. Update the Brev server IP

When the Brev server IP changes, update it in these files.

### `ace-controller-voice-interface/configs/config_riva_hybrid.yaml`

Set the remote Riva endpoints:

- `RivaASRService.server` -> `<BREV_IP>:50052`
- `RivaTTSService.server` -> `<BREV_IP>:50051`

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
TURN_SERVER_URL=turn:160.211.45.53:3478
```

Notes:

- `CONFIG_PATH` should point to `./configs/config_riva_hybrid.yaml`
- update `TURN_SERVER_URL` when the IP changes

## 2. Start the app

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

## 3. Open the UI

Open:

```text
http://localhost:4400
```

## 4. Stop the app

```bash
docker compose --profile ace-controller -f ace-controller-voice-interface/docker-compose.yml down
docker compose -f agent/docker-compose.yaml down
docker rm -f turn-server 2>/dev/null || true
```
