# Docker Setup for Federated Learning Systems

This project supports Docker deployment for both **timeseries/regression** and **CAE (Convolutional Autoencoder)** FL systems. All Docker-related files live in the **`docker/`** folder.

**Important:** Run all `docker-compose` commands from the **project root** (parent of `docker/`), using `-f docker/...` so that the build context is the project root and volumes point to the correct paths.

**Contents of `docker/`:** `Dockerfile`, `Dockerfile.cae`, `Dockerfile.acfl`,
`docker-compose.yml`, `docker-compose.cae.yml`, `docker-compose.acfl.yml`,
`entrypoint_acfl.sh`, `DOCKER.md`.

---

## Quick Start

### Timeseries/Regression FL System

From the **project root**:

```bash
# Build and run (ensure data/ contains GPS/IMU CSVs – see Timeseries System Details)
docker-compose -f docker/docker-compose.yml up --build
```

After the first build, run `docker-compose -f docker/docker-compose.yml up` (or `up -d`) without `--build` to start containers.

This starts:
- 1 FL server (port 8080)
- 3 FL clients (GPS/IMU timeseries; data from `data/` mounted at `/app/data`)

### CAE (Convolutional Autoencoder) FL System

From the **project root**:

```bash
docker-compose -f docker/docker-compose.cae.yml up --build
```

This starts:
- 1 FL server (port 5544)
- 3 FL clients (clean/noisy image pairs)

### ACFL (packet classification + poisoning defense)

From the **project root** (needs stock `data/train/packets/packet_client_00{0,1,2}.csv`):

```bash
docker compose -f docker/docker-compose.acfl.yml up --build
```

This starts:
- 1 ACFL server (port **8090**) with Layers 1–3
- 3 FL clients on `config_classification_drone` (TinyPacketCNN)

Optional Byzantine poisoning on `client_2`:

```bash
ACFL_ATTACK_TYPE=byzantine ACFL_ATTACK_CLIENTS=client_2 \
  docker compose -f docker/docker-compose.acfl.yml up --build
```

---

## Docker Files (in `docker/`)

### Timeseries System
- **`docker/Dockerfile`** - Base image for timeseries/regression FL
- **`docker/docker-compose.yml`** - Compose file for timeseries system

### CAE System
- **`docker/Dockerfile.cae`** - Base image for CAE FL
- **`docker/docker-compose.cae.yml`** - Compose file for CAE system

### ACFL System
- **`docker/Dockerfile.acfl`** - Base image for ACFL packet FL
- **`docker/docker-compose.acfl.yml`** - Compose file for ACFL system
- **`docker/entrypoint_acfl.sh`** - Server/client role switcher

Build context for all is the **project root** (`context: ..`). A **`.dockerignore`** at the project root keeps the context small (excludes `.git`, `__pycache__`, checkpoints, plots, `data/`, etc.). Dockerfiles use **CPU-only PyTorch** and **multi-stage builds** to reduce image size and disk use.

---

## Disk use

Images are kept smaller by:
- **CPU-only PyTorch** – installed from `https://download.pytorch.org/whl/cpu` (smaller than the default wheel).
- **Multi-stage builds** – dependencies are installed in a builder stage; only runtime artifacts are copied into the final image (no `build-essential` in the final layer).
- **`.dockerignore`** – excludes cache, outputs, and large dirs from the build context.
- **`requirements-docker.txt`** – used in Docker only (torch/torchvision come from the CPU index); `requirements.txt` remains for local installs.

---

## CAE System Details

### Prerequisites

Ensure your data directories exist:
```bash
mkdir -p data/payload/images/clean
mkdir -p data/payload/images/noise
# Place your clean/noisy image pairs in these directories
```

### Environment Variables

**Server:**
- `FL_ROLE=server`
- `FL_SERVER_HOST=0.0.0.0` (default)
- `FL_SERVER_PORT=5544` (default)
- `FL_NUM_CLIENTS=3` (default)
- `FL_MIN_CLIENTS=2` (default)
- `FL_AGGREGATION=fedavg` (default)

**Client:**
- `FL_ROLE=client`
- `FL_CLIENT_ID=client_1` (or client_2, client_3)
- `FL_SERVER_URL=http://fl-server-cae:5544`
- `FL_CLEAN_DIR=/app/data/payload/images/clean`
- `FL_NOISY_DIR=/app/data/payload/images/noise`
- `FL_NUM_ROUNDS=10` (default)
- `FL_LEARNING_RATE=0.001` (default)
- `FL_BATCH_SIZE=32` (default)
- `FL_EPOCHS_PER_ROUND=1` (default)

### Volumes

Paths are relative to the **project root** (e.g. `../fl-payload/checkpoints` in the compose file resolves to project root’s `fl-payload/checkpoints` when using `-f docker/docker-compose.cae.yml` from project root).

**Server:**
- `../fl-payload/checkpoints` → `/app/checkpoints` - Checkpoints directory
- `../fl-payload/plots` → `/app/plots` - Training loss plots

**Clients:**
- `../data/payload/images/clean` → `/app/data/payload/images/clean` (read-only)
- `../data/payload/images/noise` → `/app/data/payload/images/noise` (read-only)

### Running CAE System

From the **project root**:

1. **Build and start all services:**
   ```bash
   docker-compose -f docker/docker-compose.cae.yml up --build
   ```

2. **Start in detached mode:**
   ```bash
   docker-compose -f docker/docker-compose.cae.yml up -d --build
   ```

3. **View logs:**
   ```bash
   docker-compose -f docker/docker-compose.cae.yml logs -f
   ```

4. **Stop services:**
   ```bash
   docker-compose -f docker/docker-compose.cae.yml down
   ```

5. **Stop and remove volumes:**
   ```bash
   docker-compose -f docker/docker-compose.cae.yml down -v
   ```

---

## Timeseries System Details

### Prerequisites

- **Data:** Place GPS and IMU CSV files under `data/` (e.g. `data/waypoint_injection/` or `data/train/gps-imu/`). The compose file mounts `../data` → `/app/data` in the container.
- **Paths:** Client env vars `FL_GPS_FILE` and `FL_IMU_FILE` must be paths **inside the container**, e.g. `/app/data/waypoint_injection/your_gps.csv` and `/app/data/waypoint_injection/your_imu.csv`. The default in `docker-compose.yml` uses `waypoint_injection/mission_2_wp_23_attack_add_wp_5_alt_0005_*.csv`. If your files live elsewhere under `data/`, edit the compose file or override the env vars.

### Environment Variables

**Server:** `FL_ROLE=server`, `FL_SERVER_HOST`, `FL_SERVER_PORT=8080`, `FL_NUM_CLIENTS=3`, `FL_MIN_CLIENTS=2`, `FL_AGGREGATION=fedavg`, `FL_INPUT_SIZE=12`.

**Clients:** `FL_ROLE=client`, `FL_CLIENT_ID=client_1|client_2|client_3`, `FL_SERVER_URL=http://fl-server:8080`, `FL_GPS_FILE`, `FL_IMU_FILE` (paths under `/app/data`), plus optional `FL_WINDOW_SIZE`, `FL_OVERLAP`, etc.

### Volumes

- **Server:** `../fl-time-series/checkpoints` → `/app/checkpoints`
- **Clients:** `../data` → `/app/data:ro`

### Running

From the **project root**:

```bash
# Foreground (logs in terminal)
docker-compose -f docker/docker-compose.yml up

# Detached
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop
docker-compose -f docker/docker-compose.yml down
```

---

## ACFL System Details

### Prerequisites

Stock packet CSVs under the project root:

```bash
ls data/train/packets/packet_client_000.csv \
   data/train/packets/packet_client_001.csv \
   data/train/packets/packet_client_002.csv
```

For the paper SITL agent–mission layout, mount
`data/sitl_dataset/fl/packets` and set `FL_DATA_DIR` / drop
`FL_PACKET_FILE` so each client resolves files via `--data-dir`.

### Environment Variables

**Server:**

| Variable | Default | Meaning |
|----------|---------|---------|
| `FL_ROLE` | `server` | |
| `FL_SERVER_HOST` | `0.0.0.0` | Bind address |
| `FL_SERVER_PORT` | `8090` | HTTP port |
| `FL_NUM_CLIENTS` / `FL_MIN_CLIENTS` | `3` / `2` | Swarm size / quorum |
| `FL_CONFIG_MODULE` | `config_classification_drone` | TinyPacketCNN profile |
| `FL_LEARNING_MODE` | `classification` | |
| `FL_MAX_ROUNDS` | `12` | Caps client local rounds |
| `ACFL_MALICIOUS_CLIENTS` | `client_2` | Ground truth for trust metrics |
| `ACFL_NO_LAYER1` / `2` / `3` | `0` | Set `1` to disable a layer |
| `ACFL_PROBE_FILE` | _(empty)_ | Optional Layer‑1 probe `.pt` |

**Clients:**

| Variable | Default | Meaning |
|----------|---------|---------|
| `FL_ROLE` | `client` | |
| `FL_CLIENT_ID` | `client_1`… | Must match attack list IDs |
| `FL_SERVER_URL` | `http://acfl-server:8090` | |
| `FL_DATA_DIR` | `/app/data/train/packets` | Mounted host `data/` |
| `FL_PACKET_FILE` | per-client CSV | Explicit packet path |
| `FL_CLIENT_STAGGER_SEC` | `5` | Delay × (client index − 1) |
| `ACFL_ATTACK_CLIENTS` | _(empty)_ | e.g. `client_2` |
| `ACFL_ATTACK_TYPE` | _(empty)_ | `byzantine`, `gradient_scaling`, `backdoor`, … |
| `ACFL_ATTACK_SCALE` / `ACFL_ATTACK_NOISE` | `10` / `1` | Attack knobs |

### Volumes

- **Server:** `../fl-time-series/checkpoints` → `/app/checkpoints` (trust scores + round reports)
- **Clients:** `../data` → `/app/data:ro`

### Running

From the **project root**:

```bash
# Benign ACFL
docker compose -f docker/docker-compose.acfl.yml up --build

# Detached + logs
docker compose -f docker/docker-compose.acfl.yml up -d --build
docker compose -f docker/docker-compose.acfl.yml logs -f

# Poisoned (Byzantine client_2)
ACFL_ATTACK_TYPE=byzantine ACFL_ATTACK_CLIENTS=client_2 \
  docker compose -f docker/docker-compose.acfl.yml up --build

# Layer-1 only
ACFL_NO_LAYER2=1 ACFL_NO_LAYER3=1 \
  docker compose -f docker/docker-compose.acfl.yml up --build

# Stop
docker compose -f docker/docker-compose.acfl.yml down
```

Build the image alone:

```bash
docker build -f docker/Dockerfile.acfl -t fl-acfl:latest .
```

---

## Troubleshooting

### Port Conflicts

- **CAE system** uses port **5544** (default)
- **Timeseries system** uses port **8080** (default)
- **ACFL system** uses port **8090** (default)
- Change ports in the corresponding `docker/docker-compose*.yml` if needed

### Data Not Found

**Timeseries (GPS/IMU):**
- Error like `No such file or directory: '/app/data/...gps.csv'` means the path in `FL_GPS_FILE` / `FL_IMU_FILE` does not exist inside the container. The host folder `data/` is mounted at `/app/data`, so use paths like `/app/data/waypoint_injection/filename_gps.csv`. Ensure the files exist under your project’s `data/` directory and that the compose env vars match (including subfolders, e.g. `waypoint_injection/`).

**CAE (images):**
- Ensure image directories exist at project root: `data/payload/images/clean` and `data/payload/images/noise`
- Run `docker-compose` from **project root** with `-f docker/docker-compose.cae.yml` so volume paths resolve correctly
- Verify file permissions

### Model Loading Errors

- Ensure `fl-payload/models/` directory contains `cae_model.py` and `cae_model_small.py`
- Check that `config.py` in `fl-payload/` has correct `MODEL_PATH` and `MODEL_CLASS`

### Network Issues

- Both systems use separate Docker networks (`fl-network` and `fl-cae-network`)
- Clients connect to server via service name: `http://fl-server-cae:5544`
- **CAE:** The server has a healthcheck (TCP port 5544). Clients use `condition: service_healthy` so they start only after the server is listening, avoiding "Cannot connect to host" on registration.

---

## Development

### Building Individual Images

From the **project root** (build context must be project root so `COPY requirements-docker.txt`, `COPY fl-time-series/` or `COPY fl-payload/` / `COPY acfl/` work):

```bash
# CAE system
docker build -f docker/Dockerfile.cae -t fl-cae:latest .

# Timeseries system
docker build -f docker/Dockerfile -t fl-timeseries:latest .

# ACFL system
docker build -f docker/Dockerfile.acfl -t fl-acfl:latest .
```

### Running Individual Containers

From the **project root** (so `$(pwd)` is the project root):

```bash
# CAE server
docker run -d \
  --name fl-server-cae \
  -p 5544:5544 \
  -e FL_ROLE=server \
  -e FL_NUM_CLIENTS=3 \
  -v $(pwd)/fl-payload/checkpoints:/app/checkpoints \
  fl-cae:latest

# CAE client
docker run -d \
  --name fl-client-cae-1 \
  -e FL_ROLE=client \
  -e FL_CLIENT_ID=client_1 \
  -e FL_SERVER_URL=http://fl-server-cae:5544 \
  -v $(pwd)/data/payload/images:/app/data/payload/images:ro \
  --link fl-server-cae \
  fl-cae:latest
```

---

## Notes

- **CAE system** requires OpenGL libraries (`libgl1`, `libglib2.0-0`) for image processing
- **Checkpoints** are saved to `fl-payload/checkpoints/` at project root (mounted into the container)
- **Server** aggregates when at least `MIN_CLIENTS` (default: 2) have submitted
- **Clients** use staggered starts (`CLIENT_STAGGER_SEC=10`) to avoid simultaneous requests
- Both systems can run simultaneously on different ports/networks
