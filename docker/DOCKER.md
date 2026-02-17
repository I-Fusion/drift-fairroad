# Docker Setup for Federated Learning Systems

This project supports Docker deployment for both **timeseries/regression** and **CAE (Convolutional Autoencoder)** FL systems. All Docker-related files live in the **`docker/`** folder.

**Important:** Run all `docker-compose` commands from the **project root** (parent of `docker/`), using `-f docker/...` so that the build context is the project root and volumes point to the correct paths.

---

## Quick Start

### Timeseries/Regression FL System

From the **project root**:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

This starts:
- 1 FL server (port 8080)
- 3 FL clients (GPS/IMU timeseries data)

### CAE (Convolutional Autoencoder) FL System

From the **project root**:

```bash
docker-compose -f docker/docker-compose.cae.yml up --build
```

This starts:
- 1 FL server (port 5544)
- 3 FL clients (clean/noisy image pairs)

---

## Docker Files (in `docker/`)

### Timeseries System
- **`docker/Dockerfile`** - Base image for timeseries/regression FL
- **`docker/docker-compose.yml`** - Compose file for timeseries system

### CAE System
- **`docker/Dockerfile.cae`** - Base image for CAE FL
- **`docker/docker-compose.cae.yml`** - Compose file for CAE system

Build context for both is the **project root** (`context: ..`), so `COPY requirements.txt` and `COPY fl-payload/` resolve correctly. Dockerfiles are referenced as `dockerfile: docker/Dockerfile` and `dockerfile: docker/Dockerfile.cae`.

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

See `docker/docker-compose.yml` for timeseries/regression FL system configuration. Run from project root with `docker-compose -f docker/docker-compose.yml up --build`.

---

## Troubleshooting

### Port Conflicts

- **CAE system** uses port **5544** (default)
- **Timeseries system** uses port **8080** (default)
- Change ports in `docker/docker-compose.cae.yml` or `docker/docker-compose.yml` if needed

### Data Not Found

- Ensure image directories exist at project root: `data/payload/images/clean` and `data/payload/images/noise`
- Run `docker-compose` from **project root** with `-f docker/docker-compose.cae.yml` so volume paths resolve correctly
- Verify file permissions

### Model Loading Errors

- Ensure `fl-payload/models/` directory contains `cae_model.py` and `cae_model_small.py`
- Check that `config.py` in `fl-payload/` has correct `MODEL_PATH` and `MODEL_CLASS`

### Network Issues

- Both systems use separate Docker networks (`fl-network` and `fl-cae-network`)
- Clients connect to server via service name: `http://fl-server-cae:5544`

---

## Development

### Building Individual Images

From the **project root** (build context must be project root so `COPY requirements.txt` and `COPY fl-payload/` work):

```bash
# CAE system
docker build -f docker/Dockerfile.cae -t fl-cae:latest .

# Timeseries system
docker build -f docker/Dockerfile -t fl-timeseries:latest .
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

- **CAE system** requires OpenGL libraries (`libgl1-mesa-glx`, `libglib2.0-0`) for image processing
- **Checkpoints** are saved to `fl-payload/checkpoints/` at project root (mounted into the container)
- **Server** aggregates when at least `MIN_CLIENTS` (default: 2) have submitted
- **Clients** use staggered starts (`CLIENT_STAGGER_SEC=10`) to avoid simultaneous requests
- Both systems can run simultaneously on different ports/networks
