# -----------------------------------------------------------------------------
# Stage 1: install dependencies (CPU-only PyTorch to save ~500MB+ per image)
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU-only PyTorch (smaller than default CUDA/cpu+cuda wheel)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# -----------------------------------------------------------------------------
# Stage 2: runtime image (no build-essential, smaller)
# -----------------------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Copy only installed packages from builder (no build tools)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy timeseries FL application code
COPY fl-time-series/*.py .
COPY fl-time-series/models/ models/

RUN mkdir -p checkpoints data

# Environment variables (can be overridden)
ENV FL_ROLE=client
ENV FL_CLIENT_ID=client_1
ENV FL_SERVER_URL=http://fl-server:8080
ENV FL_GPS_FILE=/app/data/gps.csv
ENV FL_IMU_FILE=/app/data/imu.csv
ENV FL_WINDOW_SIZE=50
ENV FL_OVERLAP=25
ENV FL_WINDOWS_PER_ROUND=10
ENV FL_LEARNING_RATE=0.001
ENV FL_HIDDEN_SIZE=64
ENV FL_NUM_LAYERS=2
ENV FL_SERVER_HOST=0.0.0.0
ENV FL_SERVER_PORT=8080
ENV FL_NUM_CLIENTS=3
ENV FL_MIN_CLIENTS=2
ENV FL_AGGREGATION=fedavg
ENV FL_INPUT_SIZE=12
EXPOSE 8080

ENV FL_CONFIG_MODULE=config_regression
CMD ["sh", "-c", "\
    if [ \"$FL_ROLE\" = \"server\" ]; then \
        python fl_server_time_series.py \
            --host $FL_SERVER_HOST \
            --port $FL_SERVER_PORT \
            --num-clients $FL_NUM_CLIENTS \
            --min-clients $FL_MIN_CLIENTS \
            --aggregation $FL_AGGREGATION \
            --learning-mode regression \
            --input-size $FL_INPUT_SIZE; \
    else \
        python fl_client_time_series.py \
            --client-id $FL_CLIENT_ID \
            --gps-file $FL_GPS_FILE \
            --imu-file $FL_IMU_FILE \
            --server-url $FL_SERVER_URL; \
    fi"]
