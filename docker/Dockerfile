FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY models/ models/
COPY aggregation.py .

# Create directories
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

# For server
ENV FL_SERVER_HOST=0.0.0.0
ENV FL_SERVER_PORT=8080
ENV FL_NUM_CLIENTS=3
ENV FL_MIN_CLIENTS=2
ENV FL_AGGREGATION=fedavg
ENV FL_INPUT_SIZE=12

# Expose server port
EXPOSE 8080

# Run the application
CMD ["sh", "-c", "\
    if [ \"$FL_ROLE\" = \"server\" ]; then \
        python fl_server.py \
            --host $FL_SERVER_HOST \
            --port $FL_SERVER_PORT \
            --num-clients $FL_NUM_CLIENTS \
            --min-clients $FL_MIN_CLIENTS \
            --aggregation $FL_AGGREGATION \
            --input-size $FL_INPUT_SIZE \
            --hidden-size $FL_HIDDEN_SIZE \
            --num-layers $FL_NUM_LAYERS; \
    else \
        python fl_client.py \
            --client-id $FL_CLIENT_ID \
            --gps-file $FL_GPS_FILE \
            --imu-file $FL_IMU_FILE \
            --server-url $FL_SERVER_URL \
            --window-size $FL_WINDOW_SIZE \
            --overlap $FL_OVERLAP \
            --windows-per-round $FL_WINDOWS_PER_ROUND \
            --lr $FL_LEARNING_RATE \
            --hidden-size $FL_HIDDEN_SIZE \
            --num-layers $FL_NUM_LAYERS; \
    fi"]
