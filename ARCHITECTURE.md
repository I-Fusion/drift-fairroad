# Federated Learning System - Detailed Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Communication Protocol](#communication-protocol)
5. [Aggregation Strategies](#aggregation-strategies)
6. [Security Considerations](#security-considerations)
7. [Scalability](#scalability)

## System Overview

This Federated Learning (FL) system implements a distributed machine learning architecture where multiple clients collaboratively train a global model without sharing their raw data. The system is specifically designed for GPS time series positioning data.

### Key Principles

- **Privacy-Preserving**: Raw data never leaves client devices
- **Decentralized Training**: Computation distributed across clients
- **Centralized Aggregation**: Server coordinates and aggregates updates
- **Synchronous Updates**: All clients participate in each round

## Component Architecture

### 1. FL Server

**Responsibilities:**
- Maintain the global model state
- Coordinate training rounds
- Aggregate client updates
- Manage client registration and synchronization
- Save model checkpoints

**Key Components:**

```
┌─────────────────────────────────────────────────────┐
│                    FL Server                        │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ Global Model │  │  Aggregator  │  │ HTTP API │  │
│  │  (LSTM)      │  │  (FedAvg)    │  │ (aiohttp)│  │
│  └──────────────┘  └──────────────┘  └──────────┘  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  Client      │  │  Checkpoint  │                │
│  │  Registry    │  │  Manager     │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

**API Endpoints:**

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/register` | POST | Client registration | `{client_id}` | `{status, current_round}` |
| `/get_model` | POST | Download global model | `{client_id}` | Model weights (binary) |
| `/submit_update` | POST | Submit local updates | Model weights + headers | `{status, round}` |
| `/status` | GET | Server status | None | Server state info |

### 2. FL Client

**Responsibilities:**
- Load and prepare local data
- Download global model from server
- Perform local training
- Compute and submit model updates
- Track local metrics

**Key Components:**

```
┌─────────────────────────────────────────────────────┐
│                    FL Client                        │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ Local Model  │  │   Training   │  │   Data   │  │
│  │  (LSTM)      │  │   Engine     │  │  Loader  │  │
│  └──────────────┘  └──────────────┘  └──────────┘  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  HTTP Client │  │   Metrics    │                │
│  │  (aiohttp)   │  │   Tracker    │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

**Training Loop:**

```python
for round in range(num_rounds):
    # 1. Get global model
    weights = download_model_from_server()
    model.set_weights(weights)

    # 2. Train locally
    for epoch in range(local_epochs):
        for batch in data_loader:
            loss = train_step(batch)

    # 3. Submit update
    submit_weights_to_server(model.get_weights(), loss)
```

### 3. Neural Network Model

**Architecture: LSTM-based Time Series Predictor**

```
Input: GPS Sequence (batch_size, seq_len=10, features=2)
   │
   ├─> LSTM Layer 1 (hidden_size=64)
   │      └─> Dropout (0.2)
   │
   ├─> LSTM Layer 2 (hidden_size=64)
   │      └─> Dropout (0.2)
   │
   ├─> Take last hidden state
   │
   ├─> Fully Connected (64 -> 32)
   │      └─> ReLU
   │      └─> Dropout (0.2)
   │
   └─> Fully Connected (32 -> 2)

Output: Next GPS Coordinates (batch_size, 2)
```

**Model Parameters:**
- Total parameters: ~67,000
- Input features: 2 (latitude, longitude)
- Sequence length: 10 time steps
- Output: 2 (next lat, lon)

## Data Flow

### Round N Training Flow

```
┌──────────┐                    ┌──────────┐
│ Client 1 │                    │  Server  │
└────┬─────┘                    └────┬─────┘
     │                               │
     │ 1. POST /register             │
     │ ─────────────────────────────>│
     │                               │
     │ 2. {status: registered}       │
     │ <─────────────────────────────│
     │                               │
     │ 3. POST /get_model            │
     │ ─────────────────────────────>│
     │                               │
     │ 4. global_weights (binary)    │
     │ <─────────────────────────────│
     │                               │
     ├─ 5. Local Training            │
     │     (5 epochs on local data)  │
     │                               │
     │ 6. POST /submit_update        │
     │    + local_weights            │
     │    + num_samples              │
     │    + loss                     │
     │ ─────────────────────────────>│
     │                               ├─ 7. Aggregate
     │                               │    when all
     │                               │    clients ready
     │ 8. {status: aggregated}       │
     │ <─────────────────────────────│
     │                               │
     └─ Round N+1 starts             └─ Save checkpoint
```

### Data Privacy

```
┌─────────────────────────────────────────────────────┐
│             Client's Private Data                   │
│  ┌────────────────────────────────────────────┐     │
│  │ GPS Trajectories (Raw Data)                │     │
│  │ - Latitude, Longitude sequences            │     │
│  │ - Time series data                         │     │
│  │                                             │     │
│  │ ❌ NEVER leaves the client                 │     │
│  └────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
                      │
                      │ Local Training
                      ▼
        ┌──────────────────────────┐
        │   Model Gradients/       │
        │   Weights Update         │
        │                          │
        │ ✅ Shared with server    │
        └──────────────────────────┘
```

## Communication Protocol

### Message Format

**Client → Server (Submit Update):**

```http
POST /submit_update HTTP/1.1
Host: server:8080
Content-Type: application/octet-stream
X-Client-ID: client_1
X-Num-Samples: 1000
X-Loss: 0.0234

[Binary pickled model weights]
```

**Server → Client (Model Response):**

```http
HTTP/1.1 200 OK
Content-Type: application/octet-stream
X-Round: 5

[Binary pickled global model weights]
```

### Serialization

- **Format**: Python pickle (binary)
- **Compression**: None (can add gzip for production)
- **Alternative**: Protocol Buffers, FlatBuffers for cross-platform

### Network Considerations

- **Bandwidth**: ~260KB per model transfer (64 hidden units)
- **Latency**: Async I/O minimizes blocking
- **Retries**: Client implements exponential backoff

## Aggregation Strategies

### 1. FedAvg (Federated Averaging)

**Algorithm:**

```
global_weights = Σ(client_weights_i) / num_clients

for each parameter θ:
    θ_global = (θ_1 + θ_2 + ... + θ_n) / n
```

**Pros:**
- Simple and efficient
- Works well with balanced data
- Low computational overhead

**Cons:**
- Doesn't account for data size differences
- Assumes IID data distribution

### 2. FedAvgM (Federated Averaging with Momentum)

**Algorithm:**

```
# Standard aggregation
θ_avg = Σ(θ_i) / n

# Apply server-side momentum
v_t = β * v_{t-1} + (1 - β) * θ_avg
θ_global = v_t

where β = 0.9 (momentum parameter)
```

**Pros:**
- Smoother convergence
- Reduces oscillations
- Better for non-IID data

**Cons:**
- Requires momentum state tracking
- Slightly more complex

### 3. Weighted Average

**Algorithm:**

```
total_samples = Σ(num_samples_i)

for each parameter θ:
    θ_global = Σ(θ_i * num_samples_i / total_samples)
```

**Pros:**
- Accounts for dataset size differences
- Fair representation of all data
- Better for imbalanced scenarios

**Cons:**
- Clients must report sample counts
- Can be dominated by large datasets

### Comparison

| Strategy | Data Balance | Convergence | Overhead | Best For |
|----------|-------------|-------------|----------|----------|
| FedAvg | Assumes balanced | Fast | Low | Homogeneous data |
| FedAvgM | Handles some imbalance | Smooth | Medium | Non-IID data |
| Weighted | Handles imbalance | Fair | Medium | Heterogeneous sizes |

## Security Considerations

### Current Implementation

- **Authentication**: Client ID based (simple)
- **Authorization**: Registered clients only
- **Data Protection**: Model weights only (no raw data)
- **Transport**: HTTP (plaintext)

### Production Recommendations

```
┌────────────────────────────────────────────┐
│          Security Enhancements             │
├────────────────────────────────────────────┤
│ 1. TLS/SSL encryption (HTTPS)              │
│ 2. JWT-based authentication                │
│ 3. Secure aggregation (encrypted weights)  │
│ 4. Differential privacy (DP-SGD)           │
│ 5. Byzantine-robust aggregation            │
│ 6. Client verification (signatures)        │
│ 7. Rate limiting                           │
│ 8. Audit logging                           │
└────────────────────────────────────────────┘
```

### Threat Model

| Threat | Risk | Mitigation |
|--------|------|------------|
| Model inversion | Medium | Differential privacy |
| Poisoning attacks | High | Byzantine-robust aggregation |
| Eavesdropping | High | TLS encryption |
| Impersonation | Medium | Strong authentication |
| DoS | Medium | Rate limiting, quotas |

## Scalability

### Current Scale

- **Clients**: 3 (configurable)
- **Model Size**: ~260KB
- **Data per Client**: 1000 samples
- **Round Time**: ~10-30 seconds

### Scaling Considerations

**Horizontal Scaling (More Clients):**

```
Clients    | Aggregation Time | Network Load
-----------|------------------|-------------
10         | ~1s             | ~2.6 MB
100        | ~10s            | ~26 MB
1000       | ~100s           | ~260 MB
```

**Optimization Strategies:**

1. **Asynchronous FL**: Don't wait for all clients
2. **Client Sampling**: Select subset per round
3. **Model Compression**: Quantization, pruning
4. **Federated Dropout**: Reduce model size
5. **Hierarchical FL**: Multi-tier aggregation

### Architecture for 1000+ Clients

```
                    ┌────────────┐
                    │   Server   │
                    │  (Global)  │
                    └─────┬──────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼───┐        ┌────▼───┐       ┌────▼───┐
    │ Edge  │        │ Edge   │       │ Edge   │
    │Server1│        │Server2 │       │Server3 │
    └───┬───┘        └────┬───┘       └────┬───┘
        │                 │                 │
    ┌───┴────┐       ┌────┴────┐      ┌────┴────┐
    │Clients │       │Clients  │      │Clients  │
    │ 1-100  │       │ 101-200 │      │ 201-300 │
    └────────┘       └─────────┘      └─────────┘
```

## Performance Metrics

### Training Metrics

- **Round Time**: Time for one FL round
- **Convergence Rate**: Rounds to target accuracy
- **Communication Cost**: Data transferred per round
- **Client Utilization**: % of clients participating

### System Metrics

- **Throughput**: Updates aggregated per second
- **Latency**: Time from update to aggregation
- **Availability**: Server uptime percentage
- **Scalability**: Max concurrent clients

### Example Monitoring

```python
# Server-side metrics
{
    "round": 10,
    "clients_registered": 3,
    "clients_ready": 3,
    "avg_loss": 0.0234,
    "aggregation_time": 0.123,
    "round_duration": 45.67
}
```

## Container Architecture

### Docker Architecture

```
┌──────────────────────────────────────────────────┐
│              Docker Compose Network              │
│                  (fl-network)                    │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────┐                             │
│  │   fl-server    │  Port 8080 (exposed)        │
│  │  [Container]   │                             │
│  └────────┬───────┘                             │
│           │                                      │
│  ┌────────┼────────────────────────────┐        │
│  │        │                            │        │
│  │  ┌─────▼──────┐  ┌──────────┐  ┌──────────┐ │
│  │  │ fl-client-1│  │fl-client-2│  │fl-client-3│ │
│  │  │ [Container]│  │[Container]│  │[Container]│ │
│  │  └────────────┘  └──────────┘  └──────────┘ │
│  │                                              │
└──┴──────────────────────────────────────────────┘
```

**Container Configuration:**

Each container runs the same image but with different environment variables:

```bash
# Server
FL_ROLE=server
FL_NUM_CLIENTS=3

# Client
FL_ROLE=client
FL_CLIENT_ID=client_1
FL_SERVER_HOST=fl-server
```

This allows a single codebase to determine its role at runtime based on configuration.

## Future Enhancements

1. **Dynamic Client Selection**: Server selects subset of clients per round
2. **Adaptive Aggregation**: Choose strategy based on data distribution
3. **Model Personalization**: Per-client model fine-tuning
4. **Cross-Silo FL**: Enterprise-grade multi-organization setup
5. **Streaming Data**: Real-time GPS data ingestion
6. **Model Versioning**: Track and rollback global models
7. **A/B Testing**: Test multiple aggregation strategies
8. **Advanced Privacy**: Secure multi-party computation (SMPC)

## References

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492)
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)
