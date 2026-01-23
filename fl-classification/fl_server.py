"""
Federated Learning Server with Packet Support

Coordinates training and aggregates model updates.
Compatible with both fl_client.py and fl_client_with_packets.py.
Handles both supervised learning (classification) and time series prediction (regression).
"""
import asyncio
import pickle
import torch
from aiohttp import web
import logging
import importlib
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Dict, Set

from aggregation import FederatedAggregator
import config as config
from config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class FLServerClassification:
    """Federated Learning Server for the Classification Task."""

    def __init__(
        self,
        host: str,
        port: int,
        num_clients: int,
        min_clients: int,
        aggregation_strategy: str,
        output_size: Optional[int] = None,
        learning_mode: Optional[str] = None,
        input_size: Optional[int] = None
    ):
        """
        Initialize FL Server.
        
        Args:
            host: Server host address
            port: Server port
            num_clients: Expected number of clients
            min_clients: Minimum clients needed to start training
            aggregation_strategy: Aggregation strategy ('fedavg', 'fedavgm', 'weighted')
            output_size: Model output size (None = use config default)
            learning_mode: 'classification' or 'regression' (None = auto-detect from config)
        """
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.min_clients = min_clients

        # Determine learning mode
        if learning_mode is None:
            # Try to infer from config
            if hasattr(config, 'USE_LABELS') and config.USE_LABELS:
                self.learning_mode = 'classification'
            elif hasattr(config, 'PACKET_FILE') and config.PACKET_FILE:
                self.learning_mode = 'classification'
            else:
                self.learning_mode = 'regression'
        else:
            self.learning_mode = learning_mode

        # Determine input size
        if input_size is None:
            # Try to get from config
            if hasattr(config, 'INPUT_SIZE'):
                input_size = config.INPUT_SIZE
            else:
                # Calculate from features
                gps_features = getattr(config, 'GPS_FEATURES', [])
                imu_features = getattr(config, 'IMU_FEATURES', [])
                packet_features = getattr(config, 'PACKET_FEATURES', [])
                input_size = len(gps_features) + len(imu_features) + len(packet_features)
                if input_size == 0:
                    input_size = 12  # Default fallback
        
        self.input_size = input_size
        
        # Determine output size
        if output_size is None:
            if self.learning_mode == 'classification':
                # Binary classification: output_size = 1
                output_size = 1
            else:
                # Regression: use config default or use input_size
                if hasattr(config, 'MODEL_CONFIG') and 'output_size' in config.MODEL_CONFIG:
                    output_size = config.MODEL_CONFIG['output_size']
                else:
                    output_size = self.input_size  # Default to input_size for regression

        self.output_size = output_size
        logger.info(f"Learning mode: {self.learning_mode}, Output size: {self.output_size}")
        #print(f"[SERVER DEBUG] Initialization: learning_mode={self.learning_mode}, input_size={self.input_size}, output_size={self.output_size}")

        # Load model dynamically from config
        logger.info(f"Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}")
        #print(f"[SERVER DEBUG] Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}")
        model_module = importlib.import_module(config.MODEL_PATH)
        model_class = getattr(model_module, config.MODEL_CLASS)

        # Update model config with determined output size
        model_config = config.MODEL_CONFIG.copy()
        model_config['output_size'] = self.output_size
        
        # Ensure input_size is set (use the determined input_size)
        model_config['input_size'] = self.input_size

        # Initialize model
        self.model = model_class(**model_config)
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        logger.info(f"Model config: input_size={model_config.get('input_size')}, output_size={self.output_size}")
        #print(f"[SERVER DEBUG] Model initialized: {self.model.count_parameters()} parameters, input_size={model_config.get('input_size')}, output_size={self.output_size}")

        # Aggregator
        self.aggregator = FederatedAggregator(strategy=aggregation_strategy)
        #print(f"[SERVER DEBUG] Aggregator initialized: strategy={aggregation_strategy}")

        # State
        self.current_round = 0
        self.client_weights = {}
        self.client_samples = {}
        self.client_losses = {}
        self.client_modes = {}  # Track learning mode per client
        self.registered_clients = set()
        self.ready_clients = set()
        self.round_lock = asyncio.Lock()

        # Loss tracking for plotting
        self.loss_history = {}  # {client_id: [losses per round]}
        self.round_losses = []  # Average loss per round

        # Flag to track if training has started
        self.training_started = False
        
        # Track current operation for status monitoring
        self.current_operation = "idle"
        self.operation_start_time = None
        self.operation_details = {}

        logger.info(f"Server initialized with {aggregation_strategy} aggregation")
        #print(f"[SERVER DEBUG] Server fully initialized: {aggregation_strategy} aggregation, expecting {num_clients} clients, min={min_clients}")

    async def handle_register(self, request: web.Request) -> web.Response:
        """Handle client registration."""
        #print(f"[SERVER DEBUG] ===== REGISTER REQUEST RECEIVED =====")
        data = await request.json()
        client_id = data.get("client_id")
        client_mode = data.get("learning_mode", None)  # Optional: client can report its mode
        #print(f"[SERVER DEBUG] Register: client_id={client_id}, client_mode={client_mode}")

        async with self.round_lock:
            if client_id not in self.registered_clients:
                self.registered_clients.add(client_id)
                self.loss_history[client_id] = []  # Initialize loss history
                if client_mode:
                    self.client_modes[client_id] = client_mode
                logger.info(f"{client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
                #print(f"[SERVER DEBUG] {client_id} registered: {len(self.registered_clients)}/{self.num_clients} clients")
                if client_mode:
                    logger.info(f"  Client mode: {client_mode}")
                    #print(f"[SERVER DEBUG]   Client mode: {client_mode}")
            else:
                #print(f"[SERVER DEBUG] {client_id} already registered (duplicate request)")

            # Check if minimum clients reached and training should start
            #print(f"[SERVER DEBUG] Training status: started={self.training_started}, registered={len(self.registered_clients)}, min_required={self.min_clients}")
            if not self.training_started and len(self.registered_clients) >= self.min_clients:
                self.training_started = True
                logger.info(f"✓ Minimum {self.min_clients} clients registered. Training can start!")
                #print(f"[SERVER DEBUG] *** TRAINING CAN NOW START! *** (min clients reached)")

        return web.json_response({
            "status": "registered",
            "current_round": self.current_round,
            "can_start": self.training_started,
            "message": f"{len(self.registered_clients)}/{self.num_clients} clients registered",
            "learning_mode": self.learning_mode,
            "output_size": self.output_size
        })

    async def handle_get_model(self, request: web.Request) -> web.Response:
        """Send global model to client."""
        import time
        #print(f"[SERVER DEBUG] ===== GET_MODEL REQUEST RECEIVED =====")
        data = await request.json()
        client_id = data.get("client_id")
        requested_round = data.get("round", 0)
        #print(f"[SERVER DEBUG] Get model: client_id={client_id}, requested_round={requested_round}, current_round={self.current_round}")

        if client_id not in self.registered_clients:
            #print(f"[SERVER DEBUG] ERROR: {client_id} not registered! Registered clients: {self.registered_clients}")
            return web.json_response({"error": "Not registered"}, status=403)

        # Update operation status
        self.current_operation = f"sending_model_to_{client_id}"
        self.operation_start_time = time.time()
        self.operation_details = {"client_id": client_id, "requested_round": requested_round}
        #print(f"[SERVER DEBUG] Operation set: {self.current_operation}")

        # Wait for training to start (minimum clients)
        #print(f"[SERVER DEBUG] Checking training_started: {self.training_started}")
        wait_count = 0
        while not self.training_started:
            await asyncio.sleep(0.5)
            wait_count += 1
            if wait_count % 10 == 0:
                print(f"[SERVER DEBUG] Still waiting for training to start... (waited {wait_count * 0.5}s)")
        if wait_count > 0:
            print(f"[SERVER DEBUG] Training started! (waited {wait_count * 0.5}s)")

        # Handle round request logic:
        # - If requesting round 0, send current model immediately
        # - If requesting round N > 0, wait for round N to be aggregated (current_round >= N)
        # - But if we're still at round 0 and client requests round 1, send current model
        #   (this handles the case where first round updates were rejected)
        if requested_round == 0:
            # Round 0: send current model immediately
            logger.info(f"{client_id} requesting round 0, sending current model (round {self.current_round})")
            #print(f"[SERVER DEBUG] Round 0 requested -> sending immediately (current_round={self.current_round})")
        elif requested_round == 1 and self.current_round == 0:
            # Special case: client requests round 1 but we're still at round 0
            # This can happen if first round updates were rejected
            # Send current model (round 0) - client can use it for round 2
            logger.info(f"{client_id} requesting round 1, but server is at round 0. Sending current model.")
            #print(f"[SERVER DEBUG] Special case: requested_round=1, current_round=0 -> sending current model")
        else:
            # Wait for the requested round to be ready (aggregation completed)
            max_wait = 120  # Maximum 2 minutes wait
            wait_count = 0
            logger.info(f"{client_id} requesting round {requested_round}, current round: {self.current_round}, waiting...")
            #print(f"[SERVER DEBUG] Waiting for round {requested_round} (current={self.current_round}, max_wait={max_wait}s)...")
            while self.current_round < requested_round and wait_count < max_wait:
                await asyncio.sleep(1)
                wait_count += 1
                if wait_count % 10 == 0:
                    logger.info(f"  Still waiting... (current_round: {self.current_round}, requested: {requested_round}, wait: {wait_count}s)")
                    #print(f"[SERVER DEBUG]   Still waiting... (current_round={self.current_round}, requested={requested_round}, waited={wait_count}s)")

            if wait_count >= max_wait:
                logger.warning(f"{client_id} timed out waiting for round {requested_round} (current: {self.current_round}). Sending current model.")
                #print(f"[SERVER DEBUG] TIMEOUT: waited {wait_count}s, sending current model (round {self.current_round})")
            else:
                #print(f"[SERVER DEBUG] Round {requested_round} ready! (waited {wait_count}s)")

        logger.info(f"Serializing model weights for {client_id} (round {self.current_round})...")
        #print(f"[SERVER DEBUG] Serializing model weights for {client_id} (round {self.current_round})...")
        start_serialize = time.time()
        weights_bytes = pickle.dumps(self.model.get_weights())
        serialize_time = time.time() - start_serialize
        logger.info(f"  Model serialized in {serialize_time:.3f}s ({len(weights_bytes) / 1024 / 1024:.2f} MB)")
        #print(f"[SERVER DEBUG] Model serialized: {serialize_time:.3f}s, {len(weights_bytes) / 1024 / 1024:.2f} MB")
        
        # Reset operation status
        self.current_operation = "idle"
        self.operation_start_time = None
        self.operation_details = {}
        #print(f"[SERVER DEBUG] Sending model to {client_id} (round {self.current_round})")
        
        return web.Response(
            body=weights_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Round": str(self.current_round),
                "X-Learning-Mode": self.learning_mode,
                "X-Output-Size": str(self.output_size)
            }
        )

    async def handle_submit_update(self, request: web.Request) -> web.Response:
        """Receive model updates."""
        import time
        #print(f"[SERVER DEBUG] ===== SUBMIT_UPDATE REQUEST RECEIVED =====")
        client_id = request.headers.get("X-Client-ID")
        num_samples = int(request.headers.get("X-Num-Samples", 0))
        loss = float(request.headers.get("X-Loss", 0.0))
        client_mode = request.headers.get("X-Learning-Mode", None)  # Optional header
        #print(f"[SERVER DEBUG] Submit update: client_id={client_id}, num_samples={num_samples}, loss={loss:.6f}, client_mode={client_mode}")

        if client_id not in self.registered_clients:
            #print(f"[SERVER DEBUG] ERROR: {client_id} not registered! Registered clients: {self.registered_clients}")
            return web.json_response({"error": "Invalid client"}, status=403)

        # Update operation status
        self.current_operation = f"receiving_update_from_{client_id}"
        self.operation_start_time = time.time()
        self.operation_details = {"client_id": client_id, "round": self.current_round}
        #print(f"[SERVER DEBUG] Operation set: {self.current_operation}, round={self.current_round}")

        logger.info(f"Receiving update from {client_id}...")
        #print(f"[SERVER DEBUG] Reading weights bytes from {client_id}...")
        start_read = time.time()
        weights_bytes = await request.read()
        read_time = time.time() - start_read
        logger.info(f"  Received {len(weights_bytes) / 1024 / 1024:.2f} MB in {read_time:.3f}s")
        #print(f"[SERVER DEBUG] Received {len(weights_bytes) / 1024 / 1024:.2f} MB in {read_time:.3f}s")
        
        try:
            #print(f"[SERVER DEBUG] Deserializing weights from {client_id}...")
            start_deserialize = time.time()
            client_weights = pickle.loads(weights_bytes)
            deserialize_time = time.time() - start_deserialize
            logger.info(f"  Deserialized weights in {deserialize_time:.3f}s")
            #print(f"[SERVER DEBUG] Deserialized weights in {deserialize_time:.3f}s")
        except Exception as e:
            logger.error(f"Error deserializing weights from {client_id}: {e}")
            #print(f"[SERVER DEBUG] ERROR deserializing weights: {e}")
            self.current_operation = "idle"
            self.operation_start_time = None
            self.operation_details = {}
            return web.json_response({"error": "Invalid weights format"}, status=400)

        # Validate weight shapes match server model
        try:
            #print(f"[SERVER DEBUG] Validating weight shapes from {client_id}...")
            start_validate = time.time()
            self._validate_weight_shapes(client_weights, client_id)
            validate_time = time.time() - start_validate
            logger.info(f"  Validated weights in {validate_time:.3f}s")
            #print(f"[SERVER DEBUG] Weight shapes validated in {validate_time:.3f}s")
        except ValueError as e:
            logger.error(f"Weight shape mismatch from {client_id}: {e}")
            #print(f"[SERVER DEBUG] ERROR: Weight shape mismatch from {client_id}: {e}")
            self.current_operation = "idle"
            self.operation_start_time = None
            self.operation_details = {}
            return web.json_response({"error": f"Weight shape mismatch: {e}"}, status=400)

        async with self.round_lock:
            self.client_weights[client_id] = client_weights
            self.client_samples[client_id] = num_samples
            self.client_losses[client_id] = loss
            if client_mode:
                self.client_modes[client_id] = client_mode
            self.ready_clients.add(client_id)

            mode_str = f" [{client_mode}]" if client_mode else ""
            logger.info(
                f"Update from {client_id}{mode_str}: Loss={loss:.6f} "
                f"({len(self.ready_clients)}/{len(self.registered_clients)})"
            )
            #print(f"[SERVER DEBUG] Update stored: {client_id}{mode_str}, loss={loss:.6f}, ready={len(self.ready_clients)}/{len(self.registered_clients)}")

            # Aggregate when ALL registered clients are ready
            #print(f"[SERVER DEBUG] Checking aggregation condition: ready={len(self.ready_clients)}, registered={len(self.registered_clients)}")
            if len(self.ready_clients) >= len(self.registered_clients):
                #print(f"[SERVER DEBUG] *** ALL CLIENTS READY - STARTING AGGREGATION ***")
                # Track loss for plotting BEFORE aggregation
                for cid in self.client_losses:
                    if cid in self.loss_history:
                        self.loss_history[cid].append(self.client_losses[cid])

                await self._aggregate_and_update()

                self.current_operation = "idle"
                self.operation_start_time = None
                self.operation_details = {}
                #print(f"[SERVER DEBUG] Aggregation complete, returning 'aggregated' status to {client_id}")
                
                return web.json_response({
                    "status": "aggregated",
                    "round": self.current_round,
                    "learning_mode": self.learning_mode
                })
            else:
                waiting_for = list(self.registered_clients - self.ready_clients)
                self.current_operation = f"waiting_for_clients ({len(self.ready_clients)}/{len(self.registered_clients)})"
                self.operation_details = {
                    "ready": len(self.ready_clients),
                    "total": len(self.registered_clients),
                    "waiting_for": waiting_for
                }
                #print(f"[SERVER DEBUG] Waiting for more clients: ready={len(self.ready_clients)}/{len(self.registered_clients)}, waiting_for={waiting_for}")
                
                return web.json_response({
                    "status": "waiting",
                    "round": self.current_round,
                    "waiting_for": len(self.registered_clients) - len(self.ready_clients)
                })

    def _validate_weight_shapes(self, client_weights: Dict, client_id: str):
        """Validate that client weights have the same shape as server model."""
        server_weights = self.model.get_weights()
        #print(f"[SERVER DEBUG] Validating shapes: server has {len(server_weights)} weight keys, client has {len(client_weights)} keys")
        
        # Check that all keys match
        if set(client_weights.keys()) != set(server_weights.keys()):
            missing = set(server_weights.keys()) - set(client_weights.keys())
            extra = set(client_weights.keys()) - set(server_weights.keys())
            #print(f"[SERVER DEBUG] Key mismatch: missing={missing}, extra={extra}")
            raise ValueError(f"Key mismatch: missing {missing}, extra {extra}")
        
        # Check that shapes match
        #print(f"[SERVER DEBUG] Checking shape matches for {len(server_weights)} weight tensors...")
        for key in server_weights.keys():
            server_shape = server_weights[key].shape
            client_shape = client_weights[key].shape
            if server_shape != client_shape:
                #print(f"[SERVER DEBUG] Shape mismatch for {key}: server {server_shape} vs client {client_shape}")
                raise ValueError(
                    f"Shape mismatch for {key}: server {server_shape} vs client {client_shape}"
                )
        #print(f"[SERVER DEBUG] All weight shapes match!")

    async def _aggregate_and_update(self):
        """Aggregate and update global model."""
        import time
        #print(f"[SERVER DEBUG] ===== AGGREGATE_AND_UPDATE CALLED =====")
        if not self.client_weights:
            #print(f"[SERVER DEBUG] WARNING: No client weights to aggregate!")
            return

        try:
            # Update operation status
            self.current_operation = "aggregating_weights"
            self.operation_start_time = time.time()
            self.operation_details = {
                "round": self.current_round,
                "num_clients": len(self.client_weights),
                "strategy": self.aggregator.strategy
            }
            
            logger.info(f"Starting aggregation for round {self.current_round + 1}...")
            logger.info(f"  Strategy: {self.aggregator.strategy}, Clients: {len(self.client_weights)}")
            #print(f"[SERVER DEBUG] Starting aggregation: round={self.current_round + 1}, strategy={self.aggregator.strategy}, clients={len(self.client_weights)}")
            
            weights_list = list(self.client_weights.values())
            samples_list = list(self.client_samples.values())
            #print(f"[SERVER DEBUG] Aggregating {len(weights_list)} weight sets, samples={samples_list}")

            start_aggregate = time.time()
            aggregated_weights = self.aggregator.aggregate(
                weights_list,
                samples_list if self.aggregator.strategy == "weighted" else None
            )
            aggregate_time = time.time() - start_aggregate
            logger.info(f"  Aggregation completed in {aggregate_time:.3f}s")
            #print(f"[SERVER DEBUG] Aggregation completed in {aggregate_time:.3f}s")

            start_set_weights = time.time()
            #print(f"[SERVER DEBUG] Setting aggregated weights to model...")
            self.model.set_weights(aggregated_weights)
            set_weights_time = time.time() - start_set_weights
            logger.info(f"  Model weights updated in {set_weights_time:.3f}s")
            #print(f"[SERVER DEBUG] Model weights updated in {set_weights_time:.3f}s")

            avg_loss = sum(self.client_losses.values()) / len(self.client_losses)
            self.round_losses.append(avg_loss)  # Track average loss

            # Log learning mode distribution
            mode_dist = {}
            for cid, mode in self.client_modes.items():
                mode_dist[mode] = mode_dist.get(mode, 0) + 1
            mode_str = f", Modes: {mode_dist}" if mode_dist else ""

            logger.info(
                f"✓ Round {self.current_round + 1} complete - "
                f"Avg Loss: {avg_loss:.6f}, Clients: {len(self.ready_clients)}{mode_str}"
            )
            #print(f"[SERVER DEBUG] Round {self.current_round + 1} complete: avg_loss={avg_loss:.6f}, clients={len(self.ready_clients)}{mode_str}")

            old_round = self.current_round
            self.current_round += 1
            #print(f"[SERVER DEBUG] Round incremented: {old_round} -> {self.current_round}")
            self.client_weights.clear()
            self.client_samples.clear()
            self.client_losses.clear()
            self.ready_clients.clear()
            #print(f"[SERVER DEBUG] Cleared client state (weights, samples, losses, ready_clients)")

            start_checkpoint = time.time()
            #print(f"[SERVER DEBUG] Saving checkpoint...")
            self._save_checkpoint()
            checkpoint_time = time.time() - start_checkpoint
            logger.info(f"  Checkpoint saved in {checkpoint_time:.3f}s")
            #print(f"[SERVER DEBUG] Checkpoint saved in {checkpoint_time:.3f}s")
            
            total_time = time.time() - self.operation_start_time
            logger.info(f"  Total aggregation time: {total_time:.3f}s")
            #print(f"[SERVER DEBUG] Total aggregation time: {total_time:.3f}s")
            
            # Reset operation status
            self.current_operation = "idle"
            self.operation_start_time = None
            self.operation_details = {}
            #print(f"[SERVER DEBUG] ===== AGGREGATION COMPLETE =====")

        except Exception as e:
            logger.error(f"Aggregation error: {e}", exc_info=True)
            #print(f"[SERVER DEBUG] ERROR in aggregation: {e}")
            self.current_operation = "error"
            self.operation_details = {"error": str(e)}

    def _save_checkpoint(self):
        """Save checkpoint."""
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        path = f"{config.CHECKPOINT_DIR}/server_round_{self.current_round}.pt"
        checkpoint = {
            "round": self.current_round,
            "model_state_dict": self.model.state_dict(),
            "learning_mode": self.learning_mode,
            "output_size": self.output_size
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def save_loss_plot(self):
        """Save loss plot for all clients and average loss."""
        os.makedirs(config.PLOT_DIR, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Plot individual client losses
        for client_id, losses in self.loss_history.items():
            if losses:  # Only plot if client has loss data
                rounds = list(range(1, len(losses) + 1))
                # Add mode label if available
                label = client_id
                if client_id in self.client_modes:
                    label = f"{client_id} [{self.client_modes[client_id]}]"
                plt.plot(rounds, losses, marker='o', label=label, alpha=0.7)

        # Plot average loss
        if self.round_losses:
            rounds = list(range(1, len(self.round_losses) + 1))
            plt.plot(rounds, self.round_losses, marker='s', linewidth=2,
                    label='Average', color='black', linestyle='--')

        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        mode_str = f" ({self.learning_mode})" if self.learning_mode else ""
        plt.title(f'Federated Learning - Training Loss per Round{mode_str}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = f"{config.PLOT_DIR}/training_loss.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Loss plot saved: {plot_path}")
        return plot_path

    async def handle_status(self, request: web.Request) -> web.Response:
        """Server status."""
        import time
        print(f"[SERVER DEBUG] ===== STATUS REQUEST RECEIVED =====")
        # Count learning modes
        mode_counts = {}
        for mode in self.client_modes.values():
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        #print(f"[SERVER DEBUG] Status: round={self.current_round}, registered={len(self.registered_clients)}, ready={len(self.ready_clients)}, operation={self.current_operation}")

        # Convert loss history to lists (for JSON serialization)
        loss_history_dict = {}
        for client_id, losses in self.loss_history.items():
            # Convert numpy types to native Python types
            loss_history_dict[client_id] = [float(loss) for loss in losses]

        # Calculate operation duration if active
        operation_duration = None
        if self.operation_start_time is not None:
            operation_duration = time.time() - self.operation_start_time

        return web.json_response({
            "current_round": self.current_round,
            "registered_clients": len(self.registered_clients),
            "ready_clients": len(self.ready_clients),
            "total_expected": self.num_clients,
            "aggregation_strategy": self.aggregator.strategy,
            "learning_mode": self.learning_mode,
            "output_size": self.output_size,
            "client_modes": mode_counts,
            "loss_history": loss_history_dict,
            "round_losses": [float(loss) for loss in self.round_losses],
            "current_client_losses": {cid: float(loss) for cid, loss in self.client_losses.items()},
            "current_operation": self.current_operation,
            "operation_duration": round(operation_duration, 3) if operation_duration is not None else None,
            "operation_details": self.operation_details
        })

    async def handle_save_plot(self, request: web.Request) -> web.Response:
        """Generate and save loss plot."""
        try:
            plot_path = self.save_loss_plot()
            return web.json_response({
                "status": "success",
                "plot_path": plot_path
            })
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)

    def create_app(self) -> web.Application:
        """Create app."""
        app = web.Application()
        app.router.add_post("/register", self.handle_register)
        app.router.add_post("/get_model", self.handle_get_model)
        app.router.add_post("/submit_update", self.handle_submit_update)
        app.router.add_get("/status", self.handle_status)
        app.router.add_post("/save_plot", self.handle_save_plot)
        return app

    def run(self):
        """Start server."""
        app = self.create_app()
        logger.info(f"Starting FL Server on {self.host}:{self.port}")
        logger.info(f"Learning mode: {self.learning_mode}, Output size: {self.output_size}")
        #print(f"[SERVER DEBUG] ===== SERVER STARTING =====")
        #print(f"[SERVER DEBUG] Host: {self.host}, Port: {self.port}")
        #print(f"[SERVER DEBUG] Learning mode: {self.learning_mode}, Output size: {self.output_size}")
        #print(f"[SERVER DEBUG] Input size: {self.input_size}")
        #print(f"[SERVER DEBUG] Expected clients: {self.num_clients}, Min clients: {self.min_clients}")
        #print(f"[SERVER DEBUG] Aggregation strategy: {self.aggregator.strategy}")
        web.run_app(app, host=self.host, port=self.port, print=None)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Federated Learning Server with Packet Support')
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--num-clients", type=int, default=3, help="Expected number of clients")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients to start training")
    parser.add_argument("--aggregation", default="fedavg", 
                       choices=['fedavg', 'fedavgm', 'weighted'],
                       help="Aggregation strategy")
    parser.add_argument("--output-size", type=int, default=None,
                       help="Model output size (None = auto-detect)")
    parser.add_argument("--learning-mode", default=None,
                       choices=['classification', 'regression'],
                       help="Learning mode (None = auto-detect)")
    parser.add_argument("--input-size", type=int, default=None,
                       help="Model input size (None = auto-detect from config)")
    args = parser.parse_args()

    server = FLServerClassification(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        min_clients=args.min_clients,
        aggregation_strategy=args.aggregation,
        output_size=args.output_size,
        learning_mode=args.learning_mode,
        input_size=args.input_size
    )

    server.run()


if __name__ == "__main__":
    main()
