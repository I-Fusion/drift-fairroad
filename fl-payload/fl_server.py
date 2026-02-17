"""
Federated Learning Server

Coordinates training and aggregates model updates.
Pickle serialize/deserialize runs in a thread executor so the event loop
stays responsive for /status and other requests (avoids monitor timeouts).
"""
import asyncio
import pickle
import torch
from aiohttp import web
import logging
import importlib
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from aggregation import FederatedAggregator
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def _debug(msg: str):
    pass  # print(f"[FL-SERVER] {msg}", flush=True)


class FLServer:
    """Federated Learning Server."""

    def __init__(
        self,
        host: str,
        port: int,
        num_clients: int,
        min_clients: int,
        aggregation_strategy: str
    ):
        """Initialize FL Server."""
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.min_clients = min_clients

        # Load model dynamically from config
        logger.info(f"Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}")
        model_module = importlib.import_module(config.MODEL_PATH)
        model_class = getattr(model_module, config.MODEL_CLASS)

        # Initialize model
        self.model = model_class(**config.MODEL_CONFIG)
        logger.info(f"Model parameters: {self.model.count_parameters()}")

        # Aggregator
        self.aggregator = FederatedAggregator(strategy=aggregation_strategy)

        # State
        self.current_round = 0
        self.client_weights = {}
        self.client_samples = {}
        self.client_losses = {}
        self.registered_clients = set()
        self.ready_clients = set()
        self.round_lock = asyncio.Lock()

        # Loss tracking for plotting
        self.loss_history = {}  # {client_id: [losses per round]}
        self.round_losses = []  # Average loss per round

        # Flag to track if training has started
        self.training_started = False

        # Executor for CPU-bound pickle so event loop can still serve /status etc.
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info(f"Server initialized with {aggregation_strategy} aggregation")

    async def handle_register(self, request: web.Request) -> web.Response:
        """Handle client registration."""
        data = await request.json()
        client_id = data.get("client_id")
        _debug(f"register request from {client_id}")

        async with self.round_lock:
            if client_id not in self.registered_clients:
                self.registered_clients.add(client_id)
                self.loss_history[client_id] = []  # Initialize loss history
                logger.info(f"{client_id} registered ({len(self.registered_clients)}/{self.num_clients})")

            # Check if minimum clients reached and training should start
            if not self.training_started and len(self.registered_clients) >= self.min_clients:
                self.training_started = True
                logger.info(f"✓ Minimum {self.min_clients} clients registered. Training can start!")

        _debug(f"register done: {client_id} -> {len(self.registered_clients)}/{self.num_clients}, can_start={self.training_started}")
        return web.json_response({
            "status": "registered",
            "current_round": self.current_round,
            "can_start": self.training_started,
            "message": f"{len(self.registered_clients)}/{self.num_clients} clients registered"
        })

    async def handle_get_model(self, request: web.Request) -> web.Response:
        """Send global model to client."""
        data = await request.json()
        client_id = data.get("client_id")
        requested_round = data.get("round", 0)
        _debug(f"get_model: {client_id} requests round {requested_round} (current={self.current_round})")

        if client_id not in self.registered_clients:
            return web.json_response({"error": "Not registered"}, status=403)

        # Wait for training to start (minimum clients)
        while not self.training_started:
            await asyncio.sleep(0.5)

        # Wait for the requested round to be ready (aggregation completed)
        # If client requests round N, wait until server.current_round >= N
        max_wait = 300  # 5 minutes (clients may be staggered)
        wait_count = 0
        while self.current_round < requested_round and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1

        if self.current_round < requested_round:
            logger.warning(f"{client_id} timed out waiting for round {requested_round} (current: {self.current_round})")
            return web.json_response(
                {"error": "Round not ready", "current_round": self.current_round},
                status=503
            )

        # Read model under lock so we don't read during set_weights in aggregation
        _debug(f"get_model: acquiring lock for {client_id}")
        async with self.round_lock:
            weights = self.model.get_weights()
        _debug(f"get_model: pickling for {client_id} (executor)")
        loop = asyncio.get_running_loop()
        weights_bytes = await loop.run_in_executor(
            self._executor, lambda: pickle.dumps(weights)
        )
        _debug(f"get_model: sending {len(weights_bytes)} bytes to {client_id}")
        return web.Response(
            body=weights_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Round": str(self.current_round)
            }
        )

    async def handle_submit_update(self, request: web.Request) -> web.Response:
        """Receive model updates."""
        client_id = request.headers.get("X-Client-ID")
        num_samples = int(request.headers.get("X-Num-Samples", 0))
        loss = float(request.headers.get("X-Loss", 0.0))
        _debug(f"submit_update: {client_id} body_size=reading...")

        if client_id not in self.registered_clients:
            return web.json_response({"error": "Invalid client"}, status=403)

        weights_bytes = await request.read()
        _debug(f"submit_update: {client_id} read {len(weights_bytes)} bytes, unpickling (executor)")
        loop = asyncio.get_running_loop()
        client_weights = await loop.run_in_executor(
            self._executor, lambda: pickle.loads(weights_bytes)
        )

        _debug(f"submit_update: {client_id} acquiring lock")
        async with self.round_lock:
            self.client_weights[client_id] = client_weights
            self.client_samples[client_id] = num_samples
            self.client_losses[client_id] = loss
            self.ready_clients.add(client_id)

            logger.info(
                f"Update from {client_id}: Loss={loss:.6f} "
                f"({len(self.ready_clients)}/{len(self.registered_clients)})"
            )

            # Aggregate when ALL registered clients have submitted for this round
            if len(self.ready_clients) >= len(self.registered_clients):
                _debug(f"submit_update: all {len(self.ready_clients)} submitted -> aggregating")
                # Track loss for plotting BEFORE aggregation (when all clients have submitted)
                for cid in self.client_losses:
                    if cid in self.loss_history:
                        self.loss_history[cid].append(self.client_losses[cid])

                await self._aggregate_and_update()
                _debug(f"submit_update: aggregation done, round={self.current_round}")

                return web.json_response({
                    "status": "aggregated",
                    "round": self.current_round
                })
            else:
                waiting = len(self.registered_clients) - len(self.ready_clients)
                _debug(f"submit_update: {client_id} -> waiting ({waiting} more)")
                return web.json_response({
                    "status": "waiting",
                    "round": self.current_round,
                    "waiting_for": waiting
                })

    async def _aggregate_and_update(self):
        """Aggregate and update global model. Heavy work runs in executor so event loop stays responsive."""
        if not self.client_weights:
            return

        try:
            weights_list = list(self.client_weights.values())
            samples_list = list(self.client_samples.values())
            loop = asyncio.get_running_loop()
            _debug("_aggregate_and_update: running aggregate in executor")
            # Run aggregation in executor (CPU-heavy; blocks event loop otherwise)
            aggregated_weights = await loop.run_in_executor(
                self._executor,
                lambda: self.aggregator.aggregate(
                    weights_list,
                    samples_list if self.aggregator.strategy == "weighted" else None,
                ),
            )

            _debug("_aggregate_and_update: set_weights on main loop")
            self.model.set_weights(aggregated_weights)

            avg_loss = sum(self.client_losses.values()) / len(self.client_losses)
            self.round_losses.append(avg_loss)  # Track average loss

            logger.info(
                f"✓ Round {self.current_round + 1} complete - "
                f"Avg Loss: {avg_loss:.6f}, Clients: {len(self.ready_clients)}"
            )

            self.current_round += 1
            self.client_weights.clear()
            self.client_samples.clear()
            self.client_losses.clear()
            self.ready_clients.clear()

            # Run checkpoint save in executor so I/O doesn't block the loop
            round_done = self.current_round
            state_dict = self.model.get_weights()
            _debug(f"_aggregate_and_update: saving checkpoint round {round_done} (executor)")
            await loop.run_in_executor(
                self._executor,
                lambda: self._save_checkpoint_sync(round_done, state_dict),
            )
            _debug("_aggregate_and_update: done")

        except Exception as e:
            logger.error(f"Aggregation error: {e}")

    def _save_checkpoint_sync(self, round_num: int, state_dict):
        """Save checkpoint (sync, for use in executor)."""
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        path = f"{config.CHECKPOINT_DIR}/server_round_{round_num}.pt"
        torch.save({"round": round_num, "model_state_dict": state_dict}, path)
        logger.info(f"Checkpoint saved: {path}")

    def save_loss_plot(self):
        """Save loss plot for all clients and average loss."""
        os.makedirs(config.PLOT_DIR, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Plot individual client losses
        for client_id, losses in self.loss_history.items():
            if losses:  # Only plot if client has loss data
                rounds = list(range(1, len(losses) + 1))
                plt.plot(rounds, losses, marker='o', label=f'{client_id}', alpha=0.7)

        # Plot average loss
        if self.round_losses:
            rounds = list(range(1, len(self.round_losses) + 1))
            plt.plot(rounds, self.round_losses, marker='s', linewidth=2,
                    label='Average', color='black', linestyle='--')

        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Federated Learning - Training Loss per Round', fontsize=14, fontweight='bold')
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
        _debug("status: replying")
        return web.json_response({
            "current_round": self.current_round,
            "registered_clients": len(self.registered_clients),
            "ready_clients": len(self.ready_clients),
            "total_expected": self.num_clients,
            "aggregation_strategy": self.aggregator.strategy
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
        # CAE model weights ~48MB; allow up to 64MB for submit_update
        app = web.Application(client_max_size=64*1024*1024)
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
        web.run_app(app, host=self.host, port=self.port, print=None)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--aggregation", default="fedavg")
    args = parser.parse_args()

    server = FLServer(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        min_clients=args.min_clients,
        aggregation_strategy=args.aggregation
    )

    server.run()


if __name__ == "__main__":
    main()
