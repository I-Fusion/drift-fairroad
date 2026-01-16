"""
Federated Learning Server

Coordinates training and aggregates model updates.
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

from aggregation import FederatedAggregator
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


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

        logger.info(f"Server initialized with {aggregation_strategy} aggregation")

    async def handle_register(self, request: web.Request) -> web.Response:
        """Handle client registration."""
        data = await request.json()
        client_id = data.get("client_id")

        async with self.round_lock:
            if client_id not in self.registered_clients:
                self.registered_clients.add(client_id)
                self.loss_history[client_id] = []  # Initialize loss history
                logger.info(f"{client_id} registered ({len(self.registered_clients)}/{self.num_clients})")

            # Check if minimum clients reached and training should start
            if not self.training_started and len(self.registered_clients) >= self.min_clients:
                self.training_started = True
                logger.info(f"✓ Minimum {self.min_clients} clients registered. Training can start!")

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

        if client_id not in self.registered_clients:
            return web.json_response({"error": "Not registered"}, status=403)

        # Wait for training to start (minimum clients)
        while not self.training_started:
            await asyncio.sleep(0.5)

        # Wait for the requested round to be ready (aggregation completed)
        # If client requests round N, wait until server.current_round >= N
        # (meaning round N aggregation is complete and current_round incremented to N)
        max_wait = 120  # Maximum 2 minutes wait
        wait_count = 0
        while self.current_round < requested_round and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1

        if wait_count >= max_wait:
            logger.warning(f"{client_id} timed out waiting for round {requested_round} (current: {self.current_round})")

        weights_bytes = pickle.dumps(self.model.get_weights())
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

        if client_id not in self.registered_clients:
            return web.json_response({"error": "Invalid client"}, status=403)

        weights_bytes = await request.read()
        client_weights = pickle.loads(weights_bytes)

        async with self.round_lock:
            self.client_weights[client_id] = client_weights
            self.client_samples[client_id] = num_samples
            self.client_losses[client_id] = loss
            self.ready_clients.add(client_id)

            logger.info(
                f"Update from {client_id}: Loss={loss:.6f} "
                f"({len(self.ready_clients)}/{len(self.registered_clients)})"
            )

            # Aggregate when ALL registered clients are ready
            if len(self.ready_clients) >= len(self.registered_clients):
                # Track loss for plotting BEFORE aggregation (when all clients have submitted)
                for cid in self.client_losses:
                    if cid in self.loss_history:
                        self.loss_history[cid].append(self.client_losses[cid])

                await self._aggregate_and_update()

                return web.json_response({
                    "status": "aggregated",
                    "round": self.current_round
                })
            else:
                return web.json_response({
                    "status": "waiting",
                    "round": self.current_round,
                    "waiting_for": len(self.registered_clients) - len(self.ready_clients)
                })

    async def _aggregate_and_update(self):
        """Aggregate and update global model."""
        if not self.client_weights:
            return

        try:
            weights_list = list(self.client_weights.values())
            samples_list = list(self.client_samples.values())

            aggregated_weights = self.aggregator.aggregate(
                weights_list,
                samples_list if self.aggregator.strategy == "weighted" else None
            )

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

            self._save_checkpoint()

        except Exception as e:
            logger.error(f"Aggregation error: {e}")

    def _save_checkpoint(self):
        """Save checkpoint."""
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        path = f"{config.CHECKPOINT_DIR}/server_round_{self.current_round}.pt"
        torch.save({"round": self.current_round, "model_state_dict": self.model.state_dict()}, path)
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
