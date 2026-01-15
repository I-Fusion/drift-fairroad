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

        logger.info(f"Server initialized with {aggregation_strategy} aggregation")

    async def handle_register(self, request: web.Request) -> web.Response:
        """Handle client registration."""
        data = await request.json()
        client_id = data.get("client_id")

        async with self.round_lock:
            self.registered_clients.add(client_id)
            logger.info(f"{client_id} registered ({len(self.registered_clients)}/{self.num_clients})")

        return web.json_response({
            "status": "registered",
            "current_round": self.current_round,
            "message": f"{len(self.registered_clients)}/{self.num_clients} clients registered"
        })

    async def handle_get_model(self, request: web.Request) -> web.Response:
        """Send global model to client."""
        data = await request.json()
        client_id = data.get("client_id")

        if client_id not in self.registered_clients:
            return web.json_response({"error": "Not registered"}, status=403)

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
                f"({len(self.ready_clients)}/{self.num_clients})"
            )

            # Aggregate when ready
            if len(self.ready_clients) >= min(self.min_clients, self.num_clients):
                await self._aggregate_and_update()

                return web.json_response({
                    "status": "aggregated",
                    "round": self.current_round
                })
            else:
                return web.json_response({
                    "status": "waiting",
                    "round": self.current_round
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

            logger.info(
                f"✓ Round {self.current_round} complete - "
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

    async def handle_status(self, request: web.Request) -> web.Response:
        """Server status."""
        return web.json_response({
            "current_round": self.current_round,
            "registered_clients": len(self.registered_clients),
            "ready_clients": len(self.ready_clients),
            "total_expected": self.num_clients,
            "aggregation_strategy": self.aggregator.strategy
        })

    def create_app(self) -> web.Application:
        """Create app."""
        app = web.Application()
        app.router.add_post("/register", self.handle_register)
        app.router.add_post("/get_model", self.handle_get_model)
        app.router.add_post("/submit_update", self.handle_submit_update)
        app.router.add_get("/status", self.handle_status)
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
