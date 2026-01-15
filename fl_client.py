"""
Federated Learning Client

Loads GPS+IMU data, trains on sliding windows, syncs with server.
"""
import asyncio
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import aiohttp
import logging
import importlib

from data_preprocessing import DataPreprocessor
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class FLClient:
    """Federated Learning Client with windowed training."""

    def __init__(
        self,
        client_id: str,
        gps_file: str,
        imu_file: str,
        server_url: str
    ):
        """Initialize FL Client."""
        self.client_id = client_id
        self.server_url = server_url

        logger.info(f"Initializing FL Client: {client_id}")

        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(
            gps_file,
            imu_file,
            gps_features=config.GPS_FEATURES,
            imu_features=config.IMU_FEATURES,
            timestamp_col=config.TIMESTAMP_COL
        )
        self.X, self.y, self.num_features = preprocessor.preprocess(
            config.WINDOW_SIZE,
            config.OVERLAP
        )

        self.total_windows = len(self.X)
        logger.info(f"Total windows available: {self.total_windows}")

        # Load model dynamically from config
        logger.info(f"Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}")
        model_module = importlib.import_module(config.MODEL_PATH)
        model_class = getattr(model_module, config.MODEL_CLASS)

        # Initialize model
        self.model = model_class(**config.MODEL_CONFIG)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")

        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Track current position
        self.current_window_idx = 0

    async def register(self, session: aiohttp.ClientSession) -> bool:
        """Register with server."""
        url = f"{self.server_url}/register"
        data = {"client_id": self.client_id}

        try:
            async with session.post(url, json=data) as response:
                result = await response.json()
                logger.info(f"Registration: {result['message']}")
                return response.status == 200
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    async def get_global_model(self, session: aiohttp.ClientSession) -> bool:
        """Download global model from server."""
        url = f"{self.server_url}/get_model"
        data = {"client_id": self.client_id}

        try:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    weights_bytes = await response.read()
                    weights = pickle.loads(weights_bytes)
                    self.model.set_weights(weights)
                    logger.info("Received global model")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error getting model: {e}")
            return False

    def train_on_windows(self, start_idx: int, end_idx: int) -> float:
        """Train on windows (single pass)."""
        self.model.train()
        total_loss = 0.0
        num_windows = 0

        for window_idx in range(start_idx, end_idx):
            if window_idx >= self.total_windows:
                break

            # Get window
            X_window = torch.FloatTensor(self.X[window_idx:window_idx+1]).to(self.device)
            y_window = torch.FloatTensor(self.y[window_idx:window_idx+1]).to(self.device)

            # Train
            self.optimizer.zero_grad()
            output = self.model(X_window)
            loss = self.criterion(output, y_window)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_windows += 1

        return total_loss / num_windows if num_windows > 0 else 0.0

    async def submit_update(self, session: aiohttp.ClientSession, loss: float) -> bool:
        """Submit model update to server."""
        url = f"{self.server_url}/submit_update"

        weights = self.model.get_weights()
        weights_bytes = pickle.dumps(weights)

        headers = {
            "X-Client-ID": self.client_id,
            "X-Num-Samples": str(config.WINDOWS_PER_ROUND),
            "X-Loss": str(loss),
            "Content-Type": "application/octet-stream"
        }

        try:
            async with session.post(url, data=weights_bytes, headers=headers) as response:
                result = await response.json()
                logger.info(f"Update submitted: {result['status']}")
                return response.status == 200
        except Exception as e:
            logger.error(f"Error submitting: {e}")
            return False

    async def run_federated_learning(self):
        """Run FL training."""
        async with aiohttp.ClientSession() as session:
            # Register
            logger.info(f"Registering {self.client_id}...")
            if not await self.register(session):
                return

            await asyncio.sleep(3)

            # Training rounds
            round_num = 0
            while self.current_window_idx < self.total_windows:
                round_num += 1
                end_idx = min(
                    self.current_window_idx + config.WINDOWS_PER_ROUND,
                    self.total_windows
                )

                logger.info(f"Round {round_num}: Windows {self.current_window_idx}-{end_idx}")

                # Get global model (skip first round)
                if round_num > 1:
                    await self.get_global_model(session)

                # Train
                loss = self.train_on_windows(self.current_window_idx, end_idx)
                logger.info(f"Round {round_num} Loss: {loss:.6f}")

                # Submit
                await self.submit_update(session, loss)

                self.current_window_idx = end_idx
                await asyncio.sleep(1)

            logger.info(f"✓ {self.client_id} completed {round_num} rounds!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--gps-file", required=True)
    parser.add_argument("--imu-file", required=True)
    parser.add_argument("--server-url", required=True)
    args = parser.parse_args()

    client = FLClient(
        client_id=args.client_id,
        gps_file=args.gps_file,
        imu_file=args.imu_file,
        server_url=args.server_url
    )

    asyncio.run(client.run_federated_learning())


if __name__ == "__main__":
    main()
