"""
Federated Learning Client for Convolutional Autoencoder

Trains CAE on local clean/noisy image pairs and shares weights with server.
"""
import asyncio
import pickle
import torch
import aiohttp
from aiohttp import ClientTimeout
import logging
import importlib
from tqdm import tqdm

from data_preprocessing_cae import CAEDataPreprocessor
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def _debug(client_id: str, msg: str):
    pass  # print(f"[FL-CLIENT {client_id}] {msg}", flush=True)


class FLClientCAE:
    """Federated Learning Client for Convolutional Autoencoder"""

    def __init__(self, client_id: str, clean_dir: str, noisy_dir: str, server_url: str):
        """
        Initialize FL Client for CAE.

        Args:
            client_id: Unique client identifier (e.g., 'client_1')
            clean_dir: Directory with clean images
            noisy_dir: Directory with noisy images
            server_url: FL server URL
        """
        self.client_id = client_id
        self.server_url = server_url

        logger.info(f"\n{'='*60}")
        logger.info(f"CAE FL CLIENT: {client_id.upper()}")
        logger.info(f"{'='*60}")

        # Extract client number (0-indexed)
        self.client_num = int(client_id.split('_')[1]) - 1

        # Load data
        logger.info("Loading image pairs for this client...")
        self.preprocessor = CAEDataPreprocessor(
            clean_dir=clean_dir,
            noisy_dir=noisy_dir,
            img_size=config.AUTOENCODER_IMAGE_SIZE,
            num_clients=config.NUM_CLIENTS
        )

        self.dataloader, self.num_samples = self.preprocessor.get_dataloader(
            client_id=self.client_num,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

        logger.info(f"✓ Assigned {self.num_samples} image pairs to {client_id}")
        logger.info(f"✓ Batches per epoch: {len(self.dataloader)}")

        # Load model
        logger.info("\nLoading CAE model...")
        model_module = importlib.import_module(config.MODEL_PATH)
        model_class = getattr(model_module, config.MODEL_CLASS)
        self.model = model_class(**config.MODEL_CONFIG)

        logger.info(f"✓ Model loaded on {self.model.device}")
        logger.info(f"✓ Parameters: {self.model.count_parameters():,}")
        logger.info(f"✓ Latent dimension: {self.model.latent_dim}")

        self.current_round = 0

    async def register(self, session: aiohttp.ClientSession) -> bool:
        """Register with FL server"""
        url = f"{self.server_url}/register"
        data = {"client_id": self.client_id}
        _debug(self.client_id, "register: sending POST")
        try:
            async with session.post(url, json=data) as response:
                result = await response.json()
                _debug(self.client_id, f"register: status={response.status}, {result.get('message', '')[:50]}")
                logger.info(f"\n✓ {result['message']}")
                return response.status == 200
        except Exception as e:
            _debug(self.client_id, f"register: failed {e}")
            logger.error(f"✗ Registration failed: {e}")
            return False

    async def get_global_model(self, session: aiohttp.ClientSession, round_num: int) -> bool:
        """Download global model from server. Retries on 503 (round not ready)."""
        url = f"{self.server_url}/get_model"
        data = {"client_id": self.client_id, "round": round_num}
        max_retries = 5
        for attempt in range(max_retries):
            _debug(self.client_id, f"get_global_model: round={round_num} attempt={attempt + 1}/{max_retries}")
            logger.info(f"\n→ Requesting global model (round {round_num})... attempt {attempt + 1}/{max_retries}")
            try:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        weights_bytes = await response.read()
                        _debug(self.client_id, f"get_global_model: got {len(weights_bytes)} bytes, loading...")
                        weights = pickle.loads(weights_bytes)
                        self.model.set_weights(weights)
                        _debug(self.client_id, "get_global_model: done")
                        logger.info(f"✓ Received and loaded global model")
                        return True
                    if response.status == 503:
                        body = await response.text()
                        _debug(self.client_id, "get_global_model: 503 round not ready, retry in 10s")
                        logger.info(f"  Round not ready, retrying in 10s... ({body[:80]})")
                        await asyncio.sleep(10)
                        continue
                    body = await response.text()
                    _debug(self.client_id, f"get_global_model: error {response.status} {body[:80]}")
                    logger.error(f"✗ Error getting model: {response.status}, {body[:150]}")
                    return False
            except Exception as e:
                _debug(self.client_id, f"get_global_model: exception {e}")
                logger.error(f"✗ Error getting model: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)
                else:
                    return False
        return False

    def train_local_epoch(self, epochs_per_round=1):
        """
        Train model locally for specified epochs.

        Args:
            epochs_per_round: Number of epochs to train

        Returns:
            avg_loss: Average training loss
        """
        _debug(self.client_id, f"train_local_epoch: round {self.current_round + 1} start")
        logger.info(f"\n{'='*60}")
        logger.info(f"LOCAL TRAINING - ROUND {self.current_round + 1}")
        logger.info(f"{'='*60}")

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs_per_round):
            epoch_loss = 0.0
            batch_count = 0

            # Progress bar for batches
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs_per_round}")

            for noisy_batch, clean_batch in pbar:
                loss = self.model.train_step(noisy_batch, clean_batch)

                epoch_loss += loss
                batch_count += 1

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.6f}'})

            if batch_count == 0:
                logger.warning("No batches in epoch; skipping")
                continue
            avg_epoch_loss = epoch_loss / batch_count
            total_loss += avg_epoch_loss
            num_batches += 1

            logger.info(f"  Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.6f}")

        avg_loss = total_loss / num_batches if num_batches else 0.0

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Average Loss: {avg_loss:.6f}")
        total_batches = num_batches * len(self.dataloader) * epochs_per_round if num_batches else 0
        logger.info(f"Total Batches: {total_batches}")
        _debug(self.client_id, f"train_local_epoch: round {self.current_round + 1} done, avg_loss={avg_loss:.6f}")
        return avg_loss

    async def submit_update(self, session: aiohttp.ClientSession, avg_loss: float) -> bool:
        """Submit model update to server"""
        _debug(self.client_id, "submit_update: building payload")
        logger.info(f"\n→ Submitting model update...")
        logger.info(f"  Average loss: {avg_loss:.6f}")

        url = f"{self.server_url}/submit_update"

        weights = self.model.get_weights()
        weights_bytes = pickle.dumps(weights)
        _debug(self.client_id, f"submit_update: posting {len(weights_bytes)} bytes")
        logger.info(f"  Weight payload: {len(weights_bytes):,} bytes")

        headers = {
            "X-Client-ID": self.client_id,
            "X-Num-Samples": str(self.num_samples),
            "X-Loss": str(avg_loss),
            "Content-Type": "application/octet-stream"
        }

        try:
            async with session.post(url, data=weights_bytes, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    _debug(self.client_id, f"submit_update: done status={result.get('status', '')}")
                    logger.info(f"✓ Update submitted: {result['status']}")
                    return True
                body = await response.text()
                _debug(self.client_id, f"submit_update: error {response.status} {body[:80]}")
                logger.error(f"✗ Error submitting: {response.status}, {body[:200]}")
                return False
        except Exception as e:
            _debug(self.client_id, f"submit_update: exception {e}")
            logger.error(f"✗ Error submitting: {e}")
            return False

    async def run_federated_learning(self, num_rounds: int):
        """
        Run federated learning for specified number of rounds.

        Args:
            num_rounds: Number of FL rounds to execute
        """
        # 48MB model transfer can be very slow; use long timeouts
        timeout = ClientTimeout(total=600, sock_connect=30, sock_read=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Register with server
            if not await self.register(session):
                return

            await asyncio.sleep(3)

            # Stagger start so all clients don't request get_model at once (uses config.CLIENT_STAGGER_SEC per client index)
            stagger_sec = getattr(config, 'CLIENT_STAGGER_SEC', 10) * self.client_num
            if stagger_sec > 0:
                _debug(self.client_id, f"stagger: sleeping {stagger_sec}s before round 1 (client index {self.client_num})")
                logger.info(f"Staggering start by {stagger_sec}s to avoid simultaneous model requests...")
                await asyncio.sleep(stagger_sec)
                _debug(self.client_id, "stagger: done, starting round 1")
            else:
                _debug(self.client_id, "stagger: 0s (client_1), starting round 1")

            # Training rounds
            for round_num in range(num_rounds):
                self.current_round = round_num
                _debug(self.client_id, f"round {round_num + 1}/{num_rounds} start")

                logger.info(f"\n{'#'*60}")
                logger.info(f"# FEDERATED ROUND {round_num + 1}/{num_rounds}")
                logger.info(f"{'#'*60}")

                # Get global model (skip first round)
                if round_num > 0:
                    await self.get_global_model(session, round_num)

                # Train locally
                avg_loss = self.train_local_epoch(epochs_per_round=config.EPOCHS_PER_ROUND)

                # Submit update to server
                await self.submit_update(session, avg_loss)

                await asyncio.sleep(1)

            logger.info(f"\n{'='*60}")
            logger.info(f"✓ {self.client_id.upper()} FINISHED - {num_rounds} ROUNDS")
            logger.info(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True, help="Client ID (e.g., client_1)")
    parser.add_argument("--clean-dir", required=True, help="Directory with clean images")
    parser.add_argument("--noisy-dir", required=True, help="Directory with noisy images")
    parser.add_argument("--server-url", required=True, help="FL server URL")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of FL rounds")
    args = parser.parse_args()

    client = FLClientCAE(
        client_id=args.client_id,
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        server_url=args.server_url
    )

    asyncio.run(client.run_federated_learning(num_rounds=args.num_rounds))


if __name__ == "__main__":
    main()
