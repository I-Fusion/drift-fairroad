"""
Federated Learning Client with Packet Support

Loads GPS, IMU, and/or labeled packet data, trains on sliding windows, syncs with server.
Supports both classification (with labels) and time series prediction.
Each client loads its own data files.
"""
import asyncio
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import aiohttp
import logging
import importlib
import os
import sys
import numpy as np
from typing import Optional

from data_preprocessing_classification import DataPreprocessorClassification
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class FLClientClassification:
    """Federated Learning Client for Classification Task."""

    def __init__(
        self,
        client_id: str,
        server_url: str,
        gps_file: Optional[str] = None,
        imu_file: Optional[str] = None,
        packet_file: Optional[str] = None,
        use_labels: bool = False,
        data_dir: Optional[str] = None
    ):
        """
        Initialize FL Client.
        
        Args:
            client_id: Client identifier (e.g., 'client_1')
            server_url: Server URL
            gps_file: Path to GPS CSV file (optional)
            imu_file: Path to IMU CSV file (optional)
            packet_file: Path to labeled packet CSV file (optional)
            use_labels: If True, use packet labels for supervised learning
            data_dir: Directory containing client-specific data files (from prepare_fl_data_for_run_fl.py)
                     If provided, will look for gps_client_XXX.csv, imu_client_XXX.csv, packet_client_XXX.csv
        """
        self.client_id = client_id
        self.server_url = server_url
        self.use_labels = use_labels

        #print(f"[DEBUG {client_id}] Initializing FL Client...")
        logger.info(f"Initializing FL Client: {client_id}")

        # If data_dir is provided, try to load client-specific files
        print(f"[DEBUG {client_id}] Checking data_dir: {data_dir}")
        if data_dir:
            client_num = self._extract_client_number(client_id)
            #print(f"[DEBUG {client_id}] Extracted client number: {client_num}")
            if client_num is not None:
                # Try to find client-specific files
                gps_path = os.path.join(data_dir, f"gps_client_{client_num:03d}.csv")
                imu_path = os.path.join(data_dir, f"imu_client_{client_num:03d}.csv")
                packet_path = os.path.join(data_dir, f"packet_client_{client_num:03d}.csv")
                
                #print(f"[DEBUG {client_id}] Checking for client files:")
                print(f"  GPS: {gps_path} (exists: {os.path.exists(gps_path)})")
                print(f"  IMU: {imu_path} (exists: {os.path.exists(imu_path)})")
                print(f"  Packet: {packet_path} (exists: {os.path.exists(packet_path)})")
                
                if os.path.exists(gps_path):
                    gps_file = gps_file or gps_path
                    logger.info(f"Found client GPS file: {gps_path}")
                    #print(f"[DEBUG {client_id}] Using GPS file: {gps_path}")
                if os.path.exists(imu_path):
                    imu_file = imu_file or imu_path
                    logger.info(f"Found client IMU file: {imu_path}")
                    #print(f"[DEBUG {client_id}] Using IMU file: {imu_path}")
                if os.path.exists(packet_path):
                    packet_file = packet_file or packet_path
                    logger.info(f"Found client packet file: {packet_path}")
                    #print(f"[DEBUG {client_id}] Using packet file: {packet_path}")
                    # If packet file exists, enable supervised learning by default
                    if not use_labels:
                        use_labels = True
                        logger.info("Packet file found, enabling supervised learning mode")
                        #print(f"[DEBUG {client_id}] Auto-enabled supervised learning mode")

        # Validate that at least one data source is provided
        #print(f"[DEBUG {client_id}] Validating data files...")
        if not gps_file and not imu_file and not packet_file:
            error_msg = "At least one of gps_file, imu_file, or packet_file must be provided"
            #print(f"[DEBUG {client_id}] ERROR: {error_msg}")
            raise ValueError(error_msg)

        # Determine if we're doing supervised learning
        self.is_supervised = use_labels and packet_file is not None
        #print(f"[DEBUG {client_id}] Learning mode: {'Supervised' if self.is_supervised else 'Time Series Prediction'}")
        if self.is_supervised:
            logger.info("Mode: Supervised Learning (using packet labels)")
        else:
            logger.info("Mode: Time Series Prediction (next timestep)")

        # Preprocess data
        #print(f"[DEBUG {client_id}] Starting data preprocessing...")
        logger.info("Preprocessing data...")
        logger.info(f"Data files - GPS: {gps_file}, IMU: {imu_file}, Packet: {packet_file}")
        #print(f"[DEBUG {client_id}] Data files - GPS: {gps_file}, IMU: {imu_file}, Packet: {packet_file}")
        
        try:
            #print(f"[DEBUG {client_id}] Creating DataPreprocessorWithPackets...")
            preprocessor = DataPreprocessorClassification(
                gps_file=gps_file,
                imu_file=imu_file,
                packet_file=packet_file,
                gps_features=getattr(config, 'GPS_FEATURES', None),
                imu_features=getattr(config, 'IMU_FEATURES', None),
                packet_features=getattr(config, 'PACKET_FEATURES', None),
                timestamp_col=getattr(config, 'TIMESTAMP_COL', 'TimeUS'),
                packet_timestamp_col=getattr(config, 'PACKET_TIMESTAMP_COL', 'Timestamp'),
                label_col=getattr(config, 'LABEL_COL', 'Label'),
                sampling_strategy=getattr(config, 'SAMPLING_STRATEGY', 'downsample'),
                use_labels=self.is_supervised
            )
            #print(f"[DEBUG {client_id}] DataPreprocessor created, calling preprocess()...")
            logger.info("DataPreprocessor initialized, starting preprocessing...")
            self.X, self.y, self.num_features = preprocessor.preprocess(
                config.WINDOW_SIZE,
                config.OVERLAP
            )
            #print(f"[DEBUG {client_id}] Preprocessing complete! X shape: {self.X.shape}, y shape: {self.y.shape}, features: {self.num_features}")
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            #print(f"[DEBUG {client_id}] ERROR in preprocessing: {e}")
            logger.error(f"Data preprocessing failed: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            raise

        self.total_windows = len(self.X)
        #print(f"[DEBUG {client_id}] Total windows: {self.total_windows}")
        logger.info(f"Total windows available: {self.total_windows}")

        # Determine output size based on learning mode
        #print(f"[DEBUG {client_id}] Determining output size...")
        if self.is_supervised:
            # Binary classification: output size = 1 (or 2 for logits)
            output_size = 1
            # Check if y is binary (0/1) or already one-hot
            if len(self.y.shape) == 1:
                # Binary labels, model should output single value (logit)
                output_size = 1
            else:
                # One-hot or multi-class, use number of classes
                output_size = self.y.shape[1] if len(self.y.shape) > 1 else 1
        else:
            # Time series prediction: output size = num_features
            output_size = self.num_features

        # Load model dynamically from config
        #print(f"[DEBUG {client_id}] Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}...")
        logger.info(f"Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}")
        try:
            model_module = importlib.import_module(config.MODEL_PATH)
            model_class = getattr(model_module, config.MODEL_CLASS)
            #print(f"[DEBUG {client_id}] Model class loaded: {model_class}")
        except Exception as e:
            #print(f"[DEBUG {client_id}] ERROR loading model module: {e}")
            raise

        # Update model config with actual input and output sizes
        model_config = config.MODEL_CONFIG.copy()
        model_config['input_size'] = self.num_features
        model_config['output_size'] = output_size
        #print(f"[DEBUG {client_id}] Model config: {model_config}")

        # Initialize model
        #print(f"[DEBUG {client_id}] Creating model instance...")
        self.model = model_class(**model_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"[DEBUG {client_id}] Using device: {self.device}")
        self.model.to(self.device)

        #print(f"[DEBUG {client_id}] Model initialized successfully")
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        logger.info(f"Input size: {self.num_features}, Output size: {output_size}")

        # Training components - use appropriate loss function
        if self.is_supervised:
            # For binary classification
            if output_size == 1:
                # Binary classification with single output (sigmoid)
                # BCEWithLogitsLoss expects float targets (0.0 or 1.0)
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                # Multi-class classification
                # CrossEntropyLoss expects long targets (class indices)
                self.criterion = nn.CrossEntropyLoss()
        else:
            # Time series prediction (regression)
            self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Track current position
        self.current_window_idx = 0
        
        # Print detailed data information (after all initialization is complete)
        self._print_data_info()

    def _extract_client_number(self, client_id: str) -> Optional[int]:
        """Extract client number from client_id (e.g., 'client_1' -> 1, 'client_001' -> 1)."""
        try:
            # Try to extract number from client_id
            parts = client_id.split('_')
            if len(parts) > 1:
                return int(parts[-1]) - 1  # Convert to 0-indexed
            return None
        except (ValueError, IndexError):
            return None
    
    def _print_data_info(self):
        """Print detailed information about the training data."""
        import numpy as np
        
        print(f"\n{'='*70}")
        print(f"[{self.client_id}] TRAINING DATA INFORMATION")
        print(f"{'='*70}")
        
        # Basic statistics
        print(f"\nDataset Statistics:")
        print(f"  Total Windows: {self.total_windows}")
        print(f"  Window Size: {config.WINDOW_SIZE} timesteps")
        print(f"  Overlap: {config.OVERLAP} timesteps")
        print(f"  Features per timestep: {self.num_features}")
        print(f"  Input Shape: {self.X.shape}")
        print(f"  Target Shape: {self.y.shape}")
        
        # Learning mode
        print(f"\nLearning Mode:")
        if self.is_supervised:
            print(f"  Type: Supervised Learning (Classification)")
            print(f"  Task: Binary Classification")
            # Get output size from model if available, otherwise use a default
            try:
                output_size = self.model.output_size if hasattr(self.model, 'output_size') else 1
            except:
                output_size = 1
            print(f"  Output Size: {output_size}")
            
            # Label statistics
            if isinstance(self.y, np.ndarray):
                unique_labels, counts = np.unique(self.y, return_counts=True)
                print(f"\nLabel Distribution:")
                for label, count in zip(unique_labels, counts):
                    percentage = (count / len(self.y)) * 100
                    print(f"  Class {int(label)}: {count} samples ({percentage:.2f}%)")
                
                # Class imbalance check
                if len(unique_labels) == 2:
                    imbalance_ratio = counts[0] / counts[1] if counts[1] > 0 else float('inf')
                    print(f"  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
                    if imbalance_ratio > 10 or imbalance_ratio < 0.1:
                        print(f"  [WARNING] Significant class imbalance detected!")
        else:
            print(f"  Type: Time Series Prediction (Regression)")
            print(f"  Task: Next Timestep Prediction")
            print(f"  Output Size: {self.num_features}")
            
            # Regression statistics
            if isinstance(self.y, np.ndarray):
                print(f"\nTarget Statistics:")
                print(f"  Mean: {np.mean(self.y):.6f}")
                print(f"  Std: {np.std(self.y):.6f}")
                print(f"  Min: {np.min(self.y):.6f}")
                print(f"  Max: {np.max(self.y):.6f}")
        
        # Feature information
        print(f"\nFeature Information:")
        if hasattr(config, 'GPS_FEATURES') and config.GPS_FEATURES:
            print(f"  GPS Features ({len(config.GPS_FEATURES)}): {', '.join(config.GPS_FEATURES)}")
        if hasattr(config, 'IMU_FEATURES') and config.IMU_FEATURES:
            print(f"  IMU Features ({len(config.IMU_FEATURES)}): {', '.join(config.IMU_FEATURES)}")
        if hasattr(config, 'PACKET_FEATURES') and config.PACKET_FEATURES:
            print(f"  Packet Features ({len(config.PACKET_FEATURES)}): {', '.join(config.PACKET_FEATURES)}")
        
        # Training configuration
        print(f"\nTraining Configuration:")
        print(f"  Windows per Round: {config.WINDOWS_PER_ROUND}")
        print(f"  Expected Rounds: ~{(self.total_windows + config.WINDOWS_PER_ROUND - 1) // config.WINDOWS_PER_ROUND}")
        print(f"  Learning Rate: {config.LEARNING_RATE}")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        print(f"  Loss Function: {self.criterion.__class__.__name__}")
        
        print(f"{'='*70}\n")

    async def register(self, session: aiohttp.ClientSession) -> bool:
        """Register with server."""
        url = f"{self.server_url}/register"
        data = {"client_id": self.client_id}

        #print(f"[DEBUG {self.client_id}] Attempting to register with server at {url}...")
        try:
            async with session.post(url, json=data) as response:
                #print(f"[DEBUG {self.client_id}] Registration response status: {response.status}")
                result = await response.json()
                #print(f"[DEBUG {self.client_id}] Registration result: {result}")
                logger.info(f"Registration: {result['message']}")
                success = response.status == 200
                #print(f"[DEBUG {self.client_id}] Registration {'SUCCESS' if success else 'FAILED'}")
                return success
        except Exception as e:
            #print(f"[DEBUG {self.client_id}] Registration exception: {type(e).__name__}: {e}")
            logger.error(f"Registration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def get_global_model(self, session: aiohttp.ClientSession, round_num: int = 0) -> bool:
        """Download global model from server."""
        url = f"{self.server_url}/get_model"
        data = {"client_id": self.client_id, "round": round_num}

        #print(f"[DEBUG {self.client_id}] Requesting global model for round {round_num} from {url}...")
        try:
            # Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=180)  # 3 minutes max
            #print(f"[DEBUG {self.client_id}] Sending POST request (timeout: 180s)...")
            async with session.post(url, json=data, timeout=timeout) as response:
                #print(f"[DEBUG {self.client_id}] Response received, status: {response.status}")
                if response.status == 200:
                    #print(f"[DEBUG {self.client_id}] Reading weights bytes...")
                    weights_bytes = await response.read()
                    #print(f"[DEBUG {self.client_id}] Weights size: {len(weights_bytes)} bytes, deserializing...")
                    weights = pickle.loads(weights_bytes)
                    #print(f"[DEBUG {self.client_id}] Setting model weights...")
                    self.model.set_weights(weights)
                    #print(f"[DEBUG {self.client_id}] Global model received and set successfully")
                    logger.info(f"Received global model for round {round_num}")
                    return True
                else:
                    error_text = await response.text()
                    #print(f"[DEBUG {self.client_id}] Failed to get model: status {response.status}, {error_text}")
                    logger.error(f"Failed to get model: status {response.status}, {error_text}")
                    return False
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientError) as e:
            #print(f"[DEBUG {self.client_id}] Timeout/Error waiting for global model: {type(e).__name__}: {e}")
            logger.error(f"Timeout/Error waiting for global model (round {round_num}): {type(e).__name__}")
            return False
        except Exception as e:
            #print(f"[DEBUG {self.client_id}] Exception getting model: {type(e).__name__}: {e}")
            logger.error(f"Error getting model: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return False

    def train_on_windows(self, start_idx: int, end_idx: int) -> float:
        """Train on windows (single pass)."""
        #print(f"[DEBUG {self.client_id}] Training on windows {start_idx} to {end_idx}...")
        self.model.train()
        total_loss = 0.0
        num_windows = 0

        for window_idx in range(start_idx, end_idx):
            if window_idx >= self.total_windows:
                break

            # Get window
            X_window = torch.FloatTensor(self.X[window_idx:window_idx+1]).to(self.device)
            y_window = self.y[window_idx:window_idx+1]

            # Prepare target based on learning mode
            if self.is_supervised:
                # Classification: convert to appropriate tensor format
                y_window_array = y_window if isinstance(y_window, np.ndarray) else np.array(y_window)
                
                # Handle different label formats
                if y_window_array.ndim == 0:
                    # Scalar
                    label_value = float(y_window_array.item())
                elif y_window_array.ndim == 1:
                    # 1D array - take first element
                    label_value = float(y_window_array[0])
                else:
                    # Multi-dimensional - flatten and take first
                    label_value = float(y_window_array.flatten()[0])
                
                # For binary classification with BCEWithLogitsLoss, use FloatTensor
                if self.model.output_size == 1:
                    # Keep as 1D tensor [label_value] to match model output shape
                    y_tensor = torch.FloatTensor([label_value]).to(self.device)
                else:
                    # Multi-class: use LongTensor
                    y_tensor = torch.LongTensor([int(label_value)]).to(self.device)
            else:
                # Regression: next timestep prediction
                y_window_array = y_window if isinstance(y_window, np.ndarray) else np.array(y_window)
                y_tensor = torch.FloatTensor(y_window_array).to(self.device)
                # Squeeze if needed to match output shape
                if y_tensor.dim() > 1:
                    y_tensor = y_tensor.squeeze()

            # Train
            self.optimizer.zero_grad()
            output = self.model(X_window)
            
            # Handle output shape for classification
            if self.is_supervised:
                # For binary classification with single output
                # Ensure output and target have matching shapes
                # Both should be 1D tensors with 1 element for BCEWithLogitsLoss
                
                # Normalize output to 1D
                if output.dim() == 0:
                    # Scalar -> make it [1]
                    output = output.unsqueeze(0)
                elif output.dim() > 1:
                    # Multi-dim -> squeeze to 1D
                    output = output.squeeze()
                    if output.dim() == 0:
                        output = output.unsqueeze(0)
                
                # Normalize target to 1D
                if y_tensor.dim() == 0:
                    # Scalar -> make it [1]
                    y_tensor = y_tensor.unsqueeze(0)
                elif y_tensor.dim() > 1:
                    # Multi-dim -> squeeze to 1D
                    y_tensor = y_tensor.squeeze()
                    if y_tensor.dim() == 0:
                        y_tensor = y_tensor.unsqueeze(0)
                
                # Final check: both should be 1D with same size
                if output.dim() != y_tensor.dim() or output.size(0) != y_tensor.size(0):
                    # If sizes don't match, try to align
                    if output.size(0) == 1 and y_tensor.size(0) == 1:
                        # Both are [1], that's fine
                        pass
                    elif output.size(0) == 1:
                        # Output is [1], target might be different - squeeze output to match
                        if y_tensor.dim() == 0:
                            output = output.squeeze(0)
                    elif y_tensor.size(0) == 1:
                        # Target is [1], output might be different - squeeze target to match
                        if output.dim() == 0:
                            y_tensor = y_tensor.squeeze(0)
            else:
                # For regression, ensure output matches target shape
                if output.dim() > 2:
                    output = output.squeeze()
                # Ensure output and target have same shape
                if output.shape != y_tensor.shape:
                    # Try to reshape output to match target
                    if output.numel() == y_tensor.numel():
                        output = output.view(y_tensor.shape)

            loss = self.criterion(output, y_tensor)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_windows += 1

        avg_loss = total_loss / num_windows if num_windows > 0 else 0.0
        #print(f"[DEBUG {self.client_id}] Training complete: {num_windows} windows, avg loss: {avg_loss:.6f}")
        return avg_loss

    async def submit_update(self, session: aiohttp.ClientSession, loss: float) -> bool:
        """Submit model update to server."""
        url = f"{self.server_url}/submit_update"

        #print(f"[DEBUG {self.client_id}] Preparing to submit update (loss: {loss:.6f})...")
        weights = self.model.get_weights()
        #print(f"[DEBUG {self.client_id}] Got model weights, serializing...")
        weights_bytes = pickle.dumps(weights)
        #print(f"[DEBUG {self.client_id}] Weights serialized: {len(weights_bytes)} bytes")

        headers = {
            "X-Client-ID": self.client_id,
            "X-Num-Samples": str(config.WINDOWS_PER_ROUND),
            "X-Loss": str(loss),
            "Content-Type": "application/octet-stream"
        }
        
        # Add learning mode header if supervised
        if self.is_supervised:
            headers["X-Learning-Mode"] = "classification"
        else:
            headers["X-Learning-Mode"] = "regression"

        #print(f"[DEBUG {self.client_id}] Submitting update to {url}...")
        try:
            # Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=60)  # 1 minute max
            async with session.post(url, data=weights_bytes, headers=headers, timeout=timeout) as response:
                #print(f"[DEBUG {self.client_id}] Update response status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    #print(f"[DEBUG {self.client_id}] Update submitted successfully: {result}")
                    logger.info(f"Update submitted: {result.get('status', 'unknown')}")
                    return True
                else:
                    error_text = await response.text()
                    #print(f"[DEBUG {self.client_id}] Update submission failed: status {response.status}, {error_text}")
                    logger.error(f"Failed to submit update: status {response.status}, {error_text}")
                    return False
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientError) as e:
            #print(f"[DEBUG {self.client_id}] Timeout/Error submitting update: {type(e).__name__}: {e}")
            logger.error(f"Timeout/Error submitting update (loss: {loss:.6f}): {type(e).__name__}")
            return False
        except Exception as e:
            #print(f"[DEBUG {self.client_id}] Exception submitting update: {type(e).__name__}: {e}")
            logger.error(f"Error submitting: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return False

    async def run_federated_learning(self):
        """Run FL training."""
        #print(f"[DEBUG {self.client_id}] Starting run_federated_learning()...")
        async with aiohttp.ClientSession() as session:
            #print(f"[DEBUG {self.client_id}] ClientSession created")
            # Register
            #print(f"[DEBUG {self.client_id}] Registering with server...")
            logger.info(f"Registering {self.client_id}...")
            if not await self.register(session):
                #print(f"[DEBUG {self.client_id}] Registration FAILED - exiting")
                logger.error(f"{self.client_id} failed to register - exiting")
                return
            #print(f"[DEBUG {self.client_id}] Registration successful, waiting 3 seconds...")

            await asyncio.sleep(3)

            # Training rounds
            round_num = 0
            max_rounds = (self.total_windows + config.WINDOWS_PER_ROUND - 1) // config.WINDOWS_PER_ROUND
            #print(f"[DEBUG {self.client_id}] Starting training: {self.total_windows} windows, ~{max_rounds} rounds expected")
            logger.info(f"Starting training: {self.total_windows} windows, ~{max_rounds} rounds expected")
            
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            #print(f"[DEBUG {self.client_id}] Entering training loop (current_window_idx: {self.current_window_idx}, total_windows: {self.total_windows})...")
            while self.current_window_idx < self.total_windows:
                round_num += 1
                #print(f"[DEBUG {self.client_id}] ===== Round {round_num} =====")
                
                # Safety check to prevent infinite loops
                if round_num > max_rounds * 2:
                    #print(f"[DEBUG {self.client_id}] ERROR: Exceeded expected rounds ({max_rounds * 2}) - exiting")
                    logger.error(f"{self.client_id} exceeded expected rounds ({max_rounds * 2}) - exiting")
                    break
                
                end_idx = min(
                    self.current_window_idx + config.WINDOWS_PER_ROUND,
                    self.total_windows
                )

                #print(f"[DEBUG {self.client_id}] Round {round_num}: Processing windows {self.current_window_idx}-{end_idx} ({end_idx - self.current_window_idx} windows)")
                logger.info(f"Round {round_num}: Windows {self.current_window_idx}-{end_idx} "
                           f"({end_idx - self.current_window_idx} windows)")

                # Get global model (skip first round)
                if round_num > 1:
                    #print(f"[DEBUG {self.client_id}] Round {round_num} > 1, requesting global model for round {round_num - 1}...")
                    logger.info(f"Requesting global model for round {round_num - 1}...")
                    if not await self.get_global_model(session, round_num - 1):
                        consecutive_failures += 1
                        #print(f"[DEBUG {self.client_id}] Failed to get global model (failure {consecutive_failures}/{max_consecutive_failures})")
                        logger.warning(f"Failed to get global model (failure {consecutive_failures}/{max_consecutive_failures})")
                        if consecutive_failures >= max_consecutive_failures:
                            #print(f"[DEBUG {self.client_id}] Too many failures - exiting")
                            logger.error(f"{self.client_id} too many failures - exiting")
                            break
                        # Wait a bit before retrying
                        #print(f"[DEBUG {self.client_id}] Waiting 5 seconds before retry...")
                        await asyncio.sleep(5)
                        continue
                    consecutive_failures = 0  # Reset on success
                    print(f"[DEBUG {self.client_id}] Global model received successfully")
                else:
                    print(f"[DEBUG {self.client_id}] Round 1, skipping global model request")

                # Train
                #print(f"[DEBUG {self.client_id}] Starting training...")
                logger.info(f"Training on windows {self.current_window_idx}-{end_idx}...")
                loss = self.train_on_windows(self.current_window_idx, end_idx)
                mode_str = "Classification" if self.is_supervised else "Regression"
                #print(f"[DEBUG {self.client_id}] Training complete, loss: {loss:.6f}")
                logger.info(f"Round {round_num} {mode_str} Loss: {loss:.6f}")

                # Submit
                #print(f"[DEBUG {self.client_id}] Submitting update...")
                logger.info(f"Submitting update for round {round_num}...")
                if not await self.submit_update(session, loss):
                    consecutive_failures += 1
                    #print(f"[DEBUG {self.client_id}] Failed to submit update (failure {consecutive_failures}/{max_consecutive_failures})")
                    logger.warning(f"Failed to submit update (failure {consecutive_failures}/{max_consecutive_failures})")
                    if consecutive_failures >= max_consecutive_failures:
                        #print(f"[DEBUG {self.client_id}] Too many failures - exiting")
                        logger.error(f"{self.client_id} too many failures - exiting")
                        break
                    # Wait a bit before retrying
                    #print(f"[DEBUG {self.client_id}] Waiting 5 seconds before retry...")
                    await asyncio.sleep(5)
                    continue
                consecutive_failures = 0  # Reset on success
                #print(f"[DEBUG {self.client_id}] Update submitted successfully")

                self.current_window_idx = end_idx
                #print(f"[DEBUG {self.client_id}] Round {round_num} complete. Progress: {self.current_window_idx}/{self.total_windows} windows")
                logger.info(f"Round {round_num} complete. Progress: {self.current_window_idx}/{self.total_windows} windows")
                await asyncio.sleep(1)

            if self.current_window_idx >= self.total_windows:
                #print(f"[DEBUG {self.client_id}] ✓ Training completed! {round_num} rounds, all {self.total_windows} windows processed.")
                logger.info(f"✓ {self.client_id} completed {round_num} rounds! Processed all {self.total_windows} windows.")
            else:
                #print(f"[DEBUG {self.client_id}] ⚠ Training exited early after {round_num} rounds. Processed {self.current_window_idx}/{self.total_windows} windows.")
                logger.warning(f"⚠ {self.client_id} exited early after {round_num} rounds. "
                             f"Processed {self.current_window_idx}/{self.total_windows} windows.")


def main():
    """Main entry point."""
    import argparse

    #print(f"[DEBUG] Client process starting...")
    #print(f"[DEBUG] Python version: {sys.version}")
    #print(f"[DEBUG] Working directory: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='Federated Learning Client with Packet Support')
    parser.add_argument("--client-id", required=True, help="Client identifier (e.g., client_1)")
    parser.add_argument("--server-url", required=True, help="Server URL")
    parser.add_argument("--gps-file", default=None, help="Path to GPS CSV file (optional)")
    parser.add_argument("--imu-file", default=None, help="Path to IMU CSV file (optional)")
    parser.add_argument("--packet-file", default=None, help="Path to labeled packet CSV file (optional)")
    parser.add_argument("--data-dir", default=None, help="Directory with client-specific data files from prepare_fl_data_for_run_fl.py")
    parser.add_argument("--use-labels", action='store_true', help="Use packet labels for supervised learning (auto-enabled if packet file provided)")
    args = parser.parse_args()

    #print(f"[DEBUG] Parsed arguments:")
    #print(f"  client_id: {args.client_id}")
    #print(f"  server_url: {args.server_url}")
    #print(f"  gps_file: {args.gps_file}")
    #print(f"  imu_file: {args.imu_file}")
    #print(f"  packet_file: {args.packet_file}")
    #print(f"  data_dir: {args.data_dir}")
    #print(f"  use_labels: {args.use_labels}")

    #print(f"[DEBUG] Creating FLClientClassification instance...")
    try:
        client = FLClientClassification(
            client_id=args.client_id,
            server_url=args.server_url,
            gps_file=args.gps_file,
            imu_file=args.imu_file,
            packet_file=args.packet_file,
            use_labels=args.use_labels,
            data_dir=args.data_dir
        )
        #print(f"[DEBUG] Client instance created successfully")
    except Exception as e:
        #print(f"[DEBUG] ERROR creating client instance: {e}")
        import traceback
        traceback.print_exc()
        raise

    #print(f"[DEBUG] Starting federated learning loop...")
    try:
        asyncio.run(client.run_federated_learning())
        #print(f"[DEBUG] Federated learning loop completed")
    except Exception as e:
        #print(f"[DEBUG] ERROR in federated learning loop: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
