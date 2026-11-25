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
from typing import Dict, Optional

# Add fl-time-series (script dir) first so models/ is found here, not at project root
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from agent_mission_data import (
    build_agent_train_windows,
    list_mission_files_for_agent,
)
from data_preprocessing_time_series import DataPreprocessorTimeSeries
from fl_reproducibility import get_fl_seed, set_fl_seed

# -----------------------------------------------------------------------------
# Dynamic config loading
# -----------------------------------------------------------------------------
# Allow selecting the config module at runtime so that server and clients share
# the same configuration (e.g., "config" for GPS/IMU regression or
# "config_packets_only" for packet-only time-series).
#
# The module name is taken from the FL_CONFIG_MODULE environment variable when
# present, otherwise:
# - we first try "config_classification"
# - then fall back to "config".
CONFIG_MODULE = os.environ.get("FL_CONFIG_MODULE")
if CONFIG_MODULE is None:
    try:
        CONFIG_MODULE = "config_classification"
        config = importlib.import_module(CONFIG_MODULE)
    except ImportError:
        CONFIG_MODULE = "config"
        config = importlib.import_module(CONFIG_MODULE)
else:
    config = importlib.import_module(CONFIG_MODULE)

# Re-export UPPERCASE attributes from the selected config into this module's
# global namespace so existing code that references SERVER_HOST, SERVER_PORT,
# etc. continues to work unchanged.
for _name, _value in vars(config).items():
    if _name.isupper():
        globals()[_name] = _value

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class FLClientTimeSeries:
    """Federated Learning Client for Time-Series (classification or regression) Task."""

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

        # ACFL attack-injection hooks (used for poisoning evaluation only).
        # Configured via environment variables so the upstream CLI is unchanged.
        self._last_global_weights = None

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
                mission_paths = list_mission_files_for_agent(data_dir, client_num)
                
                #print(f"[DEBUG {client_id}] Checking for client files:")
                print(f"  GPS: {gps_path} (exists: {os.path.exists(gps_path)})")
                print(f"  IMU: {imu_path} (exists: {os.path.exists(imu_path)})")
                print(f"  Missions: {len(mission_paths)} file(s)")
                for mp in mission_paths[:3]:
                    print(f"    {mp}")
                if len(mission_paths) > 3:
                    print(f"    ... +{len(mission_paths) - 3} more")
                
                if os.path.exists(gps_path):
                    gps_file = gps_file or gps_path
                    logger.info(f"Found client GPS file: {gps_path}")
                    #print(f"[DEBUG {client_id}] Using GPS file: {gps_path}")
                if os.path.exists(imu_path):
                    imu_file = imu_file or imu_path
                    logger.info(f"Found client IMU file: {imu_path}")
                    #print(f"[DEBUG {client_id}] Using IMU file: {imu_path}")
                if os.path.exists(packet_path) or mission_paths:
                    packet_file = packet_file or (mission_paths[0] if mission_paths else packet_path)
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
            client_num = self._extract_client_number(client_id)
            mission_paths = (
                list_mission_files_for_agent(data_dir, client_num)
                if data_dir and client_num is not None
                else []
            )
            if self.is_supervised and data_dir and mission_paths:
                logger.info(
                    "Loading %d mission(s) for %s (per-mission temporal train split)",
                    len(mission_paths), client_id,
                )
                seed = int(getattr(config, "OVERSAMPLE_SEED", 42))
                if client_num is not None:
                    seed += client_num
                self.X, self.y, self.num_features, self.mission_stats = build_agent_train_windows(
                    data_dir,
                    client_num,
                    config,
                    oversample=True,
                    seed=seed,
                )
                self.temporal_split_stats = {
                    "mode": "per_mission_temporal",
                    "missions": self.mission_stats,
                }
                self.window_oversample_stats = next(
                    (m.get("oversample") for m in self.mission_stats if "oversample" in m),
                    None,
                )
                logger.info("Data preprocessing completed successfully")
            else:
                preprocessor = DataPreprocessorTimeSeries(
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
                logger.info("DataPreprocessor initialized, starting preprocessing...")
                from data_preprocessing_time_series import oversample_attack_windows, split_temporal_windows
                test_split = float(getattr(config, "TEST_SPLIT", 0.2))
                use_temporal_split = bool(getattr(config, "TEMPORAL_TRAIN_SPLIT", False))
                norm_train_fraction = (1.0 - test_split) if use_temporal_split else None
                self.X, self.y, self.num_features = preprocessor.preprocess(
                    config.WINDOW_SIZE,
                    config.OVERLAP,
                    norm_train_fraction=norm_train_fraction,
                )
                self.mission_stats = None
                self.temporal_split_stats = None
                if use_temporal_split:
                    self.X, self.y, _, _, self.temporal_split_stats = split_temporal_windows(
                        self.X, self.y, test_split=test_split,
                    )
                self.window_oversample_stats = None
                if self.is_supervised and getattr(config, 'OVERSAMPLE_ATTACK_WINDOWS', False):
                    seed = int(getattr(config, 'OVERSAMPLE_SEED', 42))
                    if client_num is not None:
                        seed += client_num
                    self.X, self.y, self.window_oversample_stats = oversample_attack_windows(
                        self.X,
                        self.y,
                        target_attack_ratio=getattr(config, 'TARGET_ATTACK_WINDOW_RATIO', 0.25),
                        seed=seed,
                    )
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

        self.max_fl_rounds = int(os.environ.get("FL_MAX_ROUNDS", "0")) or None
        self.mission_context_stats = self._compute_mission_context(
            packet_file, gps_file, imu_file
        )

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

        # Store sizes on the instance for later use (e.g., checkpoint metadata)
        self.output_size = int(output_size)
        self.input_size = int(self.num_features)
        
        # Load model dynamically from config
        #print(f"[DEBUG {client_id}] Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}...")
        logger.info(f"Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}")
        client_num = self._extract_client_number(client_id) or 0
        base_seed = get_fl_seed()
        if base_seed is not None:
            set_fl_seed(base_seed + client_num)
            self.fl_seed = base_seed + client_num
        else:
            self.fl_seed = None
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
                labels = np.asarray(self.y).reshape(-1)
                pos = float((labels >= 0.5).sum())
                neg = float(len(labels) - pos)
                if pos > 0 and neg > 0:
                    pos_weight = torch.tensor([neg / pos], device=self.device)
                    self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    logger.info(
                        "Class-balanced BCE loss (pos_weight=%.2f, attack windows=%.1f%%)",
                        pos_weight.item(), 100.0 * pos / len(labels),
                    )
                else:
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

        self._setup_acfl_attack()
        
        # Print detailed data information (after all initialization is complete)
        self._print_data_info()

    def _compute_mission_context(
        self,
        packet_file: Optional[str],
        gps_file: Optional[str],
        imu_file: Optional[str],
    ) -> Dict[str, float]:
        """Derive Layer-2 mission-context features from local labeled data."""
        attack_ratio = 0.0
        if self.is_supervised and self.y is not None:
            labels = np.asarray(self.y).reshape(-1)
            attack_ratio = float((labels >= 0.5).mean()) if len(labels) else 0.0
        elif packet_file and os.path.isfile(packet_file):
            import pandas as pd
            df = pd.read_csv(packet_file, usecols=["Label"])
            attack_ratio = float(df["Label"].mean())

        benign = float(np.clip(1.0 - attack_ratio * 3.0, 0.0, 1.0))
        return {
            "gnss_imu_consistency": benign,
            "mavlink_state_consistency": benign,
            "link_health": float(np.clip(1.0 - attack_ratio * 2.0, 0.0, 1.0)),
            "attack_label_ratio": attack_ratio,
        }

    def _mission_context_headers(self) -> Dict[str, str]:
        ctx = getattr(self, "mission_context_stats", {}) or {}
        return {
            "X-GNSS-IMU-Consistency": f"{ctx.get('gnss_imu_consistency', 1.0):.4f}",
            "X-MAVLink-Consistency": f"{ctx.get('mavlink_state_consistency', 1.0):.4f}",
            "X-Link-Health": f"{ctx.get('link_health', 1.0):.4f}",
            "X-Attack-Ratio": f"{ctx.get('attack_label_ratio', 0.0):.4f}",
        }

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
        if getattr(self, "temporal_split_stats", None) and self.temporal_split_stats.get("applied"):
            ts = self.temporal_split_stats
            print(f"  Temporal Split: train {ts['windows_train']} / "
                  f"held-out eval {ts['windows_test']} (of {ts['windows_total']} total)")
            print(f"  Train attack ratio: {100.0 * ts['train_attack_ratio']:.1f}%")
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

    def _setup_acfl_attack(self):
        """Configure poisoning attack from environment variables (ACFL evaluation).

        Recognized variables:
          ACFL_ATTACK_CLIENTS : comma-separated client IDs that are malicious
          ACFL_ATTACK_TYPE    : gradient_scaling | byzantine | backdoor |
                                model_replacement | label_flip | none
          ACFL_ATTACK_SCALE   : scale factor for scaling/replacement attacks
          ACFL_ATTACK_NOISE   : std-dev for byzantine noise
          ACFL_BACKDOOR_POISON_FRACTION : fraction of benign windows to poison
          ACFL_BACKDOOR_TRIGGER_VALUE   : trigger magnitude (matches probes)
          ACFL_BACKDOOR_TARGET_LABEL    : label for poisoned windows (0=benign)
          ACFL_LABEL_FLIP_FRACTION      : fraction of local labels to flip
        """
        attack_clients = {
            c.strip()
            for c in os.environ.get("ACFL_ATTACK_CLIENTS", "").split(",")
            if c.strip()
        }
        self.is_malicious = self.client_id in attack_clients
        self.attack_type = os.environ.get("ACFL_ATTACK_TYPE", "none")
        self.attack_simulator = None
        self._current_fl_round = 0

        if not self.is_malicious or self.attack_type == "none":
            return

        try:
            acfl_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if acfl_root not in sys.path:
                sys.path.insert(0, acfl_root)
            from acfl.attacks import AttackType, ClientAttackSimulator

            client_num = self._extract_client_number(self.client_id) or 0
            attack_seed_base = (get_fl_seed(42) or 42) + client_num

            self.attack_simulator = ClientAttackSimulator(
                attack_type=AttackType(self.attack_type),
                scale_factor=float(os.environ.get("ACFL_ATTACK_SCALE", "10.0")),
                noise_std=float(os.environ.get("ACFL_ATTACK_NOISE", "1.0")),
                backdoor_scale=float(os.environ.get("ACFL_ATTACK_SCALE", "5.0")),
                backdoor_data_poison_only=self.attack_type == "backdoor",
                attack_seed_base=attack_seed_base,
            )
            if self.attack_type == "backdoor":
                self._apply_backdoor_data_poison()
            elif self.attack_type == "label_flip":
                self._apply_label_flip_data_poison()
            logger.warning(
                f"[ACFL-ATTACK] {self.client_id} configured as MALICIOUS "
                f"(attack={self.attack_type})"
            )
        except Exception as e:
            logger.error(f"[ACFL-ATTACK] Failed to set up attack: {e}")
            self.attack_simulator = None

    def _apply_backdoor_data_poison(self):
        """Stamp probe-aligned triggers into benign training windows."""
        if not self.is_supervised or self.X is None or self.y is None:
            logger.warning(
                "[ACFL-ATTACK] %s backdoor skipped: supervised packet data required",
                self.client_id,
            )
            return

        acfl_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if acfl_root not in sys.path:
            sys.path.insert(0, acfl_root)
        from acfl.backdoor_data import poison_training_windows

        client_num = self._extract_client_number(self.client_id) or 0
        seed = (get_fl_seed(42) or 42) + client_num
        fraction = float(os.environ.get("ACFL_BACKDOOR_POISON_FRACTION", "0.35"))
        trigger_value = float(os.environ.get("ACFL_BACKDOOR_TRIGGER_VALUE", "3.0"))
        target_label = float(os.environ.get("ACFL_BACKDOOR_TARGET_LABEL", "0.0"))

        self.X, self.y, stats = poison_training_windows(
            self.X,
            self.y,
            poison_fraction=fraction,
            target_label=target_label,
            trigger_value=trigger_value,
            seed=seed,
        )
        self.backdoor_poison_stats = stats
        logger.warning(
            "[ACFL-ATTACK] %s backdoor data poison: %d/%d benign windows "
            "(target_label=%.0f, trigger=%.1f)",
            self.client_id,
            stats["num_poisoned"],
            stats.get("benign_windows", 0),
            target_label,
            trigger_value,
        )

    def _apply_label_flip_data_poison(self):
        """Flip supervised labels on a fraction of local training windows."""
        if not self.is_supervised or self.X is None or self.y is None:
            logger.warning(
                "[ACFL-ATTACK] %s label_flip skipped: supervised packet data required",
                self.client_id,
            )
            return

        acfl_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if acfl_root not in sys.path:
            sys.path.insert(0, acfl_root)
        from acfl.label_flip_data import flip_training_labels

        client_num = self._extract_client_number(self.client_id) or 0
        seed = (get_fl_seed(42) or 42) + client_num
        fraction = float(os.environ.get("ACFL_LABEL_FLIP_FRACTION", "0.5"))

        self.X, self.y, stats = flip_training_labels(
            self.X,
            self.y,
            flip_fraction=fraction,
            seed=seed,
        )
        self.label_flip_stats = stats
        logger.warning(
            "[ACFL-ATTACK] %s label_flip: %d/%d windows flipped",
            self.client_id,
            stats["num_flipped"],
            stats.get("total_windows", 0),
        )

    def _apply_acfl_attack(self, weights):
        """Apply the configured poisoning attack to outgoing weights."""
        if not getattr(self, "is_malicious", False) or self.attack_simulator is None:
            return weights
        global_w = self._last_global_weights or weights
        attacked = self.attack_simulator.attack_update(
            weights,
            global_w,
            round_number=getattr(self, "_current_fl_round", 0),
        )
        logger.warning(
            f"[ACFL-ATTACK] {self.client_id} poisoned its update "
            f"({self.attack_type})"
        )
        return attacked

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
                    from fl_comm import deserialize_weights, is_gzip_payload
                    weights = deserialize_weights(
                        weights_bytes, compressed=is_gzip_payload(weights_bytes),
                    )
                    self.model.set_weights(weights)
                    # Snapshot global weights so malicious clients can craft
                    # delta-based poisoning relative to the current global model.
                    if getattr(self, "is_malicious", False):
                        self._last_global_weights = {
                            k: v.clone() for k, v in weights.items()
                        }
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

    def train_on_windows(
        self, start_idx: int, end_idx: int, round_num: int = 0
    ) -> float:
        """Train on windows with mini-batch SGD and multiple local epochs."""
        self.model.train()
        indices = list(range(start_idx, min(end_idx, self.total_windows)))
        if not indices:
            return 0.0

        local_epochs = max(1, int(getattr(config, "LOCAL_EPOCHS", 1)))
        batch_size = max(1, int(getattr(config, "BATCH_SIZE", 32)))
        total_loss = 0.0
        num_batches = 0
        shuffle_seed = None
        if self.fl_seed is not None:
            shuffle_seed = int(self.fl_seed) + int(round_num) * 1000 + int(start_idx)

        for epoch in range(local_epochs):
            order = indices.copy()
            if epoch > 0 and shuffle_seed is not None:
                rng = np.random.default_rng(shuffle_seed + epoch)
                rng.shuffle(order)
            elif epoch > 0:
                np.random.shuffle(order)

            for batch_start in range(0, len(order), batch_size):
                batch_idx = order[batch_start: batch_start + batch_size]
                X_batch = torch.FloatTensor(self.X[batch_idx]).to(self.device)
                y_batch = np.asarray(self.y[batch_idx]).reshape(-1)

                if self.is_supervised:
                    if self.model.output_size == 1:
                        y_tensor = torch.FloatTensor(y_batch.astype(np.float32)).to(self.device)
                    else:
                        y_tensor = torch.LongTensor(y_batch.astype(np.int64)).to(self.device)
                else:
                    y_tensor = torch.FloatTensor(y_batch).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)

                if self.is_supervised and self.model.output_size == 1:
                    if output.dim() > 1:
                        output = output.squeeze(-1)
                    if y_tensor.dim() == 0:
                        y_tensor = y_tensor.unsqueeze(0)
                elif not self.is_supervised:
                    if output.dim() > 2:
                        output = output.squeeze()
                    if output.shape != y_tensor.shape and output.numel() == y_tensor.numel():
                        output = output.view(y_tensor.shape)

                loss = self.criterion(output, y_tensor)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    async def submit_update(self, session: aiohttp.ClientSession, loss: float,
                            session_complete: bool = False) -> bool:
        """Submit model update to server."""
        url = f"{self.server_url}/submit_update"

        #print(f"[DEBUG {self.client_id}] Preparing to submit update (loss: {loss:.6f})...")
        weights = self.model.get_weights()
        # ACFL evaluation: malicious clients poison their update here.
        weights = self._apply_acfl_attack(weights)
        from fl_comm import serialize_weights, transfer_settings_from_config
        use_fp16, use_gzip = transfer_settings_from_config(config)
        weights_bytes = serialize_weights(
            weights, use_fp16=use_fp16, compress=use_gzip,
        )

        headers = {
            "X-Client-ID": self.client_id,
            "X-Num-Samples": str(config.WINDOWS_PER_ROUND),
            "X-Loss": str(loss),
            "Content-Type": "application/octet-stream",
            **self._mission_context_headers(),
        }
        
        if session_complete:
            headers["X-Session-Complete"] = "1"
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
            
    def save_checkpoint(self, round_num: int, loss: float):
        """Save client model checkpoint."""
        if not getattr(config, "SAVE_CLIENT_CHECKPOINTS", True):
            return
        try:
            checkpoint_dir = getattr(config, 'CHECKPOINT_DIR', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Create client-specific checkpoint directory
            client_checkpoint_dir = os.path.join(checkpoint_dir, 'clients')
            os.makedirs(client_checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(
                client_checkpoint_dir, 
                f"{self.client_id}_round_{round_num}.pt"
            )
            
            checkpoint = {
                "round": round_num,
                "client_id": self.client_id,
                "model_state_dict": self.model.state_dict(),
                "loss": loss,
                "learning_mode": "classification" if self.is_supervised else "regression",
                "output_size": self.output_size,
                "input_size": self.num_features,
                "current_window_idx": self.current_window_idx,
                "total_windows": self.total_windows
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Client checkpoint saved: {checkpoint_path}")
            print(f"[DEBUG {self.client_id}] Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save client checkpoint: {e}")
            print(f"[DEBUG {self.client_id}] Warning: Failed to save checkpoint: {e}")

    
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
                if self.max_fl_rounds and round_num > self.max_fl_rounds:
                    logger.info(
                        "%s reached FL_MAX_ROUNDS=%d — stopping",
                        self.client_id, self.max_fl_rounds,
                    )
                    break
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
                loss = self.train_on_windows(
                    self.current_window_idx, end_idx, round_num=round_num
                )
                mode_str = "Classification" if self.is_supervised else "Regression"
                #print(f"[DEBUG {self.client_id}] Training complete, loss: {loss:.6f}")
                logger.info(f"Round {round_num} {mode_str} Loss: {loss:.6f}")

                # Submit
                #print(f"[DEBUG {self.client_id}] Submitting update...")
                logger.info(f"Submitting update for round {round_num}...")
                self._current_fl_round = round_num
                done = self.current_window_idx + (end_idx - self.current_window_idx) >= self.total_windows
                if not await self.submit_update(session, loss, session_complete=done):
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

                # Save client checkpoint after training
                self.save_checkpoint(round_num, loss)
                
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

    #print(f"[DEBUG] Creating FLClientTimeSeries instance...")
    try:
        client = FLClientTimeSeries(
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
