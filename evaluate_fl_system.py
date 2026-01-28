"""
Comprehensive Evaluation Script for Federated Learning System

This script evaluates the FL system for both classification and regression tasks.
It supports:
- Loading trained models from checkpoints
- Evaluating on test data
- Computing task-specific metrics
- Generating comprehensive visualizations
- Comparing performance across clients
- Analyzing convergence and generalization

Usage:
    python evaluate_fl_system.py --checkpoint-dir checkpoints --data-dir data/prepared_clients --task classification
    python evaluate_fl_system.py --checkpoint-dir checkpoints --data-dir data/prepared_clients --task regression
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from collections import defaultdict
import json
import glob
import importlib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)

# NOTE: The actual config module is selected at runtime in main() based on a
# --config argument. For type checkers, we hint that the symbols come from
# config_packets_only, but at runtime `config` will be whatever module is
# imported there (e.g., config_packets_only, config_regression, config, ...).
if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from config_classification import *  # type: ignore[import,unused-wildcard-import]

config = None  # Will be set in main()
from data_preprocessing_with_packets import DataPreprocessorWithPackets
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


class FLEvaluator:
    """Comprehensive evaluator for Federated Learning system."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        data_dir: Optional[str] = None,
        task: str = 'auto',
        test_split: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize FL Evaluator.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            data_dir: Directory with client data files (for test data)
            task: 'classification', 'regression', or 'auto' (auto-detect)
            test_split: Proportion of data to use for testing (if test data not available)
            device: Device to use ('cuda' or 'cpu', None = auto-detect)
        """
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.task = task
        self.test_split = test_split
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load model architecture
        self.model_module = importlib.import_module(config.MODEL_PATH)
        self.model_class = getattr(self.model_module, config.MODEL_CLASS)
        
        # Storage for results
        self.checkpoints = {}
        self.test_data = {}
        self.evaluation_results = {}
        self.client_evaluation_results = {}  # Per-client evaluation results
        self.metrics_history = defaultdict(list)
        
        print(f"Initialized FLEvaluator")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Data dir: {data_dir}")
        print(f"  Task: {task}")
        print(f"  Device: {self.device}")
    
    def load_checkpoints(self) -> Dict[int, Dict]:
        """Load all checkpoints from checkpoint directory."""
        print("\n" + "="*70)
        print("LOADING CHECKPOINTS")
        print("="*70)
        
        checkpoint_files = sorted(glob.glob(f"{self.checkpoint_dir}/server_round_*.pt"))
        
        if not checkpoint_files:
            raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")
        
        print(f"Found {len(checkpoint_files)} checkpoint files")
        
        checkpoints = {}
        for ckpt_file in checkpoint_files:
            try:
                # Extract round number from filename
                round_num = int(Path(ckpt_file).stem.split('_')[-1])
                
                checkpoint = torch.load(ckpt_file, map_location=self.device)
                checkpoints[round_num] = checkpoint
                
                print(f"  Round {round_num}: {Path(ckpt_file).name}")
                if 'learning_mode' in checkpoint:
                    print(f"    Learning mode: {checkpoint['learning_mode']}")
                if 'output_size' in checkpoint:
                    print(f"    Output size: {checkpoint['output_size']}")
            except Exception as e:
                print(f"  Warning: Failed to load {ckpt_file}: {e}")
        
        self.checkpoints = checkpoints
        print(f"\n✓ Loaded {len(checkpoints)} checkpoints")
        
        # Determine task if auto
        if self.task == 'auto' and checkpoints:
            first_ckpt = checkpoints[min(checkpoints.keys())]
            if 'learning_mode' in first_ckpt:
                self.task = first_ckpt['learning_mode']
            elif 'output_size' in first_ckpt:
                # Infer from output size
                if first_ckpt['output_size'] == 1:
                    self.task = 'classification'
                else:
                    self.task = 'regression'
            else:
                # Try to infer from config
                if hasattr(config, 'USE_LABELS') and config.USE_LABELS:
                    self.task = 'classification'
                else:
                    self.task = 'regression'
        
        print(f"Detected task: {self.task}")
        return checkpoints
    
    def load_test_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load test data for evaluation."""
        print("\n" + "="*70)
        print("LOADING TEST DATA")
        print("="*70)
        
        if not self.data_dir:
            print("  No data directory provided - will use validation split from training data")
            return {}
        
        test_data = {}
        
        # Try to find test data files
        # Option 1: Look for test_data.npz (from prepare_fl_data.py)
        test_file = os.path.join(self.data_dir, "test_data.npz")
        if os.path.exists(test_file):
            print(f"  Found test_data.npz")
            try:
                data = np.load(test_file)
                X_test = data['X']
                y_test = data['y']
                test_data['global'] = (X_test, y_test)
                print(f"    Global test set: {X_test.shape[0]} samples")
            except Exception as e:
                print(f"    Warning: Failed to load test_data.npz: {e}")
        
        # Option 2: Load client-specific test data from CSV files
        # Decide whether to prefer packet data or GPS/IMU data based on config/task.
        # For classification / packet configs, prefer packet_client_*.csv.
        # For regression configs like config_regression (USE_LABELS=False, no packet
        # features), prefer gps_client_*.csv.
        prefer_packets = False
        try:
            prefer_packets = bool(getattr(config, "USE_LABELS", False)) or self.task == "classification"
        except Exception:
            prefer_packets = self.task == "classification"

        if prefer_packets:
            client_test_files = sorted(glob.glob(f"{self.data_dir}/packet_client_*.csv"))
            if not client_test_files:
                client_test_files = sorted(glob.glob(f"{self.data_dir}/gps_client_*.csv"))
        else:
            client_test_files = sorted(glob.glob(f"{self.data_dir}/gps_client_*.csv"))
            if not client_test_files:
                client_test_files = sorted(glob.glob(f"{self.data_dir}/packet_client_*.csv"))
        
        if client_test_files:
            print(f"  Found {len(client_test_files)} client data files")
            
            for client_file in client_test_files:
                client_id = Path(client_file).stem
                print(f"    Processing {client_id}...")
                
                try:
                    # Determine file type and create appropriate preprocessor
                    if 'packet' in client_file.lower():
                        preprocessor = DataPreprocessorWithPackets(
                            packet_file=client_file,
                            packet_features=config.PACKET_FEATURES,
                            packet_timestamp_col=config.PACKET_TIMESTAMP_COL,
                            label_col=config.LABEL_COL,
                            use_labels=config.USE_LABELS
                        )
                    else:
                        # GPS/IMU data - find corresponding files
                        client_num = client_id.split('_')[-1]
                        gps_file = os.path.join(self.data_dir, f"gps_client_{client_num}.csv")
                        imu_file = os.path.join(self.data_dir, f"imu_client_{client_num}.csv")
                        
                        gps_file = gps_file if os.path.exists(gps_file) else None
                        imu_file = imu_file if os.path.exists(imu_file) else None
                        
                        preprocessor = DataPreprocessorWithPackets(
                            gps_file=gps_file,
                            imu_file=imu_file,
                            gps_features=getattr(config, 'GPS_FEATURES', None),
                            imu_features=getattr(config, 'IMU_FEATURES', None),
                            packet_features=getattr(config, 'PACKET_FEATURES', None),
                            timestamp_col=getattr(config, 'TIMESTAMP_COL', 'TimeUS'),
                            use_labels=False  # Regression for GPS/IMU
                        )
                    
                    X, y, num_features = preprocessor.preprocess(
                        window_size=config.WINDOW_SIZE,
                        overlap=config.OVERLAP
                    )
                    
                    # Split into train/test (use last portion as test)
                    n_test = int(len(X) * self.test_split)
                    if n_test > 0:
                        X_test = X[-n_test:]
                        y_test = y[-n_test:]
                        # Normalize client_id to match checkpoint naming (e.g., "packet_client_000" -> "client_1")
                        # Extract client number from filename
                        if 'packet_client_' in client_id:
                            # Extract number from "packet_client_000"
                            client_num_str = client_id.replace('packet_client_', '').replace('client_', '')
                            try:
                                client_num = int(client_num_str)
                                # Convert to 1-indexed client_id format
                                test_key = f"client_{client_num + 1}"
                            except:
                                test_key = client_id
                        else:
                            test_key = client_id
                        test_data[test_key] = (X_test, y_test)
                        print(f"      Test set: {X_test.shape[0]} samples (from {len(X)} total) [key: {test_key}]")
                    else:
                        print(f"      Warning: Not enough data for test split (need {self.test_split*100}%, have {len(X)} samples)")
                except Exception as e:
                    print(f"      Warning: Failed to process {client_id}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # If no test data found, try to create a combined test set from all clients
        if not test_data and self.data_dir:
            print("  No dedicated test data found. Creating combined test set from client data...")
            all_X_test = []
            all_y_test = []
            
            # Collect test portions from all client files
            for client_file in client_test_files:
                try:
                    client_id = Path(client_file).stem
                    if 'packet' in client_file.lower():
                        preprocessor = DataPreprocessorWithPackets(
                            packet_file=client_file,
                            packet_features=config.PACKET_FEATURES,
                            packet_timestamp_col=config.PACKET_TIMESTAMP_COL,
                            label_col=config.LABEL_COL,
                            use_labels=config.USE_LABELS
                        )
                    else:
                        client_num = client_id.split('_')[-1]
                        gps_file = os.path.join(self.data_dir, f"gps_client_{client_num}.csv")
                        imu_file = os.path.join(self.data_dir, f"imu_client_{client_num}.csv")
                        gps_file = gps_file if os.path.exists(gps_file) else None
                        imu_file = imu_file if os.path.exists(imu_file) else None
                        
                        preprocessor = DataPreprocessorWithPackets(
                            gps_file=gps_file,
                            imu_file=imu_file,
                            gps_features=getattr(config, 'GPS_FEATURES', None),
                            imu_features=getattr(config, 'IMU_FEATURES', None),
                            packet_features=getattr(config, 'PACKET_FEATURES', None),
                            timestamp_col=getattr(config, 'TIMESTAMP_COL', 'TimeUS'),
                            use_labels=False
                        )
                    
                    X, y, num_features = preprocessor.preprocess(
                        window_size=config.WINDOW_SIZE,
                        overlap=config.OVERLAP
                    )
                    n_test = int(len(X) * self.test_split)
                    if n_test > 0:
                        all_X_test.append(X[-n_test:])
                        all_y_test.append(y[-n_test:])
                except Exception as e:
                    print(f"    Warning: Failed to process {client_file}: {e}")
            
            if all_X_test:
                X_combined = np.concatenate(all_X_test, axis=0)
                y_combined = np.concatenate(all_y_test, axis=0)
                test_data['combined'] = (X_combined, y_combined)
                print(f"    Created combined test set: {X_combined.shape[0]} samples")
        
        if not test_data:
            print("  Warning: No test data available. Evaluation cannot proceed.")
            print("  Please provide test data or ensure client data files are available.")
        
        self.test_data = test_data
        print(f"\n✓ Loaded test data for {len(test_data)} test set(s)")
        return test_data
    
    def create_model(self, checkpoint: Dict) -> nn.Module:
        """Create model from checkpoint."""
        # Get model config
        model_config = config.MODEL_CONFIG.copy()
        
        # Update from checkpoint if available
        if 'model_state_dict' in checkpoint:
            # Try to infer input/output sizes from state dict
            state_dict = checkpoint['model_state_dict']
            
            # Find input size from first layer
            for key in state_dict.keys():
                if 'weight' in key and len(state_dict[key].shape) >= 2:
                    if 'lstm' in key.lower() or 'rnn' in key.lower():
                        # LSTM input size is second dimension
                        if state_dict[key].shape[1] > 0:
                            model_config['input_size'] = state_dict[key].shape[1]
                            break
                    elif 'linear' in key.lower() or 'fc' in key.lower():
                        # Linear layer - input size is second dimension
                        if state_dict[key].shape[1] > 0:
                            model_config['input_size'] = state_dict[key].shape[1]
                            break
        
        # Use config defaults if not found
        if 'input_size' not in model_config:
            model_config['input_size'] = config.INPUT_SIZE
        
        # Output size from checkpoint or config
        if 'output_size' in checkpoint:
            model_config['output_size'] = checkpoint['output_size']
        elif 'output_size' not in model_config:
            model_config['output_size'] = 1 if self.task == 'classification' else config.INPUT_SIZE
        
        # Create model
        model = self.model_class(**model_config)
        model.to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        return model
    
    def evaluate_classification(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate classification model."""
        model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                X_batch = X_tensor[i:i+1]
                output = model(X_batch)
                
                # Handle different output shapes
                if output.dim() > 1:
                    output = output.squeeze()
                
                # Get probabilities
                if output.dim() == 0:
                    prob = torch.sigmoid(output).item()
                    pred = 1 if prob > 0.5 else 0
                else:
                    if output.size(0) == 1:
                        # Binary classification
                        prob = torch.sigmoid(output[0]).item()
                        pred = 1 if prob > 0.5 else 0
                    else:
                        # Multi-class
                        probs = torch.softmax(output, dim=0)
                        pred = torch.argmax(probs).item()
                        prob = probs[pred].item()
                
                predictions.append(pred)
                probabilities.append(prob)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Flatten y_test if needed
        y_true = y_test.flatten() if y_test.ndim > 1 else y_test
        y_true = y_true.astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average='binary', zero_division=0)
        recall = recall_score(y_true, predictions, average='binary', zero_division=0)
        f1 = f1_score(y_true, predictions, average='binary', zero_division=0)
        
        # ROC AUC (for binary classification)
        try:
            if len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(y_true, probabilities)
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'y_true': y_true.tolist()
        }
        
        return results
    
    def evaluate_regression(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression model."""
        model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                X_batch = X_tensor[i:i+1]
                output = model(X_batch)
                
                # Handle output shape
                if output.dim() > 2:
                    output = output.squeeze()
                
                pred = output.cpu().numpy()
                predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        # Handle y_test shape
        if y_test.ndim == 1:
            y_true = y_test.reshape(-1, 1)
        else:
            y_true = y_test
        
        # Ensure same shape
        if predictions.shape != y_true.shape:
            min_dim = min(predictions.shape[1], y_true.shape[1])
            predictions = predictions[:, :min_dim]
            y_true = y_true[:, :min_dim]
        
        # Compute metrics
        mse = mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mse)
        
        # R2 score
        try:
            r2 = r2_score(y_true, predictions)
        except:
            r2 = None
        
        # Per-feature metrics
        per_feature_mse = []
        per_feature_mae = []
        for i in range(predictions.shape[1]):
            per_feature_mse.append(mean_squared_error(y_true[:, i], predictions[:, i]))
            per_feature_mae.append(mean_absolute_error(y_true[:, i], predictions[:, i]))
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'per_feature_mse': per_feature_mse,
            'per_feature_mae': per_feature_mae,
            'predictions': predictions.tolist(),
            'y_true': y_true.tolist()
        }
        
        return results
    
    def load_client_checkpoints(self, round_num: int) -> Dict[str, Dict]:
        """Load client checkpoints for a specific round.
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary mapping client_id to checkpoint data
        """
        client_checkpoints = {}
        client_checkpoint_dir = os.path.join(self.checkpoint_dir, 'clients')
        
        if not os.path.exists(client_checkpoint_dir):
            logger.debug(f"Client checkpoint directory not found: {client_checkpoint_dir}")
            return {}
        
        # Find all client checkpoints for this round
        pattern = f"*_round_{round_num}.pt"
        checkpoint_files = glob.glob(os.path.join(client_checkpoint_dir, pattern))
        
        if not checkpoint_files:
            logger.debug(f"No client checkpoints found for round {round_num} in {client_checkpoint_dir}")
            return {}
        
        logger.info(f"Found {len(checkpoint_files)} client checkpoints for round {round_num}")
        
        for ckpt_file in checkpoint_files:
            try:
                # Extract client_id from filename (e.g., "client_1_round_5.pt" -> "client_1")
                filename = os.path.basename(ckpt_file)
                client_id = filename.replace(f"_round_{round_num}.pt", "")
                
                checkpoint = torch.load(ckpt_file, map_location=self.device)
                client_checkpoints[client_id] = checkpoint
                logger.info(f"Loaded client checkpoint: {client_id} round {round_num}")
                print(f"      Loaded checkpoint for {client_id}")
            except Exception as e:
                logger.warning(f"Failed to load client checkpoint {ckpt_file}: {e}")
                print(f"      Warning: Failed to load {filename}: {e}")
        
        return client_checkpoints
    
    def evaluate_client_models(self, round_num: int, server_model: Optional[nn.Module] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate each client's own model on its test data.
        
        This shows how well each client's trained model performs on its own data,
        revealing client-specific performance and heterogeneity.
        
        Args:
            round_num: Round number
            server_model: Optional server model (if client checkpoints not available, falls back to server model)
            
        Returns:
            Dictionary mapping client_id to evaluation results
        """
        client_results = {}
        
        # Try to load client checkpoints first
        client_checkpoints = self.load_client_checkpoints(round_num)
        
        # Find client-specific test data and match with checkpoints
        client_test_data = {}
        for test_name, (X_test, y_test) in self.test_data.items():
            # Check if this is a client-specific test set
            if ('client_' in test_name.lower() and test_name != 'combined') or \
               test_name.startswith('packet_client'):
                # Try to match with checkpoint client IDs
                matched = False
                for ckpt_client_id in client_checkpoints.keys():
                    # Match if test_name contains client number or vice versa
                    # e.g., "client_1" matches "client_1", "packet_client_000" matches "client_1" (if 000 -> 1)
                    if test_name == ckpt_client_id:
                        client_test_data[ckpt_client_id] = (X_test, y_test)
                        matched = True
                        break
                    # Try extracting numbers
                    test_num = None
                    ckpt_num = None
                    try:
                        if 'client_' in test_name:
                            test_num = int(test_name.split('_')[-1])
                        if 'client_' in ckpt_client_id:
                            ckpt_num = int(ckpt_client_id.split('_')[-1])
                        if test_num is not None and ckpt_num is not None and test_num == ckpt_num:
                            client_test_data[ckpt_client_id] = (X_test, y_test)
                            matched = True
                            break
                    except:
                        pass
                
                if not matched:
                    # If no checkpoint match, still include for server model evaluation
                    client_test_data[test_name] = (X_test, y_test)
        
        if not client_test_data:
            print("  Note: No per-client test sets found. Client-specific evaluation skipped.")
            print("  Tip: To enable client evaluation, ensure test data is loaded per-client.")
            return {}
        
        if client_checkpoints:
            print(f"\n  Evaluating {len(client_checkpoints)} client models on their own test data...")
            print(f"  (Each client's model evaluated on its own test set)")
        else:
            if server_model is None:
                print("  Note: No client checkpoints found and no server model provided.")
                print(f"  Client checkpoints should be in: {os.path.join(self.checkpoint_dir, 'clients')}")
                return {}
            print(f"\n  No client checkpoints found. Falling back to server model evaluation.")
            print(f"  (Server model evaluated on {len(client_test_data)} client test sets)")
        
        # Evaluate each client
        for client_id, (X_test, y_test) in client_test_data.items():
            try:
                print(f"    {client_id}: {len(X_test)} samples")
                
                # Use client's own model if checkpoint exists, otherwise use server model
                if client_id in client_checkpoints:
                    checkpoint = client_checkpoints[client_id]
                    model = self.create_model(checkpoint)
                    model_type = f"{client_id}'s own model"
                    print(f"      Using {model_type} (from checkpoint)")
                elif server_model is not None:
                    model = server_model
                    model_type = "server model (fallback)"
                    print(f"      Using {model_type} (client checkpoint not available)")
                else:
                    print(f"      Skipping {client_id}: No model available")
                    continue
                
                if self.task == 'classification':
                    results = self.evaluate_classification(model, X_test, y_test)
                else:
                    results = self.evaluate_regression(model, X_test, y_test)
                
                client_results[client_id] = results
                
                # Print summary
                if self.task == 'classification':
                    print(f"      Accuracy: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}")
                else:
                    print(f"      RMSE: {results['rmse']:.6f}, MAE: {results['mae']:.6f}")
                    
            except Exception as e:
                print(f"      Error evaluating {client_id}: {e}")
                import traceback
                traceback.print_exc()
                client_results[client_id] = None
        
        return client_results
    
    def evaluate_all_checkpoints(self, rounds: Optional[List[int]] = None, evaluate_clients: bool = True) -> Dict[int, Dict[str, Any]]:
        """Evaluate all checkpoints on test data.
        
        Args:
            rounds: Optional list of specific rounds to evaluate. If None, evaluates all rounds.
        """
        print("\n" + "="*70)
        print("EVALUATING ALL CHECKPOINTS")
        print("="*70)
        
        if not self.checkpoints:
            self.load_checkpoints()
        
        if not self.test_data:
            self.load_test_data()
        
        if not self.test_data:
            print("  Error: No test data available. Cannot proceed with evaluation.")
            return {}
        
        evaluation_results = {}
        
        # Determine which rounds to evaluate
        if rounds is None:
            rounds_to_eval = sorted(self.checkpoints.keys())
        else:
            rounds_to_eval = [r for r in sorted(self.checkpoints.keys()) if r in rounds]
            if not rounds_to_eval:
                print(f"  Warning: None of the specified rounds {rounds} are available.")
                print(f"  Available rounds: {sorted(self.checkpoints.keys())}")
                return {}
        
        evaluation_results = {}
        
        # Evaluate each checkpoint
        for round_num in rounds_to_eval:
            print(f"\nEvaluating Round {round_num}...")
            checkpoint = self.checkpoints[round_num]
            
            try:
                model = self.create_model(checkpoint)
                
                # Evaluate on each test set (server model evaluation)
                round_results = {}
                for test_name, (X_test, y_test) in self.test_data.items():
                    print(f"  Test set: {test_name} ({len(X_test)} samples)")
                    
                    if self.task == 'classification':
                        results = self.evaluate_classification(model, X_test, y_test)
                    else:
                        results = self.evaluate_regression(model, X_test, y_test)
                    
                    round_results[test_name] = results
                    
                    # Print summary
                    if self.task == 'classification':
                        print(f"    Accuracy: {results['accuracy']:.4f}")
                        print(f"    F1 Score: {results['f1_score']:.4f}")
                        if results['roc_auc']:
                            print(f"    ROC AUC: {results['roc_auc']:.4f}")
                    else:
                        print(f"    RMSE: {results['rmse']:.6f}")
                        print(f"    MAE: {results['mae']:.6f}")
                        if results['r2_score']:
                            print(f"    R² Score: {results['r2_score']:.4f}")
                
                # Evaluate on per-client test data (client-specific evaluation)
                if evaluate_clients:
                    client_results = self.evaluate_client_models(round_num, server_model=model)
                    if client_results:
                        round_results['_client_results'] = client_results
                        self.client_evaluation_results[round_num] = client_results
                
                evaluation_results[round_num] = round_results
                
            except Exception as e:
                print(f"  Error evaluating round {round_num}: {e}")
                import traceback
                traceback.print_exc()
        
        self.evaluation_results = evaluation_results
        print(f"\n✓ Evaluated {len(evaluation_results)} checkpoints")
        return evaluation_results
    
    def plot_client_heterogeneity(self, output_dir: str = "evaluation_results"):
        """Plot client-specific performance to show heterogeneity."""
        print("\n" + "="*70)
        print("GENERATING CLIENT HETEROGENEITY PLOTS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.client_evaluation_results:
            print("  No client evaluation results available.")
            return
        
        rounds = sorted(self.client_evaluation_results.keys())
        if not rounds:
            print("  No client results to plot.")
            return
        
        # Get all client IDs
        all_clients = set()
        for round_results in self.client_evaluation_results.values():
            all_clients.update(round_results.keys())
        all_clients = sorted(all_clients)
        
        if self.task == 'classification':
            # Plot accuracy and F1 score per client over rounds
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Accuracy plot
            ax1 = axes[0]
            for client_id in all_clients:
                accuracies = []
                client_rounds = []
                for round_num in rounds:
                    if client_id in self.client_evaluation_results[round_num]:
                        results = self.client_evaluation_results[round_num][client_id]
                        if results:
                            accuracies.append(results['accuracy'])
                            client_rounds.append(round_num)
                
                if accuracies:
                    ax1.plot(client_rounds, accuracies, marker='o', label=client_id, linewidth=2)
            
            ax1.set_xlabel('Round', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_title('Client-Specific Accuracy Over Rounds', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # F1 Score plot
            ax2 = axes[1]
            for client_id in all_clients:
                f1_scores = []
                client_rounds = []
                for round_num in rounds:
                    if client_id in self.client_evaluation_results[round_num]:
                        results = self.client_evaluation_results[round_num][client_id]
                        if results:
                            f1_scores.append(results['f1_score'])
                            client_rounds.append(round_num)
                
                if f1_scores:
                    ax2.plot(client_rounds, f1_scores, marker='s', label=client_id, linewidth=2)
            
            ax2.set_xlabel('Round', fontsize=12)
            ax2.set_ylabel('F1 Score', fontsize=12)
            ax2.set_title('Client-Specific F1 Score Over Rounds', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = f"{output_dir}/client_heterogeneity_classification.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plot_path}")
            
            # Box plot for final round performance distribution
            if rounds:
                final_round = max(rounds)
                if final_round in self.client_evaluation_results:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                    
                    client_accs = []
                    client_f1s = []
                    client_labels = []
                    
                    for client_id in all_clients:
                        if client_id in self.client_evaluation_results[final_round]:
                            results = self.client_evaluation_results[final_round][client_id]
                            if results:
                                client_accs.append(results['accuracy'])
                                client_f1s.append(results['f1_score'])
                                client_labels.append(client_id)
                    
                    if client_accs:
                        axes[0].boxplot([client_accs], labels=['All Clients'])
                        axes[0].scatter([1] * len(client_accs), client_accs, alpha=0.6, s=100)
                        axes[0].set_ylabel('Accuracy', fontsize=12)
                        axes[0].set_title(f'Accuracy Distribution Across Clients (Round {final_round})', 
                                         fontsize=14, fontweight='bold')
                        axes[0].grid(True, alpha=0.3)
                        
                        axes[1].boxplot([client_f1s], labels=['All Clients'])
                        axes[1].scatter([1] * len(client_f1s), client_f1s, alpha=0.6, s=100)
                        axes[1].set_ylabel('F1 Score', fontsize=12)
                        axes[1].set_title(f'F1 Score Distribution Across Clients (Round {final_round})', 
                                         fontsize=14, fontweight='bold')
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plot_path = f"{output_dir}/client_performance_distribution.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"  Saved: {plot_path}")
        else:
            # Regression metrics
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # RMSE plot
            ax1 = axes[0]
            for client_id in all_clients:
                rmses = []
                client_rounds = []
                for round_num in rounds:
                    if client_id in self.client_evaluation_results[round_num]:
                        results = self.client_evaluation_results[round_num][client_id]
                        if results:
                            rmses.append(results['rmse'])
                            client_rounds.append(round_num)
                
                if rmses:
                    ax1.plot(client_rounds, rmses, marker='o', label=client_id, linewidth=2)
            
            ax1.set_xlabel('Round', fontsize=12)
            ax1.set_ylabel('RMSE', fontsize=12)
            ax1.set_title('Client-Specific RMSE Over Rounds', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # MAE plot
            ax2 = axes[1]
            for client_id in all_clients:
                maes = []
                client_rounds = []
                for round_num in rounds:
                    if client_id in self.client_evaluation_results[round_num]:
                        results = self.client_evaluation_results[round_num][client_id]
                        if results:
                            maes.append(results['mae'])
                            client_rounds.append(round_num)
                
                if maes:
                    ax2.plot(client_rounds, maes, marker='s', label=client_id, linewidth=2)
            
            ax2.set_xlabel('Round', fontsize=12)
            ax2.set_ylabel('MAE', fontsize=12)
            ax2.set_title('Client-Specific MAE Over Rounds', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = f"{output_dir}/client_heterogeneity_regression.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plot_path}")
    
    def plot_metrics_over_rounds(self, output_dir: str = "evaluation_results"):
        """Plot metrics over training rounds."""
        print("\n" + "="*70)
        print("GENERATING METRICS PLOTS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.evaluation_results:
            print("  No evaluation results available. Run evaluate_all_checkpoints() first.")
            return
        
        rounds = sorted(self.evaluation_results.keys())
        
        # Collect metrics for each test set
        for test_name in self.test_data.keys():
            if self.task == 'classification':
                accuracies = []
                f1_scores = []
                precisions = []
                recalls = []
                roc_aucs = []
                
                for round_num in rounds:
                    if test_name in self.evaluation_results[round_num]:
                        results = self.evaluation_results[round_num][test_name]
                        accuracies.append(results['accuracy'])
                        f1_scores.append(results['f1_score'])
                        precisions.append(results['precision'])
                        recalls.append(results['recall'])
                        if results['roc_auc']:
                            roc_aucs.append(results['roc_auc'])
                        else:
                            roc_aucs.append(None)
                
                # Plot classification metrics
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Accuracy
                axes[0, 0].plot(rounds, accuracies, 'o-', linewidth=2, markersize=6)
                axes[0, 0].set_xlabel('Round', fontsize=12)
                axes[0, 0].set_ylabel('Accuracy', fontsize=12)
                axes[0, 0].set_title('Accuracy over Rounds', fontsize=14, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                
                # F1 Score
                axes[0, 1].plot(rounds, f1_scores, 's-', linewidth=2, markersize=6, color='orange')
                axes[0, 1].set_xlabel('Round', fontsize=12)
                axes[0, 1].set_ylabel('F1 Score', fontsize=12)
                axes[0, 1].set_title('F1 Score over Rounds', fontsize=14, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Precision and Recall
                axes[1, 0].plot(rounds, precisions, '^-', linewidth=2, markersize=6, label='Precision', color='green')
                axes[1, 0].plot(rounds, recalls, 'v-', linewidth=2, markersize=6, label='Recall', color='red')
                axes[1, 0].set_xlabel('Round', fontsize=12)
                axes[1, 0].set_ylabel('Score', fontsize=12)
                axes[1, 0].set_title('Precision and Recall', fontsize=14, fontweight='bold')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # ROC AUC
                valid_roc_aucs = [(r, auc) for r, auc in zip(rounds, roc_aucs) if auc is not None]
                if valid_roc_aucs:
                    rounds_auc, aucs = zip(*valid_roc_aucs)
                    axes[1, 1].plot(rounds_auc, aucs, 'd-', linewidth=2, markersize=6, color='purple')
                    axes[1, 1].set_xlabel('Round', fontsize=12)
                    axes[1, 1].set_ylabel('ROC AUC', fontsize=12)
                    axes[1, 1].set_title('ROC AUC over Rounds', fontsize=14, fontweight='bold')
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'ROC AUC not available', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('ROC AUC over Rounds', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plot_path = f"{output_dir}/classification_metrics_{test_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {plot_path}")
            
            else:  # Regression
                rmses = []
                maes = []
                r2_scores = []
                
                for round_num in rounds:
                    if test_name in self.evaluation_results[round_num]:
                        results = self.evaluation_results[round_num][test_name]
                        rmses.append(results['rmse'])
                        maes.append(results['mae'])
                        if results['r2_score']:
                            r2_scores.append(results['r2_score'])
                        else:
                            r2_scores.append(None)
                
                # Plot regression metrics
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # RMSE
                axes[0].plot(rounds, rmses, 'o-', linewidth=2, markersize=6, color='red')
                axes[0].set_xlabel('Round', fontsize=12)
                axes[0].set_ylabel('RMSE', fontsize=12)
                axes[0].set_title('RMSE over Rounds', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                # MAE
                axes[1].plot(rounds, maes, 's-', linewidth=2, markersize=6, color='orange')
                axes[1].set_xlabel('Round', fontsize=12)
                axes[1].set_ylabel('MAE', fontsize=12)
                axes[1].set_title('MAE over Rounds', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                # R² Score
                valid_r2 = [(r, r2) for r, r2 in zip(rounds, r2_scores) if r2 is not None]
                if valid_r2:
                    rounds_r2, r2s = zip(*valid_r2)
                    axes[2].plot(rounds_r2, r2s, '^-', linewidth=2, markersize=6, color='green')
                    axes[2].set_xlabel('Round', fontsize=12)
                    axes[2].set_ylabel('R² Score', fontsize=12)
                    axes[2].set_title('R² Score over Rounds', fontsize=14, fontweight='bold')
                    axes[2].grid(True, alpha=0.3)
                else:
                    axes[2].text(0.5, 0.5, 'R² Score not available', 
                               ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title('R² Score over Rounds', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plot_path = f"{output_dir}/regression_metrics_{test_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {plot_path}")
    
    def plot_confusion_matrices(self, output_dir: str = "evaluation_results"):
        """Plot confusion matrices for classification tasks."""
        if self.task != 'classification':
            return
        
        print("\nGenerating confusion matrices...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        rounds = sorted(self.evaluation_results.keys())
        
        # Plot confusion matrices for key rounds (first, middle, last)
        key_rounds = [rounds[0]]
        if len(rounds) > 1:
            key_rounds.append(rounds[len(rounds)//2])
        key_rounds.append(rounds[-1])
        
        for test_name in self.test_data.keys():
            fig, axes = plt.subplots(1, len(key_rounds), figsize=(5*len(key_rounds), 4))
            if len(key_rounds) == 1:
                axes = [axes]
            
            for idx, round_num in enumerate(key_rounds):
                if test_name in self.evaluation_results[round_num]:
                    results = self.evaluation_results[round_num][test_name]
                    cm = np.array(results['confusion_matrix'])
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                              cbar_kws={'label': 'Count'})
                    axes[idx].set_xlabel('Predicted', fontsize=12)
                    axes[idx].set_ylabel('Actual', fontsize=12)
                    axes[idx].set_title(f'Round {round_num}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plot_path = f"{output_dir}/confusion_matrices_{test_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plot_path}")
    
    def plot_predictions_vs_actual(self, output_dir: str = "evaluation_results"):
        """Plot predictions vs actual values for regression tasks."""
        if self.task != 'regression':
            return
        
        print("\nGenerating predictions vs actual plots...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        rounds = sorted(self.evaluation_results.keys())
        last_round = rounds[-1]
        
        for test_name in self.test_data.keys():
            if test_name in self.evaluation_results[last_round]:
                results = self.evaluation_results[last_round][test_name]
                predictions = np.array(results['predictions'])
                y_true = np.array(results['y_true'])
                
                # Plot for each feature (up to 6 features)
                n_features = min(predictions.shape[1], 6)
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for i in range(n_features):
                    ax = axes[i]
                    ax.scatter(y_true[:, i], predictions[:, i], alpha=0.5, s=20)
                    
                    # Add diagonal line
                    min_val = min(y_true[:, i].min(), predictions[:, i].min())
                    max_val = max(y_true[:, i].max(), predictions[:, i].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                    
                    ax.set_xlabel('Actual', fontsize=12)
                    ax.set_ylabel('Predicted', fontsize=12)
                    ax.set_title(f'Feature {i+1}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(n_features, len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plot_path = f"{output_dir}/predictions_vs_actual_{test_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {plot_path}")
    
    def calculate_client_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about client performance heterogeneity."""
        if not self.client_evaluation_results:
            return {}
        
        stats = {}
        rounds = sorted(self.client_evaluation_results.keys())
        if not rounds:
            return {}
        
        final_round = max(rounds)
        if final_round not in self.client_evaluation_results:
            return {}
        
        final_results = self.client_evaluation_results[final_round]
        
        if self.task == 'classification':
            accuracies = []
            f1_scores = []
            precisions = []
            recalls = []
            
            for client_id, results in final_results.items():
                if results:
                    accuracies.append(results['accuracy'])
                    f1_scores.append(results['f1_score'])
                    precisions.append(results['precision'])
                    recalls.append(results['recall'])
            
            if accuracies:
                stats = {
                    'accuracy': {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'min': np.min(accuracies),
                        'max': np.max(accuracies),
                        'range': np.max(accuracies) - np.min(accuracies)
                    },
                    'f1_score': {
                        'mean': np.mean(f1_scores),
                        'std': np.std(f1_scores),
                        'min': np.min(f1_scores),
                        'max': np.max(f1_scores),
                        'range': np.max(f1_scores) - np.min(f1_scores)
                    },
                    'precision': {
                        'mean': np.mean(precisions),
                        'std': np.std(precisions),
                        'min': np.min(precisions),
                        'max': np.max(precisions)
                    },
                    'recall': {
                        'mean': np.mean(recalls),
                        'std': np.std(recalls),
                        'min': np.min(recalls),
                        'max': np.max(recalls)
                    }
                }
        else:
            rmses = []
            maes = []
            r2_scores = []
            
            for client_id, results in final_results.items():
                if results:
                    rmses.append(results['rmse'])
                    maes.append(results['mae'])
                    if results['r2_score']:
                        r2_scores.append(results['r2_score'])
            
            if rmses:
                stats = {
                    'rmse': {
                        'mean': np.mean(rmses),
                        'std': np.std(rmses),
                        'min': np.min(rmses),
                        'max': np.max(rmses),
                        'range': np.max(rmses) - np.min(rmses)
                    },
                    'mae': {
                        'mean': np.mean(maes),
                        'std': np.std(maes),
                        'min': np.min(maes),
                        'max': np.max(maes)
                    }
                }
                if r2_scores:
                    stats['r2_score'] = {
                        'mean': np.mean(r2_scores),
                        'std': np.std(r2_scores),
                        'min': np.min(r2_scores),
                        'max': np.max(r2_scores)
                    }
        
        return stats
    
    def generate_report(self, output_dir: str = "evaluation_results"):
        """Generate comprehensive evaluation report."""
        print("\n" + "="*70)
        print("GENERATING EVALUATION REPORT")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'task': self.task,
            'checkpoint_dir': self.checkpoint_dir,
            'data_dir': self.data_dir,
            'num_checkpoints': len(self.checkpoints),
            'evaluation_results': {},
            'has_client_evaluation': len(self.client_evaluation_results) > 0
        }
        
        rounds = sorted(self.evaluation_results.keys())
        
        # Summary statistics
        summary = {}
        for test_name in self.test_data.keys():
            if self.task == 'classification':
                best_round = None
                best_accuracy = 0
                best_f1 = 0
                
                for round_num in rounds:
                    if test_name in self.evaluation_results[round_num]:
                        results = self.evaluation_results[round_num][test_name]
                        if results['accuracy'] > best_accuracy:
                            best_accuracy = results['accuracy']
                            best_round = round_num
                        if results['f1_score'] > best_f1:
                            best_f1 = results['f1_score']
                
                summary[test_name] = {
                    'best_round': best_round,
                    'best_accuracy': best_accuracy,
                    'best_f1_score': best_f1,
                    'final_accuracy': self.evaluation_results[rounds[-1]][test_name]['accuracy'],
                    'final_f1_score': self.evaluation_results[rounds[-1]][test_name]['f1_score']
                }
            else:
                best_round = None
                best_rmse = float('inf')
                best_r2 = -float('inf')
                
                for round_num in rounds:
                    if test_name in self.evaluation_results[round_num]:
                        results = self.evaluation_results[round_num][test_name]
                        if results['rmse'] < best_rmse:
                            best_rmse = results['rmse']
                            best_round = round_num
                        if results['r2_score'] and results['r2_score'] > best_r2:
                            best_r2 = results['r2_score']
                
                summary[test_name] = {
                    'best_round': best_round,
                    'best_rmse': best_rmse,
                    'best_r2_score': best_r2 if best_r2 != -float('inf') else None,
                    'final_rmse': self.evaluation_results[rounds[-1]][test_name]['rmse'],
                    'final_mae': self.evaluation_results[rounds[-1]][test_name]['mae'],
                    'final_r2_score': self.evaluation_results[rounds[-1]][test_name]['r2_score']
                }
        
        report['summary'] = summary
        
        # Add client evaluation results if available
        if self.client_evaluation_results:
            report['client_evaluation_results'] = {}
            for round_num in rounds:
                if round_num in self.client_evaluation_results:
                    # Convert numpy types to native Python types for JSON
                    client_round_results = {}
                    for client_id, results in self.client_evaluation_results[round_num].items():
                        if results:
                            client_round_results[client_id] = {
                                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                for k, v in results.items()
                                if k != 'confusion_matrix'  # Skip confusion matrix for JSON
                            }
                    report['client_evaluation_results'][round_num] = client_round_results
            
            # Add client heterogeneity statistics
            client_stats = self.calculate_client_statistics()
            if client_stats:
                # Convert numpy types to native Python types
                report['client_heterogeneity'] = {
                    metric: {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in stats_dict.items()
                    }
                    for metric, stats_dict in client_stats.items()
                }
        
        # Save JSON report
        json_path = f"{output_dir}/evaluation_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  Saved JSON report: {json_path}")
        
        # Save text report
        txt_path = f"{output_dir}/evaluation_report.txt"
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FEDERATED LEARNING SYSTEM EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Checkpoint Directory: {self.checkpoint_dir}\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Number of Checkpoints Evaluated: {len(self.checkpoints)}\n")
            f.write(f"Rounds Evaluated: {min(rounds)} to {max(rounds)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            for test_name, stats in summary.items():
                f.write(f"Test Set: {test_name}\n")
                f.write("-" * 70 + "\n")
                
                if self.task == 'classification':
                    f.write(f"  Best Round: {stats['best_round']}\n")
                    f.write(f"  Best Accuracy: {stats['best_accuracy']:.4f}\n")
                    f.write(f"  Best F1 Score: {stats['best_f1_score']:.4f}\n")
                    f.write(f"  Final Accuracy: {stats['final_accuracy']:.4f}\n")
                    f.write(f"  Final F1 Score: {stats['final_f1_score']:.4f}\n")
                else:
                    f.write(f"  Best Round: {stats['best_round']}\n")
                    f.write(f"  Best RMSE: {stats['best_rmse']:.6f}\n")
                    if stats['best_r2_score']:
                        f.write(f"  Best R² Score: {stats['best_r2_score']:.4f}\n")
                    f.write(f"  Final RMSE: {stats['final_rmse']:.6f}\n")
                    f.write(f"  Final MAE: {stats['final_mae']:.6f}\n")
                    if stats['final_r2_score']:
                        f.write(f"  Final R² Score: {stats['final_r2_score']:.4f}\n")
                
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("DETAILED RESULTS BY ROUND\n")
            f.write("="*70 + "\n\n")
            
            for round_num in rounds:
                f.write(f"Round {round_num}:\n")
                f.write("-" * 70 + "\n")
                
                # Server model results
                for test_name in self.test_data.keys():
                    if test_name in self.evaluation_results[round_num]:
                        results = self.evaluation_results[round_num][test_name]
                        f.write(f"  Server Model - {test_name}:\n")
                        
                        if self.task == 'classification':
                            f.write(f"    Accuracy: {results['accuracy']:.4f}\n")
                            f.write(f"    Precision: {results['precision']:.4f}\n")
                            f.write(f"    Recall: {results['recall']:.4f}\n")
                            f.write(f"    F1 Score: {results['f1_score']:.4f}\n")
                            if results['roc_auc']:
                                f.write(f"    ROC AUC: {results['roc_auc']:.4f}\n")
                        else:
                            f.write(f"    RMSE: {results['rmse']:.6f}\n")
                            f.write(f"    MAE: {results['mae']:.6f}\n")
                            if results['r2_score']:
                                f.write(f"    R² Score: {results['r2_score']:.4f}\n")
                        
                        f.write("\n")
                
                # Client-specific results
                if round_num in self.client_evaluation_results:
                    f.write("  Client-Specific Performance:\n")
                    for client_id, results in self.client_evaluation_results[round_num].items():
                        if results:
                            f.write(f"    {client_id}:\n")
                            if self.task == 'classification':
                                f.write(f"      Accuracy: {results['accuracy']:.4f}\n")
                                f.write(f"      F1 Score: {results['f1_score']:.4f}\n")
                                f.write(f"      Precision: {results['precision']:.4f}\n")
                                f.write(f"      Recall: {results['recall']:.4f}\n")
                            else:
                                f.write(f"      RMSE: {results['rmse']:.6f}\n")
                                f.write(f"      MAE: {results['mae']:.6f}\n")
                            f.write("\n")
                
                f.write("\n")
            
            # Client heterogeneity statistics
            if self.client_evaluation_results:
                f.write("="*70 + "\n")
                f.write("CLIENT HETEROGENEITY STATISTICS\n")
                f.write("="*70 + "\n\n")
                
                client_stats = self.calculate_client_statistics()
                if client_stats:
                    f.write("Final Round Client Performance Distribution:\n")
                    for metric, stats_dict in client_stats.items():
                        f.write(f"  {metric.upper()}:\n")
                        f.write(f"    Mean: {stats_dict['mean']:.4f}\n")
                        f.write(f"    Std: {stats_dict['std']:.4f}\n")
                        f.write(f"    Min: {stats_dict['min']:.4f}\n")
                        f.write(f"    Max: {stats_dict['max']:.4f}\n")
                        if 'range' in stats_dict:
                            f.write(f"    Range: {stats_dict['range']:.4f}\n")
                        f.write("\n")
        
        print(f"  Saved text report: {txt_path}")
    
    def run_full_evaluation(self, output_dir: str = "evaluation_results", rounds: Optional[List[int]] = None, evaluate_clients: bool = True):
        """Run complete evaluation pipeline.
        
        Args:
            output_dir: Directory to save evaluation results
            rounds: Optional list of specific rounds to evaluate
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE FL SYSTEM EVALUATION")
        print("="*70)
        
        # Load checkpoints
        self.load_checkpoints()
        
        # Load test data
        self.load_test_data()
        
        if not self.test_data:
            print("\n❌ Error: No test data available. Cannot proceed with evaluation.")
            print("Please provide test data or ensure client data files are available.")
            return
        
        # Evaluate all checkpoints
        self.evaluate_all_checkpoints(rounds=rounds, evaluate_clients=evaluate_clients)
        
        # Generate plots
        self.plot_metrics_over_rounds(output_dir)
        if self.client_evaluation_results:
            self.plot_client_heterogeneity(output_dir)
        if self.task == 'classification':
            self.plot_confusion_matrices(output_dir)
        else:
            self.plot_predictions_vs_actual(output_dir)
        
        # Generate report
        self.generate_report(output_dir)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {output_dir}/")
        print("  - evaluation_report.json")
        print("  - evaluation_report.txt")
        print("  - Metrics plots")
        if self.client_evaluation_results:
            print("  - Client heterogeneity plots")
        if self.task == 'classification':
            print("  - Confusion matrices")
        else:
            print("  - Predictions vs actual plots")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Evaluation Script for Federated Learning System'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory with client data files (for test data)'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression', 'auto'],
        default='auto',
        help='Task type (auto = detect from checkpoints)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (if test data not available)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (None = auto-detect)'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        nargs='+',
        default=None,
        help='Specific rounds to evaluate (default: all rounds)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_packets_only',
        help='Config module to use (e.g., config_packets_only, config_regression, config)'
    )
    parser.add_argument(
        '--evaluate-clients',
        action='store_true',
        default=True,
        help='Evaluate server model on per-client test data (default: True)'
    )
    parser.add_argument(
        '--no-client-eval',
        dest='evaluate_clients',
        action='store_false',
        help='Disable client-specific evaluation'
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load selected config module and expose its UPPERCASE attributes
    # globally so existing code (SERVER_HOST, INPUT_SIZE, etc.) works.
    # ------------------------------------------------------------------
    global config
    config_module_name = args.config
    try:
        config = importlib.import_module(config_module_name)
    except ImportError as e:
        print(f"[ERROR] Could not import config module '{config_module_name}': {e}")
        sys.exit(1)

    for _name, _value in vars(config).items():
        if _name.isupper():
            globals()[_name] = _value
    
    # Create evaluator
    evaluator = FLEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        task=args.task,
        test_split=args.test_split,
        device=args.device
    )
    
    # Run full evaluation
    evaluator.run_full_evaluation(
        output_dir=args.output_dir, 
        rounds=args.rounds,
        evaluate_clients=args.evaluate_clients
    )


if __name__ == "__main__":
    main()
