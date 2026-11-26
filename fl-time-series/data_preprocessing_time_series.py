"""
Data Preprocessing Module for GPS, IMU, and Labeled Packet Data Fusion

This module handles:
1. Loading GPS, IMU, and labeled packet data from CSV files
2. Merging and aligning data by timestamp
3. Feature concatenation
4. Normalization and preprocessing
5. Sliding window generation with overlap
6. Support for supervised learning with packet labels

Clean, modular design with flexible data source support.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def oversample_attack_windows(
    X: np.ndarray,
    y: np.ndarray,
    target_attack_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Duplicate attack-labeled windows to approach a target class ratio.

    All original windows are kept once (full timeline preserved at window level).
    Only positive windows are resampled with replacement; order is shuffled for training.

    Args:
        X: Window array (num_windows, window_size, num_features)
        y: Labels (num_windows,)
        target_attack_ratio: Desired fraction of attack windows after oversampling
        seed: RNG seed for reproducible duplicate selection and shuffle

    Returns:
        X_out, y_out, stats dict
    """
    y_flat = np.asarray(y).reshape(-1)
    n = len(y_flat)
    if n == 0:
        return X, y, {"applied": False, "reason": "empty"}

    if not 0.0 < target_attack_ratio < 1.0:
        raise ValueError("target_attack_ratio must be in (0, 1)")

    pos_idx = np.flatnonzero(y_flat >= 0.5)
    neg_idx = np.flatnonzero(y_flat < 0.5)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    base_stats = {
        "applied": False,
        "windows_before": n,
        "attack_before": int(n_pos),
        "attack_ratio_before": float(n_pos / n) if n else 0.0,
    }
    if n_pos == 0 or n_neg == 0:
        base_stats["reason"] = "single_class"
        return X, y, base_stats

    desired_pos = int(round(n_neg * target_attack_ratio / (1.0 - target_attack_ratio)))
    desired_pos = max(desired_pos, n_pos)

    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int64)
    if desired_pos > n_pos:
        extra = rng.choice(pos_idx, size=desired_pos - n_pos, replace=True)
        indices = np.concatenate([indices, extra])
    rng.shuffle(indices)

    X_out = X[indices]
    y_out = y_flat[indices]
    n_pos_after = int((y_out >= 0.5).sum())
    stats = {
        **base_stats,
        "applied": desired_pos > n_pos,
        "windows_after": len(y_out),
        "attack_after": n_pos_after,
        "attack_ratio_after": float(n_pos_after / len(y_out)),
        "duplicates_added": max(0, len(y_out) - n),
        "target_attack_ratio": target_attack_ratio,
    }
    if stats["applied"]:
        logger.info(
            "Oversampled attack windows: %d -> %d windows, attack %.1f%% -> %.1f%% "
            "(+%d duplicates, target=%.0f%%)",
            n, len(y_out),
            100.0 * stats["attack_ratio_before"],
            100.0 * stats["attack_ratio_after"],
            stats["duplicates_added"],
            100.0 * target_attack_ratio,
        )
    return X_out, y_out, stats


def split_temporal_windows(
    X: np.ndarray,
    y: np.ndarray,
    test_split: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Split window arrays into chronological train (prefix) and eval (suffix) portions.

    Args:
        X: (num_windows, window_size, features)
        y: (num_windows,)
        test_split: Fraction of windows reserved for eval (suffix)

    Returns:
        X_train, y_train, X_test, y_test, stats
    """
    y_flat = np.asarray(y).reshape(-1)
    n = len(y_flat)
    if n == 0:
        empty = np.array([])
        return X, y, empty, empty, {"applied": False, "reason": "empty"}

    n_test = int(n * test_split)
    n_train = n - n_test
    if n_test <= 0 or n_train <= 0:
        return (
            X, y_flat, np.array([]), np.array([]),
            {
                "applied": False,
                "reason": "insufficient_windows",
                "windows_total": n,
                "test_split": test_split,
            },
        )

    X_train, y_train = X[:n_train], y_flat[:n_train]
    X_test, y_test = X[n_train:], y_flat[n_train:]
    stats = {
        "applied": True,
        "windows_total": n,
        "windows_train": n_train,
        "windows_test": n_test,
        "test_split": test_split,
        "train_attack_ratio": float((y_train >= 0.5).mean()),
        "test_attack_ratio": float((y_test >= 0.5).mean()),
    }
    logger.info(
        "Temporal window split: train=%d (attack %.1f%%), test=%d (attack %.1f%%)",
        n_train, 100.0 * stats["train_attack_ratio"],
        n_test, 100.0 * stats["test_attack_ratio"],
    )
    return X_train, y_train, X_test, y_test, stats


def split_post_attack_windows(
    X: np.ndarray,
    y: np.ndarray,
    fallback_test_split: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Split at the first attack window: train = prefix before attack, eval = attack onward.

    Ensures eval contains attack labels when attacks occur mid-mission (unlike a
    pure temporal suffix split). Falls back to temporal suffix for benign missions.
    """
    y_flat = np.asarray(y).reshape(-1)
    n = len(y_flat)
    if n == 0:
        empty = np.array([])
        return X, y, empty, empty, {"applied": False, "reason": "empty", "mode": "post_attack"}

    pos_idx = np.flatnonzero(y_flat >= 0.5)
    if len(pos_idx) == 0:
        out = split_temporal_windows(X, y_flat, test_split=fallback_test_split)
        stats = out[4]
        stats["mode"] = "post_attack"
        stats["fallback"] = "temporal_suffix_benign"
        return out

    k = int(pos_idx[0])
    if k <= 0 or k >= n:
        out = split_temporal_windows(X, y_flat, test_split=fallback_test_split)
        stats = out[4]
        stats["mode"] = "post_attack"
        stats["fallback"] = "temporal_suffix_edge"
        return out

    X_train, y_train = X[:k], y_flat[:k]
    X_test, y_test = X[k:], y_flat[k:]
    stats = {
        "applied": True,
        "mode": "post_attack",
        "windows_total": n,
        "windows_train": k,
        "windows_test": n - k,
        "split_index": k,
        "train_attack_ratio": float((y_train >= 0.5).mean()),
        "test_attack_ratio": float((y_test >= 0.5).mean()),
    }
    logger.info(
        "Post-attack window split: train=%d (attack %.1f%%), test=%d (attack %.1f%%)",
        k, 100.0 * stats["train_attack_ratio"],
        n - k, 100.0 * stats["test_attack_ratio"],
    )
    return X_train, y_train, X_test, y_test, stats


class DataPreprocessorTimeSeries:
    """
    Handles all data preprocessing for GPS, IMU, and packet sensor fusion.
    
    Supports:
    - GPS + IMU (original functionality)
    - GPS + IMU + Packets (new)
    - Packets only (new)
    
    Simple and clean - no complex operations.
    """

    def __init__(
        self,
        gps_file: Optional[str] = None,
        imu_file: Optional[str] = None,
        packet_file: Optional[str] = None,
        gps_features: List[str] = None,
        imu_features: List[str] = None,
        packet_features: List[str] = None,
        timestamp_col: str = 'TimeUS',
        packet_timestamp_col: str = 'Timestamp',
        label_col: str = 'Label',
        sampling_strategy: str = 'downsample',
        use_labels: bool = False
    ):
        """
        Initialize data preprocessor.
        
        Args:
            gps_file: Path to GPS CSV file (optional)
            imu_file: Path to IMU CSV file (optional)
            packet_file: Path to labeled packet CSV file (optional)
            gps_features: List of GPS feature column names to use
            imu_features: List of IMU feature column names to use
            packet_features: List of packet feature column names to use
            timestamp_col: Name of timestamp column for GPS/IMU (default: 'TimeUS')
            packet_timestamp_col: Name of timestamp column for packets (default: 'Timestamp')
            label_col: Name of label column in packet data (default: 'Label')
            sampling_strategy: 'downsample' (IMU->GPS) or 'upsample' (GPS->IMU)
            use_labels: If True, use packet labels for supervised learning (y = labels)
                        If False, use next timestep prediction (y = next timestep features)
        """
        self.gps_file = gps_file
        self.imu_file = imu_file
        self.packet_file = packet_file
        self.timestamp_col = timestamp_col
        self.packet_timestamp_col = packet_timestamp_col
        self.label_col = label_col
        self.sampling_strategy = sampling_strategy
        self.use_labels = use_labels
        
        # Validate that at least one data source is provided
        if not gps_file and not imu_file and not packet_file:
            raise ValueError("At least one of gps_file, imu_file, or packet_file must be provided")
        
        # Default features
        self.gps_features = gps_features or ['Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ']
        self.imu_features = imu_features or ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']
        self.packet_features = packet_features or ['SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol']
        
        # Will be set during preprocessing
        self.feature_means = None
        self.feature_stds = None
        self.num_features = None
        self.has_labels = False
        
        logger.info(f"DataPreprocessorWithPackets initialized")
        if gps_file:
            logger.info(f"GPS file: {gps_file}")
            logger.info(f"GPS features: {self.gps_features}")
        if imu_file:
            logger.info(f"IMU file: {imu_file}")
            logger.info(f"IMU features: {self.imu_features}")
        if packet_file:
            logger.info(f"Packet file: {packet_file}")
            logger.info(f"Packet features: {self.packet_features}")
            logger.info(f"Use labels: {use_labels}")
        if gps_file and imu_file:
            logger.info(f"Sampling strategy: {self.sampling_strategy}")

    def load_gps_data(self) -> Optional[pd.DataFrame]:
        """Load GPS data from CSV file."""
        if self.gps_file is None:
            return None
        
        logger.info(f"Loading GPS data from {self.gps_file}")
        gps_df = pd.read_csv(self.gps_file)
        
        if self.timestamp_col not in gps_df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in GPS data")
        
        # Sort by timestamp
        gps_df = gps_df.sort_values(self.timestamp_col).reset_index(drop=True)
        gps_df[self.timestamp_col] = pd.to_numeric(gps_df[self.timestamp_col], errors='coerce')
        
        logger.info(f"GPS samples: {len(gps_df)}")
        return gps_df

    def load_imu_data(self) -> Optional[pd.DataFrame]:
        """Load IMU data from CSV file."""
        if self.imu_file is None:
            return None
        
        logger.info(f"Loading IMU data from {self.imu_file}")
        imu_df = pd.read_csv(self.imu_file)
        
        if self.timestamp_col not in imu_df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in IMU data")
        
        # Sort by timestamp
        imu_df = imu_df.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Filter IMU data - keep only one sensor (I=0) to avoid duplicates
        if 'I' in imu_df.columns:
            imu_df = imu_df[imu_df['I'] == 0].reset_index(drop=True)
        
        imu_df[self.timestamp_col] = pd.to_numeric(imu_df[self.timestamp_col], errors='coerce')
        
        logger.info(f"IMU samples: {len(imu_df)}")
        return imu_df

    def load_packet_data(self) -> Optional[pd.DataFrame]:
        """Load labeled packet data from CSV file."""
        if self.packet_file is None:
            return None
        
        logger.info(f"Loading packet data from {self.packet_file}")
        try:
            packet_df = pd.read_csv(self.packet_file)
            logger.info(f"Loaded {len(packet_df)} packet rows")
        except Exception as e:
            logger.error(f"Error loading packet file: {e}")
            raise
        
        # Try to identify timestamp column
        timestamp_col_actual = self.packet_timestamp_col
        if timestamp_col_actual not in packet_df.columns:
            for col in ['Timestamp', 'timestamp', 'TimeUS', 'time']:
                if col in packet_df.columns:
                    timestamp_col_actual = col
                    logger.info(f"Using timestamp column: {timestamp_col_actual}")
                    break
            else:
                # Assume first column is timestamp
                timestamp_col_actual = packet_df.columns[0]
                logger.info(f"Using first column as timestamp: {timestamp_col_actual}")
        
        # Try to identify label column
        label_col_actual = self.label_col
        if label_col_actual not in packet_df.columns:
            for col in ['Label', 'label', 'y']:
                if col in packet_df.columns:
                    label_col_actual = col
                    break
            if label_col_actual not in packet_df.columns:
                # Assume last column is label
                label_col_actual = packet_df.columns[-1]
                logger.info(f"Using last column as label: {label_col_actual}")
        
        # Convert timestamp to numeric if needed (for consistency with GPS/IMU)
        if not pd.api.types.is_numeric_dtype(packet_df[timestamp_col_actual]):
            try:
                # Try converting to datetime first, then to numeric (microseconds since epoch)
                packet_df[timestamp_col_actual] = pd.to_datetime(packet_df[timestamp_col_actual])
                packet_df[timestamp_col_actual] = (packet_df[timestamp_col_actual] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1us')
            except:
                # If that fails, try direct numeric conversion
                packet_df[timestamp_col_actual] = pd.to_numeric(packet_df[timestamp_col_actual], errors='coerce')
        
        # Sort by timestamp
        packet_df = packet_df.sort_values(timestamp_col_actual).reset_index(drop=True)
        packet_df[timestamp_col_actual] = pd.to_numeric(packet_df[timestamp_col_actual], errors='coerce')
        
        # Store actual column names for later use
        packet_df.attrs['timestamp_col'] = timestamp_col_actual
        packet_df.attrs['label_col'] = label_col_actual
        
        logger.info(f"Packet samples: {len(packet_df)}")
        if label_col_actual in packet_df.columns:
            label_counts = packet_df[label_col_actual].value_counts()
            logger.info(f"Label distribution: {dict(label_counts)}")
            self.has_labels = True
        
        return packet_df

    def merge_data(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Load and merge GPS, IMU, and packet data by timestamp.
        
        Returns:
            merged_df: Merged dataframe with all features
            labels: Optional array of labels (if packet data with labels is used)
        """
        gps_df = self.load_gps_data()
        imu_df = self.load_imu_data()
        packet_df = self.load_packet_data()
        
        labels = None
        
        # Merge GPS and IMU if both are provided
        if gps_df is not None and imu_df is not None:
            logger.info("Merging GPS and IMU data...")
            
            if self.sampling_strategy == 'downsample':
                # Downsample IMU to match GPS rate
                logger.info("Downsampling IMU data to GPS rate")
                merged_df = pd.merge_asof(
                    gps_df[[self.timestamp_col] + self.gps_features].sort_values(self.timestamp_col),
                    imu_df[[self.timestamp_col] + self.imu_features].sort_values(self.timestamp_col),
                    on=self.timestamp_col,
                    direction='nearest'
                )
            elif self.sampling_strategy == 'upsample':
                # Upsample GPS to match IMU rate
                logger.info("Upsampling GPS data to IMU rate")
                merged_df = pd.merge_asof(
                    imu_df[[self.timestamp_col] + self.imu_features].sort_values(self.timestamp_col),
                    gps_df[[self.timestamp_col] + self.gps_features].sort_values(self.timestamp_col),
                    on=self.timestamp_col,
                    direction='nearest'
                )
                # Reorder columns to match expected format
                merged_df = merged_df[[self.timestamp_col] + self.gps_features + self.imu_features]
            else:
                raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")
            
            # Drop rows with missing values
            merged_df = merged_df.dropna().reset_index(drop=True)
            logger.info(f"Merged GPS+IMU: {len(merged_df)} samples")
        
        elif gps_df is not None:
            # Only GPS
            merged_df = gps_df[[self.timestamp_col] + self.gps_features].copy()
            logger.info(f"Using GPS only: {len(merged_df)} samples")
        
        elif imu_df is not None:
            # Only IMU
            merged_df = imu_df[[self.timestamp_col] + self.imu_features].copy()
            logger.info(f"Using IMU only: {len(merged_df)} samples")
        
        else:
            # Only packets - create empty merged_df structure
            merged_df = pd.DataFrame()
        
        # Merge packet data if provided
        if packet_df is not None:
            packet_timestamp_col = packet_df.attrs.get('timestamp_col', self.packet_timestamp_col)
            packet_label_col = packet_df.attrs.get('label_col', self.label_col)
            
            # Extract packet features and labels
            packet_feature_cols = [col for col in self.packet_features if col in packet_df.columns]
            if not packet_feature_cols:
                # If specified features not found, use all columns except timestamp and label
                packet_feature_cols = [col for col in packet_df.columns 
                                      if col not in [packet_timestamp_col, packet_label_col]]
                logger.info(f"Using all available packet columns as features: {packet_feature_cols}")
            
            packet_features_df = packet_df[[packet_timestamp_col] + packet_feature_cols].copy()
            
            # Extract labels if present
            if packet_label_col in packet_df.columns and self.use_labels:
                labels = packet_df[packet_label_col].values
                logger.info(f"Extracted {len(labels)} labels from packet data")
            
            if len(merged_df) > 0:
                # Merge packets with existing GPS/IMU data
                logger.info("Merging packet data with GPS/IMU data...")
                
                # Rename packet timestamp column to match if different
                if packet_timestamp_col != self.timestamp_col:
                    packet_features_df = packet_features_df.rename(columns={packet_timestamp_col: self.timestamp_col})
                
                merged_df = pd.merge_asof(
                    merged_df.sort_values(self.timestamp_col),
                    packet_features_df.sort_values(self.timestamp_col),
                    on=self.timestamp_col,
                    direction='nearest',
                    suffixes=('', '_packet')
                )
                
                # Align labels with merged data if using labels
                if labels is not None:
                    # Map labels to merged timestamps using nearest neighbor
                    merged_labels = []
                    merged_timestamps = merged_df[self.timestamp_col].values
                    packet_timestamps = packet_df[packet_timestamp_col].values
                    
                    for ts in merged_timestamps:
                        # Find nearest packet timestamp
                        idx = np.abs(packet_timestamps - ts).argmin()
                        merged_labels.append(labels[idx])
                    labels = np.array(merged_labels)
            else:
                # Only packets - use packet data directly
                merged_df = packet_features_df.copy()
                merged_df = merged_df.rename(columns={packet_timestamp_col: self.timestamp_col})
                logger.info(f"Using packet data only: {len(merged_df)} samples")
        
        # Drop rows with missing values
        merged_df = merged_df.dropna().reset_index(drop=True)
        
        # Build final feature list from actual columns in merged_df
        # Exclude timestamp column
        all_feature_cols = [col for col in merged_df.columns if col != self.timestamp_col]
        
        # Try to maintain order: GPS -> IMU -> Packets
        feature_cols = []
        if gps_df is not None:
            # Add GPS features in order
            for feat in self.gps_features:
                if feat in all_feature_cols and feat not in feature_cols:
                    feature_cols.append(feat)
        
        if imu_df is not None:
            # Add IMU features in order
            for feat in self.imu_features:
                if feat in all_feature_cols and feat not in feature_cols:
                    feature_cols.append(feat)
        
        if packet_df is not None:
            # Add packet features (handle potential _packet suffix)
            for feat in self.packet_features:
                if feat in all_feature_cols and feat not in feature_cols:
                    feature_cols.append(feat)
                elif f"{feat}_packet" in all_feature_cols:
                    # Use the suffixed version
                    suffixed_feat = f"{feat}_packet"
                    if suffixed_feat not in feature_cols:
                        feature_cols.append(suffixed_feat)
            
            # Add any remaining packet columns that weren't in the specified list
            for col in all_feature_cols:
                if col not in feature_cols and (col.endswith('_packet') or 
                    (col not in self.gps_features and col not in self.imu_features)):
                    feature_cols.append(col)
        
        logger.info(f"Final merged data: {len(merged_df)} samples")
        logger.info(f"Features ({len(feature_cols)}): {feature_cols}")
        
        # Store feature columns for later use
        merged_df.attrs['feature_cols'] = feature_cols
        
        return merged_df, labels

    def normalize_data(self, data: np.ndarray, fit_rows: Optional[int] = None) -> np.ndarray:
        """
        Normalize data using z-score normalization.

        Args:
            data: Input data array
            fit_rows: If set, compute mean/std on data[:fit_rows] only (temporal train split)

        Returns:
            Normalized data
        """
        if self.feature_means is None:
            if fit_rows is not None and 0 < fit_rows < len(data):
                fit_data = data[:fit_rows]
                logger.info(
                    "Fitting normalization on first %d/%d timesteps (%.0f%%)",
                    fit_rows, len(data), 100.0 * fit_rows / len(data),
                )
            else:
                fit_data = data
            self.feature_means = np.mean(fit_data, axis=0)
            self.feature_stds = np.std(fit_data, axis=0)

            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1.0

            logger.info("Computed normalization statistics")

        # Normalize
        normalized = (data - self.feature_means) / self.feature_stds
        return normalized

    def create_sliding_windows(
        self,
        data: np.ndarray,
        window_size: int,
        overlap: int,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows with overlap for time series data.
        
        Args:
            data: Input data array (num_samples, num_features)
            window_size: Size of each window
            overlap: Number of overlapping samples between windows
            labels: Optional labels array (num_samples,) for supervised learning
        
        Returns:
            X: Windows of shape (num_windows, window_size, num_features)
            y: Targets of shape (num_windows, num_features) if labels=None (next timestep)
               or (num_windows,) if labels provided (classification)
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be >= 0 and < window_size")
        
        stride = window_size - overlap
        num_samples = len(data)
        
        # Calculate number of windows
        num_windows = (num_samples - window_size) // stride
        
        if num_windows <= 0:
            raise ValueError(
                f"Not enough data for windowing. Need at least {window_size + 1} samples, "
                f"got {num_samples}"
            )
        
        X = []
        y = []
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            # Input window
            X.append(data[start_idx:end_idx])
            
            # Target: next timestep features or label
            if labels is not None and self.use_labels:
                # Use label at the end of window (or next timestep)
                # Label sequence as 1 if any element in sequence has label 1
                window_labels = labels[start_idx:end_idx]
                sequence_label = 1 if np.any(window_labels == 1) else 0
                y.append(sequence_label)
            else:
                # Target: next timestep after window
                if end_idx < num_samples:
                    y.append(data[end_idx])
                else:
                    # Use last sample if we're at the end
                    y.append(data[-1])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32 if labels is None else np.int32)
        
        if labels is not None and self.use_labels:
            logger.info(f"Created {num_windows} windows with labels")
            logger.info(f"Label distribution: {np.bincount(y)}")
        else:
            logger.info(f"Created {num_windows} windows with next-timestep prediction")
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y

    def preprocess(
        self,
        window_size: int,
        overlap: int,
        norm_train_fraction: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Full preprocessing pipeline.

        Args:
            window_size: Size of sliding window
            overlap: Overlap between windows
            norm_train_fraction: If set, fit normalization on this fraction of
                leading timesteps only (e.g. 0.8 for an 80/20 temporal split)

        Returns:
            X: Windowed input data
            y: Target data (labels if use_labels=True, else next timestep features)
            num_features: Number of features per timestep
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Load and merge data
        logger.info("Step 1/4: Loading and merging data...")
        merged_df, labels = self.merge_data()
        logger.info(f"  Merged data: {len(merged_df)} samples")
        
        # Step 2: Extract feature columns
        logger.info("Step 2/4: Extracting feature columns...")
        # Use stored feature columns from merge_data if available
        if 'feature_cols' in merged_df.attrs:
            feature_cols = merged_df.attrs['feature_cols']
        else:
            # Fallback: get all columns except timestamp
            feature_cols = [col for col in merged_df.columns if col != self.timestamp_col]
        
        if not feature_cols:
            raise ValueError("No feature columns found in merged data")
        
        logger.info(f"  Using {len(feature_cols)} features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"  Using {len(feature_cols)} features: {feature_cols}")
        data = merged_df[feature_cols].values
        
        self.num_features = len(feature_cols)
        
        # Step 3: Normalize
        logger.info("Step 3/4: Normalizing data...")
        fit_rows = None
        if norm_train_fraction is not None and 0.0 < norm_train_fraction < 1.0:
            fit_rows = max(1, int(len(data) * norm_train_fraction))
        data = self.normalize_data(data, fit_rows=fit_rows)
        logger.info(f"  Normalized data shape: {data.shape}")
        
        # Step 4: Create sliding windows
        logger.info(f"Step 4/4: Creating sliding windows (size={window_size}, overlap={overlap})...")
        X, y = self.create_sliding_windows(data, window_size, overlap, labels)
        logger.info(f"  Created {len(X)} windows")
        
        logger.info("✓ Preprocessing complete!")
        
        return X, y, self.num_features

    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalization parameters for later use."""
        return self.feature_means, self.feature_stds

    def set_normalization_params(self, means: np.ndarray, stds: np.ndarray) -> None:
        """Apply normalization statistics from another preprocessor or shared fit."""
        self.feature_means = np.asarray(means, dtype=np.float64).copy()
        self.feature_stds = np.asarray(stds, dtype=np.float64).copy()
        self.feature_stds[self.feature_stds == 0] = 1.0
