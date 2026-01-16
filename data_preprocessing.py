"""
Data Preprocessing Module for GPS and IMU Data Fusion

This module handles:
1. Loading GPS and IMU data from CSV files
2. Downsampling IMU data to match GPS sampling rate
3. Feature concatenation
4. Normalization and preprocessing
5. Sliding window generation with overlap

Clean, modular design with no complex operations.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing for GPS and IMU sensor fusion.

    Simple and clean - no complex operations.
    """

    def __init__(
        self,
        gps_file: str,
        imu_file: str,
        gps_features: List[str] = None,
        imu_features: List[str] = None,
        timestamp_col: str = 'TimeUS',
        sampling_strategy: str = 'downsample'
    ):
        """
        Initialize data preprocessor.

        Args:
            gps_file: Path to GPS CSV file
            imu_file: Path to IMU CSV file
            gps_features: List of GPS feature column names to use
            imu_features: List of IMU feature column names to use
            timestamp_col: Name of timestamp column (default: 'TimeUS')
            sampling_strategy: 'downsample' (IMU->GPS) or 'upsample' (GPS->IMU)
        """
        self.gps_file = gps_file
        self.imu_file = imu_file
        self.timestamp_col = timestamp_col
        self.sampling_strategy = sampling_strategy

        # Default features from your data structure
        self.gps_features = gps_features or ['Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ']
        self.imu_features = imu_features or ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']

        # Will be set during preprocessing
        self.feature_means = None
        self.feature_stds = None
        self.num_features = None

        logger.info(f"DataPreprocessor initialized")
        logger.info(f"Sampling strategy: {self.sampling_strategy}")
        logger.info(f"GPS features: {self.gps_features}")
        logger.info(f"IMU features: {self.imu_features}")

    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load GPS and IMU data, apply sampling strategy, and merge.

        Returns:
            Merged dataframe with all features
        """
        # Load data
        logger.info(f"Loading GPS data from {self.gps_file}")
        gps_df = pd.read_csv(self.gps_file)

        logger.info(f"Loading IMU data from {self.imu_file}")
        imu_df = pd.read_csv(self.imu_file)

        # Ensure timestamp column exists
        if self.timestamp_col not in gps_df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in GPS data")
        if self.timestamp_col not in imu_df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in IMU data")

        # Sort by timestamp
        gps_df = gps_df.sort_values(self.timestamp_col).reset_index(drop=True)
        imu_df = imu_df.sort_values(self.timestamp_col).reset_index(drop=True)

        # Filter IMU data - keep only one sensor (I=0) to avoid duplicates
        if 'I' in imu_df.columns:
            imu_df = imu_df[imu_df['I'] == 0].reset_index(drop=True)

        logger.info(f"GPS samples: {len(gps_df)}, IMU samples: {len(imu_df)}")

        # Convert timestamps to numeric
        gps_df[self.timestamp_col] = pd.to_numeric(gps_df[self.timestamp_col])
        imu_df[self.timestamp_col] = pd.to_numeric(imu_df[self.timestamp_col])

        # Apply sampling strategy
        if self.sampling_strategy == 'downsample':
            # Downsample IMU to match GPS rate (fewer samples)
            logger.info("Downsampling IMU data to GPS rate")
            merged_df = pd.merge_asof(
                gps_df[[self.timestamp_col] + self.gps_features],
                imu_df[[self.timestamp_col] + self.imu_features],
                on=self.timestamp_col,
                direction='nearest'
            )
        elif self.sampling_strategy == 'upsample':
            # Upsample GPS to match IMU rate (more samples)
            logger.info("Upsampling GPS data to IMU rate")
            merged_df = pd.merge_asof(
                imu_df[[self.timestamp_col] + self.imu_features],
                gps_df[[self.timestamp_col] + self.gps_features],
                on=self.timestamp_col,
                direction='nearest'
            )
            # Reorder columns to match expected format
            merged_df = merged_df[[self.timestamp_col] + self.gps_features + self.imu_features]
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}. Use 'downsample' or 'upsample'")

        # Drop rows with missing values
        merged_df = merged_df.dropna().reset_index(drop=True)

        logger.info(f"Merged data: {len(merged_df)} samples")
        logger.info(f"Features: {self.gps_features + self.imu_features}")

        return merged_df

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using z-score normalization.

        Args:
            data: Input data array

        Returns:
            Normalized data
        """
        # Calculate mean and std if not already done
        if self.feature_means is None:
            self.feature_means = np.mean(data, axis=0)
            self.feature_stds = np.std(data, axis=0)

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
        overlap: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows with overlap for time series data.

        Args:
            data: Input data array (num_samples, num_features)
            window_size: Size of each window
            overlap: Number of overlapping samples between windows

        Returns:
            X: Windows of shape (num_windows, window_size, num_features)
            y: Targets of shape (num_windows, num_features) - next timestep
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

            # Target: next timestep after window
            y.append(data[end_idx])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(f"Created {num_windows} windows with size={window_size}, overlap={overlap}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def preprocess(
        self,
        window_size: int,
        overlap: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Full preprocessing pipeline.

        Args:
            window_size: Size of sliding window
            overlap: Overlap between windows

        Returns:
            X: Windowed input data
            y: Target data
            num_features: Number of features per timestep
        """
        # Step 1: Load and merge data
        merged_df = self.load_and_merge_data()

        # Step 2: Extract feature columns
        feature_cols = self.gps_features + self.imu_features
        data = merged_df[feature_cols].values

        self.num_features = len(feature_cols)

        # Step 3: Normalize
        data = self.normalize_data(data)

        # Step 4: Create sliding windows
        X, y = self.create_sliding_windows(data, window_size, overlap)

        logger.info("Preprocessing complete!")

        return X, y, self.num_features

    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalization parameters for later use."""
        return self.feature_means, self.feature_stds
