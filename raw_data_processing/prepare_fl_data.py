# -*- coding: utf-8 -*-
"""
Script to prepare labeled GPS or IMU data for Federated Learning (FL) with LSTM models.
- Loads labeled CSV data (labels in last column, timestamp in first column)
- Creates sequences of specified length for LSTM training
- Labels sequences as 1 if any element in the sequence has label 1
- Splits data into training and testing sets
- Divides data among FL clients with consecutive time periods
- Saves prepared data for each client
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse


def load_labeled_data(csv_file_path):
    """
    Load labeled CSV data.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the labeled CSV file (labels in last column)
    
    Returns:
    --------
    data : numpy.ndarray
        Feature data (all columns except the last one)
    labels : numpy.ndarray
        Labels (last column)
    """
    print(f"Loading labeled data from {csv_file_path}...")
    
    # Try reading without header first
    try:
        df = pd.read_csv(csv_file_path, header=None)
    except:
        # Try with header
        df = pd.read_csv(csv_file_path)
    
    print(f"Data shape: {df.shape}")
    
    # Ensure data is sorted by time (assuming rows are already in chronological order)
    # If not, we'll keep the original order which should be chronological
    
    # Last column is the label
    labels = df.iloc[:, -1].values.astype(int)
    
    # Features: all columns except first (timestamp) and last (label)
    # After filtering in label_csv_with_periods.py, format is:
    # Column 0: timestamp, Columns 1 to N-1: features, Column N: label
    if df.shape[1] > 2:
        # Skip first column (timestamp) and last column (label)
        features = df.iloc[:, 1:-1].values.astype(float)
        print(f"Note: Skipping first column (timestamp) and using columns 1 to {df.shape[1]-2} as features")
    else:
        # Fallback: if only 2 columns, assume first is feature and second is label
        features = df.iloc[:, 0:-1].values.astype(float)
        print(f"Note: Using first column as features (assuming no timestamp column)")
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    print("Note: Data is assumed to be in chronological order (time-ordered)")
    
    return features, labels


def create_sequences(features, labels, sequence_length, stride=1):
    """
    Create sequences from time series data for LSTM training.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature data (n_samples, n_features)
    labels : numpy.ndarray
        Labels (n_samples,)
    sequence_length : int
        Length of each sequence
    stride : int
        Step size for creating sequences (default: 1 for overlapping sequences)
    
    Returns:
    --------
    X : numpy.ndarray
        Sequences of shape (n_sequences, sequence_length, n_features)
    y : numpy.ndarray
        Sequence labels of shape (n_sequences,)
        Label is 1 if any element in the sequence has label 1, else 0
    """
    print(f"\nCreating sequences with length {sequence_length} and stride {stride}...")
    
    n_samples, n_features = features.shape
    n_sequences = (n_samples - sequence_length) // stride + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough samples ({n_samples}) to create sequences of length {sequence_length}")
    
    X = np.zeros((n_sequences, sequence_length, n_features))
    y = np.zeros(n_sequences, dtype=int)
    
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        
        # Extract sequence
        X[i] = features[start_idx:end_idx]
        
        # Label sequence: 1 if any element in sequence has label 1
        sequence_labels = labels[start_idx:end_idx]
        y[i] = 1 if np.any(sequence_labels == 1) else 0
    
    print(f"Created {n_sequences} sequences")
    print(f"Sequence shape: {X.shape}")
    print(f"Sequence label distribution: {np.bincount(y)}")
    
    return X, y


def split_train_test(X, y, test_size=0.0, random_state=42, time_based=True):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature sequences (assumed to be in chronological order)
    y : numpy.ndarray
        Labels
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42, only used if time_based=False)
    time_based : bool
        If True, split by time (take test from end). If False, random split (default: True)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Training and testing sets
    """
    print(f"\nSplitting data into train/test sets (test_size={test_size}, time_based={time_based})...")
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if time_based:
        # Time-based split: take test from the end to preserve time order
        X_train = X[:n_train]
        X_test = X[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
    else:
        # Random split (for comparison, but not recommended for time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    print(f"Training set: {X_train.shape[0]} sequences (time period: first {n_train} sequences)")
    print(f"  Label distribution: {np.bincount(y_train)}")
    print(f"Testing set: {X_test.shape[0]} sequences (time period: last {n_test} sequences)")
    print(f"  Label distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test


def divide_among_clients(X_train, y_train, n_clients, distribution='consecutive'):
    """
    Divide training data among FL clients with consecutive time periods.
    Each client gets a consecutive chunk of time-ordered data.
    All clients (except possibly the last) get the same number of sequences.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training sequences (assumed to be in chronological order)
    y_train : numpy.ndarray
        Training labels
    n_clients : int
        Number of FL clients
    distribution : str
        Distribution strategy:
        - 'consecutive': Divide into consecutive time periods (default)
        - 'iid': Random shuffle (not recommended for time series)
        - 'non-iid': Sort by label (not recommended for time series)
    
    Returns:
    --------
    client_data : list of dict
        List of dictionaries, each containing 'X' and 'y' for a client
    """
    print(f"\nDividing data among {n_clients} clients (distribution: {distribution})...")
    print("Ensuring each client gets consecutive time periods with equal duration...")
    
    n_samples = len(X_train)
    
    if distribution == 'consecutive':
        # Divide into consecutive chunks - each client gets a time period
        # Calculate samples per client to ensure equal duration
        samples_per_client = n_samples // n_clients
        
        # Calculate remainder to distribute
        remainder = n_samples % n_clients
        
        client_data = []
        current_idx = 0
        
        for client_id in range(n_clients):
            # Distribute remainder samples to first 'remainder' clients
            if client_id < remainder:
                client_size = samples_per_client + 1
            else:
                client_size = samples_per_client
            
            end_idx = current_idx + client_size
            
            # Extract consecutive chunk
            client_X = X_train[current_idx:end_idx]
            client_y = y_train[current_idx:end_idx]
            
            client_data.append({
                'X': client_X,
                'y': client_y,
                'start_idx': current_idx,
                'end_idx': end_idx
            })
            
            print(f"  Client {client_id}: {len(client_X)} sequences "
                  f"(indices {current_idx} to {end_idx-1}), "
                  f"label distribution: {np.bincount(client_y)}")
            
            current_idx = end_idx
        
        # Verify all clients have similar duration
        client_sizes = [len(c['X']) for c in client_data]
        min_size = min(client_sizes)
        max_size = max(client_sizes)
        print(f"\n  Client size range: {min_size} to {max_size} sequences")
        if max_size - min_size > 1:
            print(f"  Warning: Size difference > 1 due to remainder distribution")
        else:
            print(f"  All clients have equal or near-equal duration")
    
    elif distribution == 'iid':
        # Random shuffle (not recommended for time series)
        print("  Warning: Random distribution breaks time continuity!")
        indices = np.random.permutation(n_samples)
        samples_per_client = n_samples // n_clients
        
        client_data = []
        for client_id in range(n_clients):
            start_idx = client_id * samples_per_client
            if client_id == n_clients - 1:
                end_idx = n_samples
            else:
                end_idx = (client_id + 1) * samples_per_client
            
            client_indices = indices[start_idx:end_idx]
            client_X = X_train[client_indices]
            client_y = y_train[client_indices]
            
            client_data.append({
                'X': client_X,
                'y': client_y
            })
            
            print(f"  Client {client_id}: {len(client_X)} samples, "
                  f"label distribution: {np.bincount(client_y)}")
    
    elif distribution == 'non-iid':
        # Sort by label (not recommended for time series)
        print("  Warning: Label-based distribution breaks time continuity!")
        indices = np.argsort(y_train)
        samples_per_client = n_samples // n_clients
        
        client_data = []
        for client_id in range(n_clients):
            start_idx = client_id * samples_per_client
            if client_id == n_clients - 1:
                end_idx = n_samples
            else:
                end_idx = (client_id + 1) * samples_per_client
            
            client_indices = indices[start_idx:end_idx]
            client_X = X_train[client_indices]
            client_y = y_train[client_indices]
            
            client_data.append({
                'X': client_X,
                'y': client_y
            })
            
            print(f"  Client {client_id}: {len(client_X)} samples, "
                  f"label distribution: {np.bincount(client_y)}")
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution}. "
                        f"Use 'consecutive', 'iid', or 'non-iid'")
    
    return client_data


def normalize_features(X_train, X_test, client_data):
    """
    Normalize features using StandardScaler fitted on training data.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training sequences
    X_test : numpy.ndarray
        Testing sequences
    client_data : list of dict
        Client data dictionaries
    
    Returns:
    --------
    X_train_norm, X_test_norm, client_data_norm : tuple
        Normalized data
    """
    print("\nNormalizing features...")
    
    # Reshape for scaling: (n_samples, sequence_length * n_features)
    n_samples_train, seq_len, n_features = X_train.shape
    n_samples_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(n_samples_train, -1)
    X_test_flat = X_test.reshape(n_samples_test, -1)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_flat_norm = scaler.fit_transform(X_train_flat)
    X_test_flat_norm = scaler.transform(X_test_flat)
    
    # Reshape back
    X_train_norm = X_train_flat_norm.reshape(n_samples_train, seq_len, n_features)
    X_test_norm = X_test_flat_norm.reshape(n_samples_test, seq_len, n_features)
    
    # Normalize client data
    client_data_norm = []
    for client_dict in client_data:
        client_X = client_dict['X']
        n_samples_client = client_X.shape[0]
        client_X_flat = client_X.reshape(n_samples_client, -1)
        client_X_flat_norm = scaler.transform(client_X_flat)
        client_X_norm = client_X_flat_norm.reshape(n_samples_client, seq_len, n_features)
        
        client_data_norm.append({
            'X': client_X_norm,
            'y': client_dict['y']
        })
    
    print("Normalization complete")
    
    return X_train_norm, X_test_norm, client_data_norm, scaler


def save_client_data(client_data, output_dir, prefix='client'):
    """
    Save client data to numpy files.
    
    Parameters:
    -----------
    client_data : list of dict
        List of client data dictionaries
    output_dir : str
        Output directory for saving files
    prefix : str
        Prefix for output files (default: 'client')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving client data to {output_dir}...")
    
    for client_id, client_dict in enumerate(client_data):
        client_X = client_dict['X']
        client_y = client_dict['y']
        
        # Save as .npz file
        client_file = os.path.join(output_dir, f"{prefix}_{client_id:03d}.npz")
        
        # Include metadata if available
        save_dict = {'X': client_X, 'y': client_y}
        if 'start_idx' in client_dict:
            save_dict['start_idx'] = client_dict['start_idx']
        if 'end_idx' in client_dict:
            save_dict['end_idx'] = client_dict['end_idx']
        
        np.savez(client_file, **save_dict)
        print(f"  Saved {client_file} (shape: {client_X.shape})")
    
    print(f"Saved data for {len(client_data)} clients")


def save_test_data(X_test, y_test, output_dir, filename='test_data.npz'):
    """
    Save test data to numpy file.
    
    Parameters:
    -----------
    X_test : numpy.ndarray
        Test sequences
    y_test : numpy.ndarray
        Test labels
    output_dir : str
        Output directory
    filename : str
        Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    test_file = os.path.join(output_dir, filename)
    np.savez(test_file, X=X_test, y=y_test)
    print(f"\nSaved test data to {test_file} (shape: {X_test.shape})")


def save_scaler(scaler, output_dir, filename='scaler.npz'):
    """
    Save scaler parameters for later use.
    
    Parameters:
    -----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    output_dir : str
        Output directory
    filename : str
        Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scaler_file = os.path.join(output_dir, filename)
    np.savez(scaler_file, mean=scaler.mean_, scale=scaler.scale_)
    print(f"Saved scaler to {scaler_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare labeled GPS or IMU data for Federated Learning with LSTM'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to labeled CSV file (labels in last column)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for prepared data'
    )
    parser.add_argument(
        '--sequence-length', '-s',
        type=int,
        default=10,
        help='Sequence length for LSTM (default: 10)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Stride for creating sequences (default: 1)'
    )
    parser.add_argument(
        '--n-clients', '-n',
        type=int,
        default=5,
        help='Number of FL clients (default: 5)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.0,#0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--distribution',
        type=str,
        choices=['consecutive', 'iid', 'non-iid'],
        default='consecutive',
        help='Data distribution strategy: consecutive (time-ordered chunks), iid (random), or non-iid (sorted by label) (default: consecutive)'
    )
    parser.add_argument(
        '--random-split',
        action='store_true',
        help='Use random train/test split instead of time-based (default: time-based)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize features using StandardScaler'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_state)
    
    # Load labeled data
    features, labels = load_labeled_data(args.input)
    
    # Create sequences
    X, y = create_sequences(features, labels, args.sequence_length, args.stride)
    
    # Split into train/test (time-based by default to preserve time order)
    time_based = not args.random_split
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=args.test_size, random_state=args.random_state, 
        time_based=time_based
    )
    
    # Divide among clients
    client_data = divide_among_clients(
        X_train, y_train, args.n_clients, distribution=args.distribution
    )
    
    # Normalize if requested
    if args.normalize:
        X_train, X_test, client_data, scaler = normalize_features(
            X_train, X_test, client_data
        )
        save_scaler(scaler, args.output)
    
    # Save client data
    save_client_data(client_data, args.output)
    
    # Save test data
    save_test_data(X_test, y_test, args.output)
    
    print("\n" + "="*80)
    print("Data preparation complete!")
    print(f"Output directory: {args.output}")
    print(f"  - {args.n_clients} client files (client_XXX.npz)")
    print(f"  - 1 test file (test_data.npz)")
    if args.normalize:
        print(f"  - 1 scaler file (scaler.npz)")
    print("="*80)


if __name__ == "__main__":
    main()
