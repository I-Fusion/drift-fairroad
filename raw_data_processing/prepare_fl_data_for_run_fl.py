# -*- coding: utf-8 -*-
"""
Script to prepare GPS, IMU, and labeled packet data for Federated Learning System (run_fl_system.py).

This version:
- Loads GPS and IMU CSV files (same format as expected by run_fl_system.py)
- Optionally loads labeled packet sequence CSV files
- Merges and aligns GPS/IMU data by timestamp
- Splits data into consecutive time periods for each FL client
- Saves separate GPS, IMU, and packet CSV files for each client
- Creates a client mapping file for easy configuration

The prepared data can be directly used by run_fl_system.py by pointing each client
to its respective GPS, IMU, and packet files.

Labeled packet sequences should have:
- Timestamp in first column (or 'Timestamp' column)
- Feature columns: SrcPort, DstPort, Length, MsgID, Protocol
- Label in last column (or 'Label' column)

Sample usage at the project root directory: 
[1] python .\raw_data_processing\prepare_fl_data_for_run_fl.py --gps-file .\data\waypoint_injection\mission_2_wp_23_attack_add_wp_5_alt_0005_gps.csv --imu-file .\data\waypoint_injection\mission_2_wp_23_attack_add_wp_5_alt_0005_imu.csv --output .\data\train\gps-imu
[2] python .\raw_data_processing\prepare_fl_data_for_run_fl.py --packet-file .\data\network_packets\mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv --output .\data\train\packets
"""

import numpy as np
import pandas as pd
import os
import argparse
from typing import Tuple, List, Dict


def load_gps_data(gps_file: str, timestamp_col: str = 'TimeUS') -> pd.DataFrame:
    """
    Load GPS CSV file.
    
    Parameters:
    -----------
    gps_file : str
        Path to GPS CSV file
    timestamp_col : str
        Name of timestamp column (default: 'TimeUS')
    
    Returns:
    --------
    gps_df : pd.DataFrame
        GPS dataframe
    """
    print(f"Loading GPS data from {gps_file}...")
    gps_df = pd.read_csv(gps_file)
    
    if timestamp_col not in gps_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in GPS data")
    
    # Sort by timestamp
    gps_df = gps_df.sort_values(timestamp_col).reset_index(drop=True)
    gps_df[timestamp_col] = pd.to_numeric(gps_df[timestamp_col], errors='coerce')
    
    print(f"GPS samples: {len(gps_df)}")
    print(f"GPS columns: {list(gps_df.columns)}")
    
    return gps_df


def load_packet_data(packet_file: str, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
    """
    Load labeled packet CSV file.
    
    Parameters:
    -----------
    packet_file : str
        Path to labeled packet CSV file
    timestamp_col : str
        Name of timestamp column (default: 'Timestamp')
    
    Returns:
    --------
    packet_df : pd.DataFrame
        Packet dataframe with labels
    """
    print(f"Loading packet data from {packet_file}...")
    packet_df = pd.read_csv(packet_file)
    
    # Try to identify timestamp column
    if timestamp_col not in packet_df.columns:
        # Try common timestamp column names
        for col in ['Timestamp', 'timestamp', 'TimeUS', 'time']:
            if col in packet_df.columns:
                timestamp_col = col
                print(f"Using timestamp column: {timestamp_col}")
                break
        else:
            # Assume first column is timestamp
            timestamp_col = packet_df.columns[0]
            print(f"Using first column as timestamp: {timestamp_col}")
    
    # Try to identify label column
    label_col = None
    for col in ['Label', 'label', 'y']:
        if col in packet_df.columns:
            label_col = col
            break
    if label_col is None:
        # Assume last column is label
        label_col = packet_df.columns[-1]
        print(f"Using last column as label: {label_col}")
    
    # Convert timestamp to numeric if needed (for consistency with GPS/IMU)
    if not pd.api.types.is_numeric_dtype(packet_df[timestamp_col]):
        try:
            # Try converting to datetime first, then to numeric (microseconds since epoch)
            packet_df[timestamp_col] = pd.to_datetime(packet_df[timestamp_col])
            packet_df[timestamp_col] = (packet_df[timestamp_col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1us')
        except:
            # If that fails, try direct numeric conversion
            packet_df[timestamp_col] = pd.to_numeric(packet_df[timestamp_col], errors='coerce')
    
    # Sort by timestamp
    packet_df = packet_df.sort_values(timestamp_col).reset_index(drop=True)
    packet_df[timestamp_col] = pd.to_numeric(packet_df[timestamp_col], errors='coerce')
    
    print(f"Packet samples: {len(packet_df)}")
    print(f"Packet columns: {list(packet_df.columns)}")
    if label_col in packet_df.columns:
        label_counts = packet_df[label_col].value_counts()
        print(f"Label distribution: {dict(label_counts)}")
    
    return packet_df


def load_imu_data(imu_file: str, timestamp_col: str = 'TimeUS') -> pd.DataFrame:
    """
    Load IMU CSV file.
    
    Parameters:
    -----------
    imu_file : str
        Path to IMU CSV file
    timestamp_col : str
        Name of timestamp column (default: 'TimeUS')
    
    Returns:
    --------
    imu_df : pd.DataFrame
        IMU dataframe
    """
    print(f"Loading IMU data from {imu_file}...")
    imu_df = pd.read_csv(imu_file)
    
    if timestamp_col not in imu_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in IMU data")
    
    # Filter IMU data - keep only one sensor (I=0) to avoid duplicates
    if 'I' in imu_df.columns:
        imu_df = imu_df[imu_df['I'] == 0].reset_index(drop=True)
        print(f"Filtered IMU data to sensor I=0")
    
    # Sort by timestamp
    imu_df = imu_df.sort_values(timestamp_col).reset_index(drop=True)
    imu_df[timestamp_col] = pd.to_numeric(imu_df[timestamp_col], errors='coerce')
    
    print(f"IMU samples: {len(imu_df)}")
    print(f"IMU columns: {list(imu_df.columns)}")
    
    return imu_df


def merge_gps_imu(
    gps_df: pd.DataFrame,
    imu_df: pd.DataFrame,
    timestamp_col: str = 'TimeUS',
    sampling_strategy: str = 'downsample'
) -> pd.DataFrame:
    """
    Merge GPS and IMU data by timestamp.
    
    Parameters:
    -----------
    gps_df : pd.DataFrame
        GPS dataframe
    imu_df : pd.DataFrame
        IMU dataframe
    timestamp_col : str
        Timestamp column name
    sampling_strategy : str
        'downsample' (IMU->GPS) or 'upsample' (GPS->IMU)
    
    Returns:
    --------
    merged_df : pd.DataFrame
        Merged dataframe with aligned timestamps
    """
    print(f"\nMerging GPS and IMU data (strategy: {sampling_strategy})...")
    
    if sampling_strategy == 'downsample':
        # Downsample IMU to match GPS rate
        merged_df = pd.merge_asof(
            gps_df.sort_values(timestamp_col),
            imu_df.sort_values(timestamp_col),
            on=timestamp_col,
            direction='nearest',
            suffixes=('', '_imu')
        )
    elif sampling_strategy == 'upsample':
        # Upsample GPS to match IMU rate
        merged_df = pd.merge_asof(
            imu_df.sort_values(timestamp_col),
            gps_df.sort_values(timestamp_col),
            on=timestamp_col,
            direction='nearest',
            suffixes=('_imu', '')
        )
        # Reorder to have GPS columns first
        gps_cols = [col for col in merged_df.columns if not col.endswith('_imu')]
        imu_cols = [col for col in merged_df.columns if col.endswith('_imu')]
        merged_df = merged_df[gps_cols + imu_cols]
    else:
        raise ValueError(f"Invalid sampling_strategy: {sampling_strategy}")
    
    # Drop rows with missing values
    merged_df = merged_df.dropna().reset_index(drop=True)
    
    print(f"Merged data: {len(merged_df)} samples")
    print(f"Time range: {merged_df[timestamp_col].min()} to {merged_df[timestamp_col].max()}")
    
    return merged_df


def split_among_clients(
    gps_df: pd.DataFrame = None,
    imu_df: pd.DataFrame = None,
    packet_df: pd.DataFrame = None,
    merged_df: pd.DataFrame = None,
    n_clients: int = 3,
    timestamp_col: str = 'TimeUS',
    packet_timestamp_col: str = 'Timestamp',
    sampling_strategy: str = 'downsample'
) -> List[Dict]:
    """
    Split GPS, IMU, and packet data among clients with consecutive time periods.
    
    Parameters:
    -----------
    gps_df : pd.DataFrame, optional
        Original GPS dataframe
    imu_df : pd.DataFrame, optional
        Original IMU dataframe
    packet_df : pd.DataFrame, optional
        Packet dataframe with labels
    merged_df : pd.DataFrame, optional
        Merged GPS/IMU dataframe (for determining split points)
        If None and packet_df is provided, uses packet_df for split points
    n_clients : int
        Number of clients
    timestamp_col : str
        Timestamp column name for GPS/IMU
    packet_timestamp_col : str
        Timestamp column name for packets
    sampling_strategy : str
        Sampling strategy used in merge
    
    Returns:
    --------
    client_data : list of dict
        List of dictionaries, each containing 'gps', 'imu', and/or 'packet' dataframes
    """
    print(f"\nSplitting data among {n_clients} clients...")
    
    # Determine which dataframe to use for split points
    if merged_df is not None:
        # Use merged GPS/IMU data for split points
        split_df = merged_df.sort_values(timestamp_col).reset_index(drop=True)
        split_timestamp_col = timestamp_col
    elif packet_df is not None:
        # Use packet data for split points
        split_df = packet_df.sort_values(packet_timestamp_col).reset_index(drop=True)
        split_timestamp_col = packet_timestamp_col
    elif gps_df is not None:
        # Use GPS data for split points
        split_df = gps_df.sort_values(timestamp_col).reset_index(drop=True)
        split_timestamp_col = timestamp_col
    else:
        raise ValueError("At least one of merged_df, packet_df, or gps_df must be provided")
    
    n_samples = len(split_df)
    
    # Calculate samples per client
    samples_per_client = n_samples // n_clients
    remainder = n_samples % n_clients
    
    client_data = []
    
    for client_id in range(n_clients):
        # Calculate time range for this client
        start_idx = client_id * samples_per_client + min(client_id, remainder)
        if client_id < remainder:
            end_idx = start_idx + samples_per_client + 1
        else:
            end_idx = start_idx + samples_per_client
        
        # Get split data for this client
        client_split = split_df.iloc[start_idx:end_idx].copy()
        client_timestamps = client_split[split_timestamp_col]
        min_timestamp = client_timestamps.min()
        max_timestamp = client_timestamps.max()
        
        client_dict = {
            'client_id': client_id,
            'start_timestamp': min_timestamp,
            'end_timestamp': max_timestamp,
            'num_samples': len(client_timestamps)
        }
        
        # Extract GPS data for this time range
        if gps_df is not None:
            time_buffer = (max_timestamp - min_timestamp) * 0.1 if max_timestamp > min_timestamp else 0
            client_gps = gps_df[
                (gps_df[timestamp_col] >= min_timestamp - time_buffer) & 
                (gps_df[timestamp_col] <= max_timestamp + time_buffer)
            ].copy()
            
            if len(client_gps) == 0:
                print(f"  Warning: Client {client_id} has no GPS data, using nearest samples")
                nearest_idx = gps_df[timestamp_col].sub(min_timestamp).abs().argsort()[:max(10, samples_per_client)]
                client_gps = gps_df.iloc[nearest_idx].copy()
            
            client_gps = client_gps.sort_values(timestamp_col).reset_index(drop=True)
            client_dict['gps'] = client_gps
        
        # Extract IMU data for this time range
        if imu_df is not None:
            time_buffer = (max_timestamp - min_timestamp) * 0.1 if max_timestamp > min_timestamp else 0
            client_imu = imu_df[
                (imu_df[timestamp_col] >= min_timestamp - time_buffer) & 
                (imu_df[timestamp_col] <= max_timestamp + time_buffer)
            ].copy()
            
            if len(client_imu) == 0:
                print(f"  Warning: Client {client_id} has no IMU data, using nearest samples")
                nearest_idx = imu_df[timestamp_col].sub(min_timestamp).abs().argsort()[:max(10, samples_per_client)]
                client_imu = imu_df.iloc[nearest_idx].copy()
            
            client_imu = client_imu.sort_values(timestamp_col).reset_index(drop=True)
            client_dict['imu'] = client_imu
        
        # Extract packet data for this time range
        if packet_df is not None:
            time_buffer = (max_timestamp - min_timestamp) * 0.1 if max_timestamp > min_timestamp else 0
            client_packet = packet_df[
                (packet_df[packet_timestamp_col] >= min_timestamp - time_buffer) & 
                (packet_df[packet_timestamp_col] <= max_timestamp + time_buffer)
            ].copy()
            
            if len(client_packet) == 0:
                print(f"  Warning: Client {client_id} has no packet data, using nearest samples")
                nearest_idx = packet_df[packet_timestamp_col].sub(min_timestamp).abs().argsort()[:max(10, samples_per_client)]
                client_packet = packet_df.iloc[nearest_idx].copy()
            
            client_packet = client_packet.sort_values(packet_timestamp_col).reset_index(drop=True)
            client_dict['packet'] = client_packet
        
        client_data.append(client_dict)
        
        # Print summary
        summary_parts = []
        if 'gps' in client_dict:
            summary_parts.append(f"{len(client_dict['gps'])} GPS")
        if 'imu' in client_dict:
            summary_parts.append(f"{len(client_dict['imu'])} IMU")
        if 'packet' in client_dict:
            summary_parts.append(f"{len(client_dict['packet'])} packets")
        summary = ", ".join(summary_parts)
        print(f"  Client {client_id}: {summary} (time: {min_timestamp:.0f} to {max_timestamp:.0f})")

    # ------------------------------------------------------------------
    # Optional post-processing: equalize packet sample counts per client
    #
    # For classification with packets only, different clients can end up
    # with slightly different numbers of packet rows, which leads to a
    # different number of sliding windows per client. To keep all clients
    # aligned (same number of windows/rounds), we truncate each client's
    # packet dataframe to the minimum packet length across clients.
    #
    # This keeps timestamps ordered and only discards extra tail samples
    # from clients that had more data.
    # ------------------------------------------------------------------
    packet_lengths = [
        len(c["packet"]) for c in client_data if "packet" in c and len(c["packet"]) > 0
    ]
    if packet_lengths:
        min_packets = min(packet_lengths)
        if min_packets <= 0:
            print("Warning: cannot equalize packet samples (non‑positive minimum length).")
        else:
            print(f"\nEqualizing packet samples across clients to {min_packets} rows each...")
            for c in client_data:
                if "packet" in c:
                    original_len = len(c["packet"])
                    if original_len > min_packets:
                        c["packet"] = c["packet"].iloc[:min_packets].reset_index(drop=True)
                        c["num_samples"] = min_packets
                        print(
                            f"  Client {c['client_id']}: trimmed packets from "
                            f"{original_len} to {min_packets}"
                        )

    return client_data


def save_client_files(
    client_data: List[Dict],
    output_dir: str,
    gps_prefix: str = 'gps',
    imu_prefix: str = 'imu',
    packet_prefix: str = 'packet'
):
    """
    Save GPS, IMU, and packet CSV files for each client.
    
    Parameters:
    -----------
    client_data : list of dict
        List of client data dictionaries
    output_dir : str
        Output directory
    gps_prefix : str
        Prefix for GPS files
    imu_prefix : str
        Prefix for IMU files
    packet_prefix : str
        Prefix for packet files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving client files to {output_dir}...")
    
    client_mapping = {}
    
    for client_dict in client_data:
        client_id = client_dict['client_id']
        mapping_info = {
            'start_timestamp': client_dict['start_timestamp'],
            'end_timestamp': client_dict['end_timestamp']
        }
        
        # Save GPS file if present
        if 'gps' in client_dict:
            gps_df = client_dict['gps']
            gps_filename = f"{gps_prefix}_client_{client_id:03d}.csv"
            gps_path = os.path.join(output_dir, gps_filename)
            gps_df.to_csv(gps_path, index=False)
            mapping_info['gps_file'] = gps_path
            mapping_info['num_gps_samples'] = len(gps_df)
            print(f"  Client {client_id}:")
            print(f"    GPS: {gps_path} ({len(gps_df)} samples)")
        
        # Save IMU file if present
        if 'imu' in client_dict:
            imu_df = client_dict['imu']
            imu_filename = f"{imu_prefix}_client_{client_id:03d}.csv"
            imu_path = os.path.join(output_dir, imu_filename)
            imu_df.to_csv(imu_path, index=False)
            mapping_info['imu_file'] = imu_path
            mapping_info['num_imu_samples'] = len(imu_df)
            if 'gps' not in client_dict:
                print(f"  Client {client_id}:")
            print(f"    IMU: {imu_path} ({len(imu_df)} samples)")
        
        # Save packet file if present
        if 'packet' in client_dict:
            packet_df = client_dict['packet']
            packet_filename = f"{packet_prefix}_client_{client_id:03d}.csv"
            packet_path = os.path.join(output_dir, packet_filename)
            packet_df.to_csv(packet_path, index=False)
            mapping_info['packet_file'] = packet_path
            mapping_info['num_packet_samples'] = len(packet_df)
            if 'gps' not in client_dict and 'imu' not in client_dict:
                print(f"  Client {client_id}:")
            print(f"    Packet: {packet_path} ({len(packet_df)} samples)")
        
        client_mapping[client_id] = mapping_info
    
    # Save mapping file
    mapping_file = os.path.join(output_dir, 'client_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("Client File Mapping\n")
        f.write("=" * 80 + "\n\n")
        for client_id, info in client_mapping.items():
            f.write(f"Client {client_id}:\n")
            if 'gps_file' in info:
                f.write(f"  GPS: {info['gps_file']}\n")
                f.write(f"  GPS Samples: {info['num_gps_samples']}\n")
            if 'imu_file' in info:
                f.write(f"  IMU: {info['imu_file']}\n")
                f.write(f"  IMU Samples: {info['num_imu_samples']}\n")
            if 'packet_file' in info:
                f.write(f"  Packet: {info['packet_file']}\n")
                f.write(f"  Packet Samples: {info['num_packet_samples']}\n")
            f.write(f"  Time Range: {info['start_timestamp']:.0f} to {info['end_timestamp']:.0f}\n")
            f.write("\n")
    
    print(f"\nSaved client mapping to {mapping_file}")
    
    # Also save as JSON for programmatic access
    import json
    json_mapping_file = os.path.join(output_dir, 'client_mapping.json')
    # Convert numpy types to native Python types for JSON
    json_mapping = {}
    for client_id, info in client_mapping.items():
        json_entry = {
            'start_timestamp': float(info['start_timestamp']),
            'end_timestamp': float(info['end_timestamp'])
        }
        if 'gps_file' in info:
            json_entry['gps_file'] = info['gps_file']
            json_entry['num_gps_samples'] = int(info['num_gps_samples'])
        if 'imu_file' in info:
            json_entry['imu_file'] = info['imu_file']
            json_entry['num_imu_samples'] = int(info['num_imu_samples'])
        if 'packet_file' in info:
            json_entry['packet_file'] = info['packet_file']
            json_entry['num_packet_samples'] = int(info['num_packet_samples'])
        json_mapping[str(client_id)] = json_entry
    with open(json_mapping_file, 'w') as f:
        json.dump(json_mapping, f, indent=2)
    
    print(f"Saved JSON mapping to {json_mapping_file}")
    
    return client_mapping


def create_config_template(client_mapping: Dict, output_dir: str):
    """
    Create a template config file showing how to use the prepared data.
    
    Parameters:
    -----------
    client_mapping : dict
        Client mapping dictionary
    output_dir : str
        Output directory
    """
    config_template = os.path.join(output_dir, 'config_template.py')
    
    with open(config_template, 'w') as f:
        f.write("# Configuration template for using prepared client data\n")
        f.write("# Copy relevant parts to your config.py or modify run_fl_system.py\n\n")
        f.write("# Example: Modify run_fl_system.py to use client-specific files\n")
        f.write("# In run_fl_system.py, change the client command to:\n\n")
        f.write("for i in range(1, NUM_CLIENTS + 1):\n")
        f.write("    client_id = f'client_{i}'\n")
        if any('gps_file' in info for info in client_mapping.values()):
            f.write("    gps_file = f'{output_dir}/gps_client_{i-1:03d}.csv'\n")
        if any('imu_file' in info for info in client_mapping.values()):
            f.write("    imu_file = f'{output_dir}/imu_client_{i-1:03d}.csv'\n")
        if any('packet_file' in info for info in client_mapping.values()):
            f.write("    packet_file = f'{output_dir}/packet_client_{i-1:03d}.csv'\n")
        f.write("    # ... use gps_file, imu_file, and/or packet_file in client command\n\n")
        f.write("# Client file mapping:\n")
        for client_id, info in client_mapping.items():
            f.write(f"# Client {client_id}:\n")
            if 'gps_file' in info:
                f.write(f"#   GPS: {info['gps_file']}\n")
            if 'imu_file' in info:
                f.write(f"#   IMU: {info['imu_file']}\n")
            if 'packet_file' in info:
                f.write(f"#   Packet: {info['packet_file']}\n")
            f.write("\n")
    
    print(f"Created config template: {config_template}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare GPS, IMU, and labeled packet data for Federated Learning System (run_fl_system.py)'
    )
    parser.add_argument(
        '--gps-file', '-g',
        type=str,
        default=None,
        help='Path to GPS CSV file (optional)'
    )
    parser.add_argument(
        '--imu-file', '-i',
        type=str,
        default=None,
        help='Path to IMU CSV file (optional)'
    )
    parser.add_argument(
        '--packet-file', '-p',
        type=str,
        default=None,
        help='Path to labeled packet CSV file (optional)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for prepared client data'
    )
    parser.add_argument(
        '--n-clients', '-n',
        type=int,
        default=3,
        help='Number of FL clients (default: 3)'
    )
    parser.add_argument(
        '--timestamp-col',
        type=str,
        default='TimeUS',
        help='Timestamp column name (default: TimeUS)'
    )
    parser.add_argument(
        '--sampling-strategy',
        type=str,
        choices=['downsample', 'upsample'],
        default='downsample',
        help='Sampling strategy: downsample (IMU->GPS) or upsample (GPS->IMU) (default: downsample)'
    )
    parser.add_argument(
        '--gps-prefix',
        type=str,
        default='gps',
        help='Prefix for GPS output files (default: gps)'
    )
    parser.add_argument(
        '--imu-prefix',
        type=str,
        default='imu',
        help='Prefix for IMU output files (default: imu)'
    )
    parser.add_argument(
        '--packet-prefix',
        type=str,
        default='packet',
        help='Prefix for packet output files (default: packet)'
    )
    parser.add_argument(
        '--packet-timestamp-col',
        type=str,
        default='Timestamp',
        help='Timestamp column name for packet data (default: Timestamp)'
    )
    
    args = parser.parse_args()
    
    # Validate that at least one data source is provided
    if not args.gps_file and not args.imu_file and not args.packet_file:
        parser.error("At least one of --gps-file, --imu-file, or --packet-file must be provided")
    
    print("=" * 80)
    print("PREPARING DATA FOR FEDERATED LEARNING")
    print("=" * 80)
    if args.gps_file:
        print(f"GPS File: {args.gps_file}")
    if args.imu_file:
        print(f"IMU File: {args.imu_file}")
    if args.packet_file:
        print(f"Packet File: {args.packet_file}")
    print(f"Output Directory: {args.output}")
    print(f"Number of Clients: {args.n_clients}")
    if args.gps_file and args.imu_file:
        print(f"Sampling Strategy: {args.sampling_strategy}")
    print("=" * 80 + "\n")
    
    # Load data
    gps_df = None
    imu_df = None
    packet_df = None
    merged_df = None
    
    if args.gps_file:
        gps_df = load_gps_data(args.gps_file, args.timestamp_col)
    
    if args.imu_file:
        imu_df = load_imu_data(args.imu_file, args.timestamp_col)
    
    if args.packet_file:
        packet_df = load_packet_data(args.packet_file, args.packet_timestamp_col)
    
    # Merge GPS and IMU data if both are provided
    if gps_df is not None and imu_df is not None:
        merged_df = merge_gps_imu(
            gps_df, imu_df, 
            timestamp_col=args.timestamp_col,
            sampling_strategy=args.sampling_strategy
        )
    
    # Split among clients
    client_data = split_among_clients(
        gps_df=gps_df,
        imu_df=imu_df,
        packet_df=packet_df,
        merged_df=merged_df,
        n_clients=args.n_clients,
        timestamp_col=args.timestamp_col,
        packet_timestamp_col=args.packet_timestamp_col,
        sampling_strategy=args.sampling_strategy
    )
    
    # Save client files
    client_mapping = save_client_files(
        client_data, args.output,
        gps_prefix=args.gps_prefix,
        imu_prefix=args.imu_prefix,
        packet_prefix=args.packet_prefix
    )
    
    # Create config template
    create_config_template(client_mapping, args.output)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output}")
    if args.gps_file:
        print(f"  - {args.n_clients} GPS files ({args.gps_prefix}_client_XXX.csv)")
    if args.imu_file:
        print(f"  - {args.n_clients} IMU files ({args.imu_prefix}_client_XXX.csv)")
    if args.packet_file:
        print(f"  - {args.n_clients} Packet files ({args.packet_prefix}_client_XXX.csv)")
    print(f"  - client_mapping.txt (human-readable mapping)")
    print(f"  - client_mapping.json (programmatic mapping)")
    print(f"  - config_template.py (usage instructions)")
    print("\nTo use with run_fl_system.py:")
    print("  1. Modify run_fl_system.py to use client-specific files")
    if args.gps_file or args.imu_file:
        print("  2. Or update config.py GPS_FILE and IMU_FILE for each client")
    print("  3. See config_template.py for examples")
    print("=" * 80)


if __name__ == "__main__":
    main()
