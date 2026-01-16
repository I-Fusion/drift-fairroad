# -*- coding: utf-8 -*-
"""
Script to load packet data from .npy file, filter rows with port 5762, and label rows
based on time periods from CSV file.

The .npy file should contain columns: Timestamp, SrcPort, DstPort, Length, MsgID, Protocol
"""

import numpy as np
import pandas as pd
from datetime import datetime


def load_periods_from_csv(periods_csv_path):
    """
    Load time periods from a CSV file.
    
    Parameters:
    -----------
    periods_csv_path : str
        Path to the CSV file containing time periods
    
    Returns:
    --------
    periods : list of dict
        List of dictionaries containing 'start_time' and 'end_time' for each period
    """
    print(f"Loading time periods from {periods_csv_path}...")
    periods_df = pd.read_csv(periods_csv_path)
    
    periods = []
    for _, row in periods_df.iterrows():
        periods.append({
            'period': int(row['period']),
            'start_time': pd.to_datetime(row['start_time']),
            'end_time': pd.to_datetime(row['end_time']),
            'duration_seconds': row['duration_seconds'],
        })
    
    print(f"Loaded {len(periods)} time periods")
    return periods


def load_and_label_npy(npy_file_path, periods_csv_path, output_path=None):
    """
    Load packet data from .npy file, filter rows with port 5762, and label based on time periods.
    
    Parameters:
    -----------
    npy_file_path : str
        Path to the .npy file containing packet data
    periods_csv_path : str
        Path to the CSV file containing time periods
    output_path : str, optional
        Path to save the labeled data. If None, saves to same directory as input with '_labeled' suffix
    
    Returns:
    --------
    labeled_df : pandas.DataFrame
        DataFrame with labeled data (Label column added)
    """
    print(f"\nLoading packet data from {npy_file_path}...")
    
    # Load the numpy array
    data = np.load(npy_file_path, allow_pickle=True)
    
    print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    print(f"Data type: {type(data)}")
    if hasattr(data, 'dtype'):
        print(f"Data dtype: {data.dtype}")
    
    # Handle different data formats
    if isinstance(data, np.ndarray):
        if data.dtype.names is not None:
            # Structured array - convert to DataFrame
            df = pd.DataFrame(data)
            # Ensure column names match expected format
            if 'Timestamp' not in df.columns and len(df.columns) >= 6:
                df.columns = ['Timestamp', 'SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol'] + list(df.columns[6:])
        elif len(data.shape) == 2 and data.shape[1] >= 6:
            # 2D array - assume columns are in order
            df = pd.DataFrame(data, columns=['Timestamp', 'SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol'])
        elif len(data.shape) == 1:
            # 1D array of objects (list of tuples/arrays)
            try:
                # Convert to list and then to DataFrame
                data_list = data.tolist()
                df = pd.DataFrame(data_list, columns=['Timestamp', 'SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol'])
            except Exception as e:
                print(f"Warning: Could not convert 1D array directly. Trying alternative method...")
                # Try accessing first element to understand structure
                if len(data) > 0:
                    first_elem = data[0]
                    if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) >= 6:
                        df = pd.DataFrame([list(row) for row in data], columns=['Timestamp', 'SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol'])
                    else:
                        raise ValueError(f"Cannot parse data structure. First element type: {type(first_elem)}")
                else:
                    raise ValueError("Empty data array")
        else:
            raise ValueError(f"Unsupported array shape: {data.shape}")
    elif isinstance(data, (list, tuple)):
        # Direct list/tuple
        df = pd.DataFrame(data, columns=['Timestamp', 'SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol'])
    else:
        # Try to convert to list
        try:
            data_list = list(data)
            df = pd.DataFrame(data_list, columns=['Timestamp', 'SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol'])
        except:
            raise ValueError(f"Cannot parse data structure. Type: {type(data)}")
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows for debugging
    if len(df) > 0:
        print(f"\nFirst row sample:")
        print(df.iloc[0])
    
    # Convert timestamps to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        print("\nConverting timestamps to datetime...")
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        except Exception as e:
            print(f"Warning: Could not convert all timestamps: {e}")
            print("Attempting to handle mixed formats...")
            # Try converting row by row
            timestamps_converted = []
            for ts in df['Timestamp']:
                try:
                    timestamps_converted.append(pd.to_datetime(ts))
                except:
                    timestamps_converted.append(pd.NaT)
            df['Timestamp'] = timestamps_converted
    
    # Filter: Remove rows where port 5762 is in SrcPort or DstPort
    print(f"\nFiltering rows with port 5762 in SrcPort or DstPort...")
    original_count = len(df)
    
    # Convert ports to numeric if needed
    df['SrcPort'] = pd.to_numeric(df['SrcPort'], errors='coerce')
    df['DstPort'] = pd.to_numeric(df['DstPort'], errors='coerce')
    
    # Filter out rows where either SrcPort or DstPort is 5762
    filter_mask = (df['SrcPort'] != 5762) & (df['DstPort'] != 5762)
    df = df[filter_mask].copy().reset_index(drop=True)
    
    removed_count = original_count - len(df)
    print(f"Removed {removed_count} rows with port 5762")
    print(f"Remaining rows: {len(df)}")
    
    # Load time periods
    periods = load_periods_from_csv(periods_csv_path)
    
    # Label rows based on time periods
    print(f"\nLabeling rows based on time periods...")
    labels = []
    
    for idx, row in df.iterrows():
        timestamp = row['Timestamp']
        
        if pd.isna(timestamp):
            # Invalid timestamp - label as 0
            labels.append(0)
        else:
            # Check if timestamp falls within any period
            in_period = False
            for period in periods:
                if period['start_time'] <= timestamp <= period['end_time']:
                    in_period = True
                    break
            labels.append(1 if in_period else 0)
    
    # Add label column
    df['Label'] = labels
    
    # Print statistics
    label_counts = pd.Series(labels).value_counts()
    print(f"\nLabel statistics:")
    print(f"  Rows labeled as 1 (in period): {label_counts.get(1, 0)}")
    print(f"  Rows labeled as 0 (not in period): {label_counts.get(0, 0)}")
    print(f"  Total rows: {len(labels)}")
    
    # Save labeled data
    if output_path is None:
        output_path = npy_file_path.replace('.npy', '_labeled.csv')
    
    print(f"\nSaving labeled data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    
    return df


if __name__ == "__main__":
    # File paths
    npy_file_path = r'D:\simulations\mission_2_wp_23_attack_add_wp_4_random_wind_10.npy'
    periods_csv_path = r'D:\simulations\mission_2_wp_23_attack_add_wp_4_random_wind_10_tcp_port_5762_time_periods.csv'
  
    # Optional: specify output path
    output_path = None  # Will auto-generate if None
    
    try:
        labeled_df = load_and_label_npy(npy_file_path, periods_csv_path, output_path)
        
        print("\n" + "="*80)
        print("Labeling complete!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        print("Please update the file paths in the script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
