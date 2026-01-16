# -*- coding: utf-8 -*-
"""
Script to label CSV rows based on time periods found from .npy file.
The CSV file should have timestamps in the second column (index 1).
Rows are labeled as 1 if their timestamp falls within any time period, 0 otherwise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def parse_csv_timestamp(timestamp_str):
    """
    Parse CSV timestamp. Handles multiple formats:
    - Full datetime: '1/12/2026 2:25:15 PM' or '1/12/2026 14:25:15'
    - Time only: 'MM:SS.m' (minutes:seconds.milliseconds)
    - Seconds: numeric value
    
    Parameters:
    -----------
    timestamp_str : str
        Timestamp string in various formats
    
    Returns:
    --------
    datetime_obj : datetime or None
        Parsed datetime object, or None if parsing fails
    """
    try:
        timestamp_str = str(timestamp_str).strip()
        
        # Try parsing as full datetime first (handles formats like "1/12/2026 2:25:15 PM")
        try:
            # Try common datetime formats
            dt = pd.to_datetime(timestamp_str)
            return dt
        except:
            pass
        
        # Try parsing as time-only format 'MM:SS.m'
        parts = timestamp_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds_part = parts[1]
            seconds = float(seconds_part)
            # Return as timedelta from midnight (will need to be combined with date)
            return timedelta(hours=0, minutes=minutes, seconds=seconds)
        
        # Try parsing as numeric seconds
        try:
            seconds = float(timestamp_str)
            return timedelta(seconds=seconds)
        except:
            pass
        
        return None
    except:
        return None

def find_time_periods_from_npy(npy_file_path, gap_threshold_seconds=1.0):
    """
    Find distinct time periods in a .npy file based on timestamp gaps.
    
    Parameters:
    -----------
    npy_file_path : str
        Path to the .npy file containing packet data
    gap_threshold_seconds : float
        Minimum gap in seconds to consider as a new time period (default: 1.0)
    
    Returns:
    --------
    periods : list of dict
        List of dictionaries containing 'start_time' and 'end_time' for each period
    """
    # Load the numpy array
    print(f"Loading data from {npy_file_path}...")
    data = np.load(npy_file_path, allow_pickle=True)
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Extract timestamps from the first column
    timestamps = data[:, 0]
    
    # Convert to pandas Series for easier handling
    # Handle different timestamp formats
    if isinstance(timestamps[0], (datetime, pd.Timestamp)):
        # Already datetime objects
        timestamps_dt = pd.to_datetime(timestamps)
    else:
        # Try to convert if they're strings or other formats
        timestamps_dt = pd.to_datetime(timestamps)
    
    # Sort timestamps to ensure chronological order
    sorted_indices = np.argsort(timestamps_dt)
    timestamps_sorted = timestamps_dt[sorted_indices]
    
    print(f"Total timestamps: {len(timestamps_sorted)}")
    print(f"Time range: {timestamps_sorted[0]} to {timestamps_sorted[-1]}")
    
    # Find gaps between consecutive timestamps
    periods = []
    period_start = timestamps_sorted[0]
    period_end = timestamps_sorted[0]
    
    for i in range(1, len(timestamps_sorted)):
        time_diff = (timestamps_sorted[i] - timestamps_sorted[i-1]).total_seconds()
        
        if time_diff > gap_threshold_seconds:
            # Gap detected - end current period and start new one
            periods.append({
                'period': len(periods) + 1,
                'start_time': period_start,
                'end_time': period_end,
                'duration_seconds': (period_end - period_start).total_seconds(),
            })
            period_start = timestamps_sorted[i]
            period_end = timestamps_sorted[i]
        else:
            # Continue current period
            period_end = timestamps_sorted[i]
    
    # Add the last period
    periods.append({
        'period': len(periods) + 1,
        'start_time': period_start,
        'end_time': period_end,
        'duration_seconds': (period_end - period_start).total_seconds(),
    })
    
    return periods

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
    
    return periods

def label_csv_with_periods(csv_file_path, periods, reference_time=None, file_type=None):
    """
    Label CSV rows based on whether their timestamps fall within the time periods.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file to label
    periods : list of dict
        List of time periods with 'start_time' and 'end_time'
    reference_time : datetime, optional
        Reference time to align CSV relative timestamps with absolute timestamps.
        If None, uses the earliest start_time from periods.
    file_type : str, optional
        Type of CSV file: 'gps' or 'imu'. If None, will auto-detect from filename.
    
    Returns:
    --------
    labeled_df : pandas.DataFrame
        DataFrame with added 'Label' column (1 if in period, 0 otherwise)
    """
    print(f"\nLoading CSV file: {csv_file_path}...")
    
    # Auto-detect file type from filename if not provided
    if file_type is None:
        filename_lower = csv_file_path.lower()
        if 'gps' in filename_lower:
            file_type = 'gps'
        elif 'imu' in filename_lower:
            file_type = 'imu'
        else:
            file_type = 'gps'  # Default to GPS if cannot detect
            print(f"Warning: Could not detect file type from filename. Defaulting to 'gps'")
    
    print(f"Detected file type: {file_type.upper()}")
    
    # Read CSV file (assuming no header)
    try:
        df = pd.read_csv(csv_file_path, header=None)
    except:
        # Try with header
        df = pd.read_csv(csv_file_path)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Original CSV columns: {df.shape[1]}")
    
    # Apply file-type specific filtering
    if file_type.lower() == 'gps':
        # For GPS: Remove rows where 5th column (index 4) == 1
        # Keep only columns 2, 11, 12, 13 (indices 1, 10, 11, 12)
        original_rows = len(df)
        
        # Remove rows where column 4 (5th column) == 1
        if df.shape[1] > 4:
            filter_mask = df.iloc[:, 4] != 1
            df = df[filter_mask].reset_index(drop=True)
            removed_rows = original_rows - len(df)
            if removed_rows > 0:
                print(f"Removed {removed_rows} rows where 5th column == 1")
        
        # Keep only columns 2, 11, 12, 13 (indices 1, 10, 11, 12)
        if df.shape[1] >= 13:
            df = df.iloc[:, [1, 10, 11, 12]].copy()
            print(f"Kept only columns 2, 11, 12, 13 (indices 1, 10, 11, 12)")
        else:
            print(f"Warning: CSV has only {df.shape[1]} columns, cannot apply GPS filtering")
        
        # After filtering, timestamp is in first column (index 0)
        timestamp_col_idx = 0
        
    elif file_type.lower() == 'imu':
        # For IMU: Remove columns 1, 3, 4, 12, 13, 15, 16 (indices 0, 2, 3, 11, 12, 14, 15)
        # Original timestamp is assumed to be in column 1 (index 1)
        original_timestamp_idx = 1
        
        # Get all column indices
        all_cols = list(range(df.shape[1]))
        # Columns to remove: 0, 2, 3, 11, 12, 14, 15
        cols_to_remove = [0, 2, 3, 11, 12, 14, 15]
        # Keep only columns that are not in cols_to_remove
        cols_to_keep = [i for i in all_cols if i not in cols_to_remove and i < 18]
        
        if len(cols_to_keep) > 0:
            df = df.iloc[:, cols_to_keep].copy()
            print(f"Removed columns 1, 3, 4, 12, 13, 15, 16 (indices 0, 2, 3, 11, 12, 14, 15)")
            print(f"Kept {len(cols_to_keep)} columns")
            
            # Find where the original timestamp column (index 1) is now
            if original_timestamp_idx in cols_to_keep:
                timestamp_col_idx = cols_to_keep.index(original_timestamp_idx)
                print(f"Timestamp column (original index {original_timestamp_idx}) is now at index {timestamp_col_idx}")
            else:
                # Timestamp column was removed, use first column as fallback
                timestamp_col_idx = 0
                print(f"Warning: Original timestamp column was removed. Using first column as timestamp.")
        else:
            print(f"Warning: All columns would be removed. Keeping original columns.")
            timestamp_col_idx = 1  # Use original timestamp position
        
    else:
        # Unknown file type, use original behavior (timestamp in column 1)
        print(f"Warning: Unknown file type '{file_type}'. Using default timestamp column (index 1)")
        timestamp_col_idx = 1
    
    print(f"Filtered CSV shape: {df.shape}")
    print(f"Filtered CSV columns: {df.shape[1]}")
    print(f"Timestamp column index: {timestamp_col_idx}")
    
    # Extract timestamps from the appropriate column
    timestamp_col = df.iloc[:, timestamp_col_idx]
    
    # Parse CSV timestamps
    print("Parsing CSV timestamps...")
    parsed_timestamps = timestamp_col.apply(parse_csv_timestamp)
    
    # Convert parsed timestamps to datetime objects
    csv_times_absolute = pd.Series([None] * len(df), dtype='datetime64[ns]')
    
    # Find reference time if not provided
    if reference_time is None:
        # Use the earliest start time from periods
        reference_time = min(period['start_time'] for period in periods)
        print(f"Using reference time: {reference_time}")
    
    # Process each parsed timestamp
    for idx, parsed_ts in enumerate(parsed_timestamps):
        if parsed_ts is None:
            continue
        
        if isinstance(parsed_ts, pd.Timestamp) or isinstance(parsed_ts, datetime):
            # Already a full datetime
            csv_times_absolute[idx] = pd.to_datetime(parsed_ts)
        elif isinstance(parsed_ts, timedelta):
            # Time delta - need to combine with reference date
            # Extract date from reference_time and add the time delta
            ref_date = pd.to_datetime(reference_time).date()
            csv_times_absolute[idx] = pd.to_datetime(ref_date) + parsed_ts
        else:
            # Try to convert directly
            try:
                csv_times_absolute[idx] = pd.to_datetime(parsed_ts)
            except:
                pass
    
    # Remove rows with invalid timestamps
    valid_mask = csv_times_absolute.notna()
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} rows have invalid timestamps and will be labeled as 0")
    
    if valid_mask.any():
        csv_min = csv_times_absolute[valid_mask].min()
        csv_max = csv_times_absolute[valid_mask].max()
        period_min = min(p['start_time'] for p in periods)
        period_max = max(p['end_time'] for p in periods)
        
        print(f"CSV time range: {csv_min} to {csv_max}")
        print(f"Period time range: {period_min} to {period_max}")
        
        # Check if dates align - if not, align periods to CSV date
        csv_date = csv_min.date()
        period_date = period_min.date()
        
        if csv_date != period_date:
            print(f"Warning: Date mismatch detected. CSV date: {csv_date}, Period date: {period_date}")
            print("Aligning period dates to match CSV dates...")
            
            # Align periods to use the same date as CSV timestamps
            for period in periods:
                # Replace date but keep time
                period_start_time = period['start_time'].time()
                period_end_time = period['end_time'].time()
                
                # Combine CSV date with period times
                period['start_time'] = pd.to_datetime(f"{csv_date} {period_start_time}")
                period['end_time'] = pd.to_datetime(f"{csv_date} {period_end_time}")
            
            print(f"Aligned period time range: {min(p['start_time'] for p in periods)} to {max(p['end_time'] for p in periods)}")
    
    # Create label column: 1 if timestamp is in any period, 0 otherwise
    print("Labeling rows based on time periods...")
    labels = []
    
    for idx, csv_time in enumerate(csv_times_absolute):
        if pd.isna(csv_time):
            # Invalid timestamp - label as 0
            labels.append(0)
        else:
            in_period = False
            for period in periods:
                if period['start_time'] <= csv_time <= period['end_time']:
                    in_period = True
                    break
            labels.append(1 if in_period else 0)
    
    # Add label column to dataframe
    df['Label'] = labels
    
    # Ensure timestamp is in first column, then features, then label
    # Get current column indices
    n_cols_before_label = df.shape[1] - 1  # Number of columns before adding label
    label_col_idx = df.shape[1] - 1  # Label column index
    
    # If timestamp is not already in first position, reorder
    if timestamp_col_idx != 0:
        # Get feature columns (all except timestamp and label)
        feature_cols = [i for i in range(n_cols_before_label) if i != timestamp_col_idx]
        # Create new column order: timestamp, features, label
        new_column_order = [timestamp_col_idx] + feature_cols + [label_col_idx]
        df = df.iloc[:, new_column_order].copy()
        print(f"Reordered columns: timestamp moved to first position")
    
    # Reset column names to be sequential (0, 1, 2, ...)
    df.columns = range(df.shape[1])
    
    print(f"\nFinal CSV shape: {df.shape}")
    print(f"Column order: timestamp (col 0), features (cols 1-{df.shape[1]-2}), label (col {df.shape[1]-1})")
    
    # Print statistics
    label_counts = pd.Series(labels).value_counts()
    print(f"\nLabel statistics:")
    print(f"  Rows labeled as 1 (in period): {label_counts.get(1, 0)}")
    print(f"  Rows labeled as 0 (not in period): {label_counts.get(0, 0)}")
    print(f"  Total rows: {len(labels)}")
    
    return df

if __name__ == "__main__":
    # Path to the .npy file (to find time periods)
    npy_file_path = r'D:\simulations\mission_2_wp_23_attack_add_wp_4_random_wind_10_tcp_port_5762.npy'
    # Path to the CSV file to label
    csv_file_path = r'C:\Users\86152\ardupilot\Tools\autotest\logs\mission_2_wp_23_attack_add_wp_4_random_wind_10_imu.csv'
    # Optional: Path to saved time periods CSV (if you've already run find_time_periods.py)
    # If provided, will use this instead of loading from .npy file
    periods_csv_path = r'D:\simulations\mission_2_wp_23_attack_add_wp_4_random_wind_10_tcp_port_5762_time_periods.csv'
    # Gap threshold in seconds (only used if loading from .npy file)
    gap_threshold = 1.0  # 1 second gap
    
    try:
        # Load time periods
        if periods_csv_path:
            periods = load_periods_from_csv(periods_csv_path)
        else:
            periods = find_time_periods_from_npy(npy_file_path, gap_threshold_seconds=gap_threshold)
        
        print(f"\nFound {len(periods)} time periods:")
        for period in periods:
            start_str = period['start_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            end_str = period['end_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"  Period {period['period']}: {start_str} to {end_str} ({period['duration_seconds']:.3f}s)")
        
        # Label the CSV file
        labeled_df = label_csv_with_periods(csv_file_path, periods)
        
        # Save labeled CSV
        output_labeled_csv = csv_file_path.replace('.csv', '_labeled.csv')
        labeled_df.to_csv(output_labeled_csv, index=False, header=False)
        print(f"\nLabeled CSV saved to: {output_labeled_csv}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        print("Please update the file paths in the script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

