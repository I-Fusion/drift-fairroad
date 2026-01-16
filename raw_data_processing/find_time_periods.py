# -*- coding: utf-8 -*-
"""
Script to find starting and ending time of distinct time periods in a .npy file.
The .npy file should contain packet data with timestamps in the first column.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def find_time_periods(npy_file_path, gap_threshold_seconds=1.0):
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
    
    # Convert to pandas Series for easier handling if needed
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
                'packet_count': None  # Will calculate later
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
        'packet_count': None
    })
    
    # Count packets in each period
    for period in periods:
        mask = (timestamps_dt >= period['start_time']) & (timestamps_dt <= period['end_time'])
        period['packet_count'] = np.sum(mask)
    
    return periods, timestamps_sorted

def print_time_periods(periods):
    """Print time periods in a readable format."""
    print("\n" + "="*80)
    print("DISTINCT TIME PERIODS")
    print("="*80)
    print(f"{'Period':<8} {'Start Time':<30} {'End Time':<30} {'Duration (s)':<15} {'Packets':<10}")
    print("-"*80)
    
    for period in periods:
        start_str = period['start_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        end_str = period['end_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        duration = period['duration_seconds']
        
        print(f"{period['period']:<8} {start_str:<30} {end_str:<30} {duration:<15.3f} {period['packet_count']:<10}")
    
    print("="*80)
    print(f"Total periods found: {len(periods)}")
    print("="*80)

if __name__ == "__main__":
    # Path to the .npy file
    npy_file_path = r'D:\simulations\mission_2_wp_23_attack_add_wp_4_random_wind_10_tcp_port_5762.npy'
    # Gap threshold in seconds (adjust as needed)
    # If there's a gap larger than this, it's considered a new time period
    gap_threshold = 1.0  # 1 second gap
    
    try:
        periods, timestamps = find_time_periods(npy_file_path, gap_threshold_seconds=gap_threshold)
        print_time_periods(periods)
        
        # Optionally save to CSV
        periods_df = pd.DataFrame(periods)
        output_csv = npy_file_path.replace('.npy', '_time_periods.csv')
        periods_df.to_csv(output_csv, index=False)
        print(f"\nTime periods saved to: {output_csv}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {npy_file_path}")
        print("Please update the file path in the script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

