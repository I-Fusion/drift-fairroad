# Configuration template for using prepared client data
# Copy relevant parts to your config.py or modify run_fl_system.py

# Example: Modify run_fl_system.py to use client-specific files
# In run_fl_system.py, change the client command to:

for i in range(1, NUM_CLIENTS + 1):
    client_id = f'client_{i}'
    packet_file = f'{output_dir}/packet_client_{i-1:03d}.csv'
    # ... use gps_file, imu_file, and/or packet_file in client command

# Client file mapping:
# Client 0:
#   Packet: .\data\train\packets\packet_client_000.csv

# Client 1:
#   Packet: .\data\train\packets\packet_client_001.csv

# Client 2:
#   Packet: .\data\train\packets\packet_client_002.csv

