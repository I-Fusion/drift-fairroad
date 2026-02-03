# Configuration template for using prepared client data
# Copy relevant parts to your config.py or modify run_fl_system.py

# Example: Modify run_fl_system.py to use client-specific files
# In run_fl_system.py, change the client command to:

for i in range(1, NUM_CLIENTS + 1):
    client_id = f'client_{i}'
    gps_file = f'{output_dir}/gps_client_{i-1:03d}.csv'
    imu_file = f'{output_dir}/imu_client_{i-1:03d}.csv'
    # ... use gps_file, imu_file, and/or packet_file in client command

# Client file mapping:
# Client 0:
#   GPS: .\data\validate\gps-imu\gps_client_000.csv
#   IMU: .\data\validate\gps-imu\imu_client_000.csv

# Client 1:
#   GPS: .\data\validate\gps-imu\gps_client_001.csv
#   IMU: .\data\validate\gps-imu\imu_client_001.csv

# Client 2:
#   GPS: .\data\validate\gps-imu\gps_client_002.csv
#   IMU: .\data\validate\gps-imu\imu_client_002.csv

