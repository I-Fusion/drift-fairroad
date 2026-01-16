from pymavlink import mavutil, mavwp
import random

master = mavutil.mavlink_connection('tcp:localhost:5762')
master.wait_heartbeat(blocking=True)                                       
print('[+] Connected to drone!')

# current mission plan
## 1. Request the mission count
master.mav.mission_request_list_send(
    master.target_system, master.target_component
)

## 2. Wait for the MISSION_COUNT message
msg = master.recv_match(type=['MISSION_COUNT'], blocking=True)
mission_count = msg.count
print(f"Mission count: {mission_count}")

lat = []
lon = []    
alt = []
## 3. Request each mission item
for i in range(mission_count):
    master.mav.mission_request_int_send(
        master.target_system, master.target_component, i
    )
    
    # 4. Wait for the MISSION_ITEM_INT message
    item = master.recv_match(type=['MISSION_ITEM_INT'], blocking=True)
    print(f"Received item {i}: {item.seq}, Lat: {item.x}, Lon: {item.y}, alt: {item.z}")
    lat.append(1e-7*item.x)
    lon.append(1e-7*item.y)
    alt.append(item.z)

print("[+] Mission download complete.")
                                                  

seq = 0
frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
radius = 10

# the index of the current active waypoint
master.mav.command_long_send(
    master.target_system, 
    master.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 
    0,      # Confirmation
    42,     # Message ID for MISSION_CURRENTas well as GPS and IMU data
    500000, # Interval in microseconds
    0, 0, 0, 0, 0
)

msg = master.recv_match(type='MISSION_CURRENT', blocking=True)

if msg:
    # 'seq' is the index of the current waypoint
    idx = msg.seq +1 
    print(f"Current Waypoint Sequence: {msg.seq}")
else:
    idx = 1    

def get_random_union_value():
    """
    Generates a random float over the union of 
    [-0.0005, -0.0001] and [0.0001, 0.0005].
    """
    # Total width of one interval is 0.0004
    # We pick a random value in [0.0001, 0.0005]
    val = random.uniform(0.0001, 0.0005)
    
    # Randomly flip the sign to cover the negative interval
    if random.random() < 0.5:
        return -val
    return val

insert_lat = (lat[idx-1]+lat[idx])/2+get_random_union_value()#0.0001
insert_lon = (lon[idx-1]+lon[idx])/2+get_random_union_value()#0.0001
insert_alt = (alt[idx-1]+alt[idx])/2+10

lat.insert(idx,insert_lat)
lon.insert(idx,insert_lon)
alt.insert(idx,insert_alt)
print(f"Inserting waypoint at index {idx}")

N = len(lat)
wp = mavwp.MAVWPLoader()  
for i in range(N):
    wp.add(
            mavutil.mavlink.MAVLink_mission_item_message(
                master.target_system,
                master.target_component,
                seq,
                frame,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 0, 0, radius, 0, 0,
                lat[i],lon[i],alt[i])
            )

    seq += 1                                                                       

print(wp.count())
master.waypoint_clear_all_send()                                     
master.waypoint_count_send(wp.count())                          

for i in range(wp.count()):
    msg = master.recv_match(type=['MISSION_REQUEST'],blocking=True)
    master.mav.send(wp.wp(msg.seq))
    print(f'Sending waypoint {msg.seq}') 

