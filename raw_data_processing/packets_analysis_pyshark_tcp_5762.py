# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 10:57:34 2026
Processing the .pcapng file after applying the filter "tcp.port == 5762" to find the attack periods.
@author: Yajie Bao
"""

import binascii
from multiprocessing.resource_sharer import stop
import pyshark
import struct
import pandas as pd
import datetime
import numpy as np

def parse_port(p):
    if p is None:
        return None
    p_str = str(p)
    # If value looks like 'IP:port' (e.g. '127.0.0.1:53280'), keep port part
    if ':' in p_str:
        p_str = p_str.split(':')[-1]
    try:
        return int(p_str)
    except ValueError:
        return None

def parse_int(v):
    if v is None:
        return None
    try:
        return int(str(v))
    except ValueError:
        return None

class mavlink2(object):
    def __init__(self, pkt_buf):
        msg_len = pkt_buf[1]
        unpacker_format = f'BBBBBBB3s{msg_len}s2s'

        unpacked = struct.unpack(unpacker_format, pkt_buf)
        self.magic = unpacked[0]
        self.len = unpacked[1]
        self.incompat_flags = unpacked[2]
        self.compat_flags = unpacked[3]
        self.seq = unpacked[4]
        self.sysid = unpacked[5]
        self.compid = unpacked[6]

        self.msgid = bytearray(unpacked[7])
        self.msgid.reverse()
        self.msgid = f'0x{self.msgid.hex()}'

        self.msg = unpacked[8]

        self.checksum = bytearray(unpacked[9])
        self.checksum.reverse()
        self.checksum = f'0x{self.checksum.hex()}'

        assert len(self.msg) == self.len

def extract_packet_info(packet):
    try:
        timestamp = packet.sniff_time
        #time_epoch = float(packet.frame_info.time_epoch)
        # 2. Access UDP header fields using dot notation
        # Extract raw values and normalize to integers (port numbers and length)
        if hasattr(packet, 'udp'):
            src_port_raw = getattr(packet.udp, 'srcport', None)
            dst_port_raw = getattr(packet.udp, 'dstport', None)
            length_raw = getattr(packet.udp, 'length', None)

            src_port = parse_port(src_port_raw)
            dst_port = parse_port(dst_port_raw)
            length = parse_int(length_raw)

            print(f"UDP {packet.ip.src}:{src_port} -> {packet.ip.dst}:{dst_port} | Length: {length}")
            
            # 3. Extract the payload (data)
            # Note: 'packet.data.data' is the common field for raw UDP data
            if hasattr(packet, 'data'):
                raw_payload_bytes = packet.udp.payload.binary_value
                try:
                    ml2 = mavlink2(raw_payload_bytes)
                    print(f"MAVLink Msg ID: {int(ml2.msgid,16)}, Length: {ml2.len}, SysID: {ml2.sysid}, CompID: {ml2.compid}")
                    return timestamp, src_port, dst_port, length, int(ml2.msgid,16), int(0)# Indicate UDP packet
                except Exception as e:
                    print(f"  [Error] Parsing MAVLink: {e}")
        elif hasattr(packet, 'tcp'):
            src_port_raw = getattr(packet.tcp, 'srcport', None)
            dst_port_raw = getattr(packet.tcp, 'dstport', None)
            length_raw = getattr(packet.tcp, 'length', None)

            src_port = parse_port(src_port_raw)
            dst_port = parse_port(dst_port_raw)
            length = parse_int(length_raw)

            print(f"TCP {packet.ip.src}:{src_port} -> {packet.ip.dst}:{dst_port} | Length: {length}")
            
            raw_hex = ""
            if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'payload'):
                raw_hex = packet.tcp.payload
            elif hasattr(packet, 'data'):
                raw_hex = packet.data.data
            
            if not raw_hex:
                return

            # Convert hex string (e.g. 'fd:09:...') to bytes
            raw_payload = bytes.fromhex(raw_hex.replace(':', ''))
            
            # 3. Find the MAVLink 2 Magic Byte (0xFD)
            # TCP is a stream; the message might not start at index 0
            start_idx = raw_payload.find(0xFD)
            if start_idx == -1:
                return
            
            # 4. Parse the message
            try:
                ml2 = mavlink2(raw_payload[start_idx:])
                print(f"  [MAVLink] MsgID: {ml2.msgid}, Len: {ml2.len}, SysID: {ml2.sysid}, CompID: {ml2.compid}")
                return timestamp, src_port, dst_port, length, int(ml2.msgid,16), int(1) # Indicate TCP packet
            except Exception as e:
                print(f"  [Error] Parsing MAVLink: {e}")

        else:
            return None
    except AttributeError:
        # Handle packets missing expected layers
        pass

timestamp_list = []
src_port_list = []  
dst_port_list = []  
length_list = []
msg_id_list = []
protocol_list = []
with pyshark.FileCapture(r'D:\simulations\mission_2_wp_23_attack_add_wp_4_random_wind_10_tcp_port_5762.pcapng') as filtered_cap:
    # Iterate through the packets
    for pkt in filtered_cap:
        # 1. Get raw hex string (e.g., '48656c6c6f')
        result = extract_packet_info(pkt)
        if result:
            timestamp, src_port, dst_port, length, msg_id, protocol = result
            timestamp_list.append(timestamp)
            src_port_list.append(src_port)
            dst_port_list.append(dst_port)
            length_list.append(length)
            msg_id_list.append(msg_id)
            protocol_list.append(protocol)
            
df = pd.DataFrame({
    'Timestamp': timestamp_list,
    'SrcPort': src_port_list,
    'DstPort': dst_port_list,
    'Length': length_list,
    'MsgID': msg_id_list,
    'Protocol': protocol_list
})

np.save(r'D:\simulations\mission_2_wp_23_attack_add_wp_4_random_wind_10_tcp_port_5762.npy', df.to_numpy())
