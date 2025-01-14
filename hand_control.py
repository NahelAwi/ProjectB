import asyncio
import random
import math
from bleak import BleakClient, BleakScanner

# Replace with your device's MAC address
DEVICE_ADDRESS = "B8:D6:1A:43:60:52"

# UUIDs for the Direct Execute Service
SERVICE_UUID = "e0198000-7544-42c1-0000-b24344b6aa70"
CHARACTERISTIC_UUID = "e0198000-7544-42c1-0001-b24344b6aa70"


def create_command(torque, time_units, motors, directions):
    """
    Create the raw byte command to control the hand.
    :param torque: Torque stop threshold (1 byte).
    :param time_units: Time stop threshold (1 byte).
    :param motors: Motors activated (1 byte).
    :param directions: Motors direction (1 byte).
    :return: Byte array representing the command.
    """
    # Length of command = 1 (length byte) + 4 (1 movement)
    length = 1 + 4
    return bytes([length, torque, time_units, motors, directions])

def rotate_hand(time_units, direction, torque=0):
    directions_mask = 0
    motors_mask = 0
    directions_mask |= (direction << 7)
    motors_mask |= (1 << 7)

    # Create the command
    command = create_command(torque, time_units, motors_mask, directions_mask)
    
    return command

def grip(direction, time_in_ms, torque=0):
    time_units = int(time_in_ms / 50)
    directions_mask = 0
    motors_mask = 0
    directions_mask |= ((direction << 3) | (direction << 4) | (direction << 5) | (direction << 6))
    motors_mask |= ((1 << 3) | (1 << 4) | (1 << 5) | (1 << 6))

    # Create the command
    command = create_command(torque, time_units, motors_mask, directions_mask)
    
    return command

# # Scan for devices to confirm the address
# found = False
# print("Scanning for BLE devices...")
# devices = await BleakScanner.discover()
# for device in devices:
#     print(f"Found: {device.name} ({device.address})")
#     if device.address == DEVICE_ADDRESS:
#         found = True
        
# if found == False:
#     print("Error: not found!")
#     exit(-1)

async def rotate(queue):
    RPM = 10
    angular_v = RPM*360/60
    angular_v_in_ms = angular_v / 1000
    gripped = False

    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print(f"Connected to {DEVICE_ADDRESS}")
            while True:
                # angle = random.randint(-90, 90)
                angle = queue.get()
                if angle is None:
                    print("unexpected behaviour - angle shouldn't return None - as queue.get() is blocking")
                    exit(-99)
                if angle == 0:
                    if not gripped:
                        command = grip(1, 1000)
                        await client.write_gatt_char(CHARACTERISTIC_UUID, command)  # does this wait for the whole rotation to happen ? if not, do the wait below (in the sleep)
                        await asyncio.sleep(1)
                        gripped = True
                    continue
                direction = 1 if (angle >= 0) else 0
                angle = abs(angle)
                rotation_time_in_ms = (angle / angular_v_in_ms)
                time_units = int(rotation_time_in_ms / 50)  # Convert to time units (50ms per unit)
                
                if gripped:
                    command = grip(0, 1000)
                    await client.write_gatt_char(CHARACTERISTIC_UUID, command)  # does this wait for the whole rotation to happen ? if not, do the wait below (in the sleep)
                    await asyncio.sleep(1)
                    gripped = False

                command = rotate_hand(time_units, direction)
                await client.write_gatt_char(CHARACTERISTIC_UUID, command)  # does this wait for the whole rotation to happen ? if not, do the wait below (in the sleep)
                print(f"Sent command: {command}")
                await asyncio.sleep(rotation_time_in_ms/1000)    # sleep for the calculated time above ?
                # await asyncio.sleep(2)
        else:
            print(f"Failed to connect to {DEVICE_ADDRESS}")


def hand_control_thread(queue):
    asyncio.run(rotate(queue))
