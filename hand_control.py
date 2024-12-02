import asyncio
from bleak import BleakClient

# Replace with your device's MAC address
DEVICE_ADDRESS = "B8:D6:1A:43:60:52"

# UUIDs for the Direct Execute Service
SERVICE_UUID = "e0198000-7544-42c1-0000-b24344b6aa70"
CHARACTERISTIC_UUID = "e0198000-7544-42c1-0001-b24344b6aa70"

def create_rotation_command(torque, time_units, motors, directions):
    """
    Create the raw byte command to rotate the hand.
    :param torque: Torque stop threshold (1 byte).
    :param time_units: Time stop threshold (1 byte).
    :param motors: Motors activated (1 byte).
    :param directions: Motors direction (1 byte).
    :return: Byte array representing the command.
    """
    # Length of command = 1 (length byte) + 4 (1 movement)
    length = 1 + 4
    return bytes([length, torque, time_units, motors, directions])

async def rotate_hand_to_angle(angle, torque=0x01, speed=180):
    """
    Rotate the hand to a specific angle.
    :param angle: Target angle (in degrees).
    :param torque: Torque stop threshold (default: low torque).
    :param speed: Speed in degrees per second (default: 180).
    """
    # Calculate time units based on angle and speed
    time_in_ms = (angle / speed) * 1000  # Time in milliseconds
    time_units = int(time_in_ms / 50)  # Convert to time units (50ms per unit)

    # Motor activation and direction (all motors, default direction)
    motors = 0x1F
    directions = 0x00

    # Create the command
    command = create_rotation_command(torque, time_units, motors, directions)

    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print(f"Connected to {DEVICE_ADDRESS}")
            await client.write_gatt_char(CHARACTERISTIC_UUID, command)
            print(f"Sent command: {command}")
        else:
            print(f"Failed to connect to {DEVICE_ADDRESS}")

# Example usage: Rotate hand to 53 degrees
angle_to_rotate = 53

asyncio.run(rotate_hand_to_angle(angle_to_rotate))
