import asyncio
from bleak import BleakScanner

async def scan_ble_devices():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Name: {device.name}, Address: {device.address}")

loop = asyncio.get_event_loop()
loop.run_until_complete(scan_ble_devices())
loop.close()

