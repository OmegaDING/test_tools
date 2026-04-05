import asyncio
import time
from bleak import BleakScanner

# 解析iBeacon的major和minor（从广播数据中提取）
def parse_ibeacon_data(adv_data):
    major = None
    minor = None
    # iBeacon的manufacturer data标识（Apple公司ID：0x004C）
    if 0x004C in adv_data.manufacturer_data:
        manu_data = adv_data.manufacturer_data[0x004C]
        # 验证是否为iBeacon数据（长度需满足且前缀匹配）
        if len(manu_data) >= 25 and manu_data[:2] == b'\x02\x15':
            # 解析major（大端序）
            major = (manu_data[20] << 8) | manu_data[21]
            # 解析minor（大端序）
            minor = (manu_data[22] << 8) | manu_data[23]
    return major, minor

async def scan_ble():
    print("正在扫描附近蓝牙device...")
    # 扫描蓝牙设备
    devices = await BleakScanner.discover(timeout=2, return_adv=True)
    
    # 获取当前毫秒级时间戳
    scan_timestamp = int(time.time() * 1000)

    # 遍历设备并输出完整信息
    for mac, data in devices.items():
        device, adv_data = data
        rssi = adv_data.rssi
        name = device.name or "未知设备"
        major, minor = parse_ibeacon_data(adv_data)
        
        # 补全所有字段输出到控制台
        print(f"MAC: {mac:20} "
              f"Major: {major if major else 'N/A':6} "
              f"Minor: {minor if minor else 'N/A':6} "
              f"信号: {rssi:4} dBm  "
              f"名称: {name:20}"
            #   f"扫描时间戳(ms): {scan_timestamp}"
            )

# 启动运行
asyncio.run(scan_ble())