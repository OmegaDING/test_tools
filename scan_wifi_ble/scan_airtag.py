import asyncio
import time
from bleak import BleakScanner

# 苹果设备类型映射（AirTag 专用）
AIRTAG_DEVICE_TYPES = {
    0x01: "AirTag",
    0x02: "AirPods 1/2",
    0x03: "AirPods Pro",
    0x04: "AirPods Max",
    0x05: "AirPods 3",
    0x06: "AirPods Pro 2",
    0x10: "第三方查找配件",
}

def parse_airtag(adv_data):
    """解析 AirTag 广播数据（苹果查找网络协议）"""
    airtag_info = None
    # print(f"解析广播数据: {adv_data.manufacturer_data}")  # 调试输出原始数据
    # import pdb; pdb.set_trace()  # 设置断点调试，检查 adv_data 的结构和内容
    # 苹果公司ID固定 0x004C
    if 0x004C in adv_data.manufacturer_data:
        data = adv_data.manufacturer_data[0x004C]
        
        # AirTag 广播特征码
        if len(data) >= 27 and data[0] == 0x12:
            try:
                device_type = data[2]
                battery = data[3] >> 6
                battery_status = ["低", "中", "高", "满"][battery]
                is_sounding = (data[4] & 0x01) == 0x01

                airtag_info = {
                    "type": AIRTAG_DEVICE_TYPES.get(device_type, "未知苹果设备"),
                    "battery": battery_status,
                    "sounding": is_sounding,
                    "rssi": adv_data.rssi
                }
            except:
                pass
    return airtag_info

async def scan_airtags():
    print("=" * 60)
    print("🎯 实时扫描附近 AirTag (Ctrl+C 停止)")
    print("=" * 60)
    
    detected = set()  # 去重
    
    def callback(device, adv_data):
        mac = device.address.upper()
        # 修复：变量名从 adv 改为正确的 adv_data
        airtag = parse_airtag(adv_data)
        
        if airtag and mac not in detected:
            detected.add(mac)
            ts = int(time.time() * 1000)
            
            print(f"\n✅ 发现 AirTag 设备")
            print(f"   MAC 地址: {mac}")
            print(f"   设备类型: {airtag['type']}")
            print(f"   信号强度: {airtag['rssi']} dBm")
            print(f"   电池状态: {airtag['battery']}")
            print(f"   播放声音: {'是' if airtag['sounding'] else '否'}")
            print(f"   扫描时间: {ts}")
            print("-" * 50)

    # 持续扫描
    async with BleakScanner(callback):
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(scan_airtags())
    except KeyboardInterrupt:
        print("\n🛑 已停止扫描")