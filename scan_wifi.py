import pywifi
import time

# 初始化 WiFi 接口
wifi = pywifi.PyWiFi()
iface = wifi.interfaces()[0]  # 获取第一个WiFi网卡

print("===== 开始扫描 WiFi (2秒)... =====")
iface.scan()          # 触发扫描
time.sleep(2)         # 等待扫描完成
wifi_list = iface.scan_results()  # 获取结果
print(f"共找到 {len(wifi_list)} 个 WiFi 网络:\n")
# print(wifi_list)

# 输出所有 WiFi
# for i, ap in enumerate(wifi_list, 1):
#     ssid = ap.ssid.strip() or "隐藏WiFi"
#     bssid = ap.bssid.hex(':')  # 直接转 MAC 地址
#     dbm = ap.signal            # 直接获取 dBm（真实值！）
    
    # print(f"[{i}] SSID: {ssid:22} MAC: {bssid:20} 信号: {dbm} dBm")

print("\n===== 扫描完成 =====\n")
for i, ap in enumerate(wifi_list, 1):
    ssid = ap.ssid
    bssid = ap.bssid  # 直接用，Windows 上已经是 MAC 字符串
    dbm = ap.signal   # 直接是 dBm 真实信号强度
    
    # 格式化输出
    print(f"[{i}] SSID: {ssid:20} MAC: {bssid:20} signal: {dbm} dBm")