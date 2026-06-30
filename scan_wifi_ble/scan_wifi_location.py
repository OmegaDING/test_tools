import pywifi
import time
import requests

# API 地址
API_URL = "http://api.cellocation.com:84/wifi/"

def scan_wifi():
    """
    使用 pywifi 扫描附近的 WiFi，并返回一个包含 BSSID 和 SSID 的列表
    """
    wifi = pywifi.PyWiFi()
    # 获取第一个无线网卡
    iface = wifi.interfaces()[0]
    
    print("===== 开始扫描 WiFi，请稍候... =====")
    iface.scan()
    # 等待扫描结果，通常需要几秒钟
    time.sleep(4) 
    scan_results = iface.scan_results()
    
    wifi_list = []
    for result in scan_results:
        # 过滤掉没有 BSSID 或 SSID 的结果
        if result.bssid and result.bssid != '00:00:00:00:00:00':
            ssid = result.ssid if result.ssid else "隐藏网络"
            wifi_list.append({'bssid': result.bssid.strip(":"), 'ssid': ssid})
    
    # 简单去重，保留第一个遇到的同名 MAC
    unique_wifi = {ap['bssid']: ap for ap in wifi_list}.values()
    return list(unique_wifi)

def get_location_by_mac(mac):
    """
    调用 API 根据 MAC 地址查询位置信息
    """
    params = {
        'mac': mac,
        'output': 'json',
        # 'coord' : 'gcj02'  # 返回的坐标系，可选 'wgs84', 'gcj02', 'bd09'
    }
    try:
        # print(f"正在查询 MAC 地址: {mac} 的位置信息...")
        response = requests.get(API_URL, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"errcode": response.status_code, "address": "请求失败"}
    except Exception as e:
        return {"errcode": -1, "address": str(e)}

def main():
    # 1. 扫描 WiFi
    wifi_list = scan_wifi()
    print(f"扫描完成，共发现 {len(wifi_list)} 个唯一的 WiFi 热点。\n")
    
    if not wifi_list:
        return

    # 2. 打印表头
    # 使用格式化字符串来对齐列
    print(f"{'SSID':<25} {'MAC地址':<20} {'错误码':<8} {'地址/备注'}")
    print("-" * 100)
    
    # 3. 遍历 WiFi 列表并查询位置
    for i, ap in enumerate(wifi_list):
        mac = ap['bssid']
        ssid = ap['ssid']
        # print(f"正在查询 ({i+1}/{len(wifi_list)}): {ssid} ({mac})")
        
        location_info = get_location_by_mac(mac)
        
        # 4. 格式化并打印结果
        errcode = location_info.get("errcode")
        address = location_info.get("address", "无")
        if address == '':
            address = "无"
        
        # 如果查询成功，可以拼接更详细的地址信息
        if errcode == 0:
            lat = location_info.get("lat")
            lon = location_info.get("lon")
            display_address = f"{address} (Lat: {lat}, Lon: {lon})"
        else:
            display_address = address
        # import pdb; pdb.set_trace()
        print(f"{ssid:<25} {mac:<20} {errcode:<8} {display_address}")
        
        # 为了避免请求过于频繁，增加一点延时
        time.sleep(0.2)
        
    print("\n===== 所有查询完成 =====")

if __name__ == "__main__":
    main()