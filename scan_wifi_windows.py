import subprocess
import time

def scan_wifi():
    print("===== 正在扫描周边 WiFi（不推荐使用） =====")

    try:
        subprocess.run("netsh wlan disconnect", shell=True, capture_output=True)
        time.sleep(0.5)
        # 执行系统命令扫描所有WiFi
        output = subprocess.check_output(
            "netsh wlan show networks mode=Bssid",
            shell=True, encoding="utf-8", errors="ignore"
        )
        time.sleep(2)

        ssid = ""
        bssid = ""
        signal = ""

        # 逐行解析
        for line in output.splitlines():
            # print(line)
            line = line.strip()

            if line.startswith("SSID"):
                ssid = line.split(":", 1)[1].strip()
            elif "BSSID" in line:
                bssid = line.split(":", 1)[1].strip()
            elif "信号" in line:
                signal = line.split(":", 1)[1].strip()
                # 打印完整信息
                print(f"SSID: {ssid:25} MAC: {bssid:20} 信号: {signal}")
                ssid = bssid = signal = ""

    except Exception as e:
        print(f"扫描过程中发生错误: {e}")
        print("扫描失败，请以管理员身份运行")

if __name__ == "__main__":
    scan_wifi()