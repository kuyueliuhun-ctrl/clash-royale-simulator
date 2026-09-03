#!/usr/bin/env python3
"""CDP 端口转发器：WSL 127.0.0.1:9222 → Windows 主机 <网关IP>:9222

WSL2 NAT 拓扑下，Windows 上 Chrome 的 CDP 端口对 WSL 不可见（loopback 隔离）。
本转发器在 WSL 侧把 127.0.0.1:9222 桥接到 Windows 主机的 9222，
使 harness 浏览器桥（默认找 localhost:9222）能连上用户 Chrome 的调试端口。
网关 IP 每次启动动态推导，WSL 重启后重跑本脚本即可。
"""
import socket
import threading
import subprocess
import sys
import time


def windows_host_ip():
    """动态取 WSL→Windows 的网关 IP（即 Windows vEthernet (WSL) 地址）"""
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True)
        return out.split()[2]
    except Exception:
        return "172.28.144.1"  # 上次已知值兜底


def pipe(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        try: a.close()
        except OSError: pass
        try: b.close()
        except OSError: pass


def handle(client, upstream):
    try:
        remote = socket.create_connection((upstream, 9222), timeout=5)
    except OSError as e:
        sys.stderr.write(f"[forward] 上游 {upstream}:9222 不可达: {e}\n")
        client.close()
        return
    threading.Thread(target=pipe, args=(client, remote), daemon=True).start()
    threading.Thread(target=pipe, args=(remote, client), daemon=True).start()


def main():
    upstream = windows_host_ip()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 9222))
    srv.listen(16)
    print(f"[forward] 127.0.0.1:9222 → {upstream}:9222 就绪", flush=True)
    while True:
        client, addr = srv.accept()
        threading.Thread(target=handle, args=(client, upstream), daemon=True).start()


if __name__ == "__main__":
    while True:
        try:
            main()
        except OSError as e:
            print(f"[forward] 重启监听: {e}", flush=True)
            time.sleep(2)
