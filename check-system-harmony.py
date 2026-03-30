import requests
import json
import socket
import os

def check_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex((host, port)) == 0

def diagnose():
    print("\n--- [Checking] Trading System Harmony Diagnostic ---")
    
    # 1. Backend API (Port 4000)
    backend_up = check_port("127.0.0.1", 4000)
    print(f"[{'PASSED' if backend_up else 'FAILED'}] Backend API (Port 4000)")
    
    if backend_up:
        try:
            res = requests.get("http://127.0.0.1:4000/api/health", timeout=3).json()
            connected = res.get("connected", False)
            print(f"    - cTrader Connectivity: [{'CONNECTED' if connected else 'DISCONNECTED'}]")
            print(f"    - Paper Mode Ready: [{'YES' if res.get('paper_ready') else 'NO'}]")
        except:
            print("    - Error fetching health data.")

    # 2. Frontend Server (Port 5173)
    frontend_up = check_port("127.0.0.1", 5173)
    print(f"[{'PASSED' if frontend_up else 'FAILED'}] Frontend Dashboard (Port 5173)")

    # 3. Ollama AI (Port 11434)
    ollama_up = check_port("127.0.0.1", 11434)
    print(f"[{'PASSED' if ollama_up else 'FAILED'}] Ollama AI Engine (Port 11434)")
    
    if ollama_up:
        try:
            models = requests.get("http://127.0.0.1:11434/api/tags", timeout=3).json()
            model_names = [m['name'] for m in models.get('models', [])]
            print(f"    - Configured Models: {', '.join(model_names[:3])}...")
        except:
            print("    - Error reaching Ollama API.")

    print("\n--- [Conclusion] ---")
    if backend_up and frontend_up and ollama_up:
        print("[SUCCESS] ALL COMPONENTS ARE IN HARMONY!")
    else:
        print("[WARNING] SOME COMPONENTS ARE DISCONNECTED. Check your 'start-local.cmd' terminal logs.")

if __name__ == "__main__":
    diagnose()
