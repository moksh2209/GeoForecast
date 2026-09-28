import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000/api"

def get(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def main():
    print("Testing /api/health...")
    status, data = get(f"{BASE_URL}/health")
    print(status, data)
    
    print("\nTesting /api/districts...")
    status, data = get(f"{BASE_URL}/districts")
    print(status, data[:3] if isinstance(data, list) else data)
    
    for d in ["Pune", "Nashik", "Nagpur"]:
        for h in [1, 3, 6]:
            print(f"\nTesting /api/forecast/{d}?horizon={h}...")
            status, data = get(f"{BASE_URL}/forecast/{d}?horizon={h}")
            print(status)
            if status == 200:
                print(f"Pred: {data['predicted_groundwater_level_m_bgl']:.2f}, Model: {data['model']}")
            else:
                print(data)
                
    print("\nTesting invalid district...")
    status, data = get(f"{BASE_URL}/forecast/InvalidXYZ?horizon=1")
    print(status, data)
    
    print("\nTesting Maharashtra...")
    for h in [1, 3, 6]:
        print(f"\nTesting /api/forecast/maharashtra?horizon={h}...")
        status, data = get(f"{BASE_URL}/forecast/maharashtra?horizon={h}")
        print(status)
        if status == 200:
            print(f"Pred: {data['predicted_groundwater_level_m_bgl']:.2f}, Model: {data['model']}")
            
if __name__ == "__main__":
    main()
