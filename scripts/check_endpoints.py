import requests

BASE_URL = "http://127.0.0.1:8000"

print("=== 1. GET /health ===")
print(requests.get(f"{BASE_URL}/health").json())

print("\n=== 2. GET /models ===")
print(requests.get(f"{BASE_URL}/models").json())

print("\n=== 3. POST /predict (200 OK) ===")
valid_payload = {"features": [0.0] * 54 + [1.0, 1.0, 1.0]}
resp = requests.post(f"{BASE_URL}/predict", json=valid_payload)
print("Status:", resp.status_code)
print("Response:", resp.json())

print("\n=== 4. POST /predict (422 Error) ===")
invalid_payload = {"features": [0.0] * 56}
resp = requests.post(f"{BASE_URL}/predict", json=invalid_payload)
print("Status:", resp.status_code)
print("Response:", resp.json())
