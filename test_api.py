import requests
import json

API_URL = "http://127.0.0.1:8000"

print("Testing Accelerator Optimization Copilot API Endpoints\n")
print("=" * 60)

# Test payload
payload = {
    "model_type": "Transformer",
    "num_layers": 4,
    "batch_size": 32,
    "seq_length": 128,
    "reuse_probability": 0.3
}

# Test 1: Simulate endpoint
print("\n1. Testing /simulate endpoint (LRU policy)...")
try:
    r = requests.post(f"{API_URL}/simulate?policy=LRU", json=payload, timeout=10)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=2)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Predict schedule endpoint
print("\n2. Testing /predict-schedule endpoint...")
try:
    r = requests.post(f"{API_URL}/predict-schedule", json=payload, timeout=10)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Returned {len(data)} trace entries")
    if len(data) > 0:
        print(f"   Sample entry: {json.dumps(data[0], indent=2)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Validate trace endpoint
print("\n3. Testing /validate-trace endpoint...")
try:
    r = requests.post(f"{API_URL}/validate-trace", json=payload, timeout=10)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=2)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Predict compile endpoint
print("\n4. Testing /predict-compile endpoint...")
try:
    compile_payload = {
        "code": "def matmul(a, b): return a @ b",
        "model_type": "Transformer",
        "opt_level": "O2"
    }
    r = requests.post(f"{API_URL}/predict-compile", json=compile_payload, timeout=10)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=2)}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("✅ API endpoint testing complete!")
