import requests
import json

# Test the updated validation endpoint
url = "http://127.0.0.1:8000/validate-trace"

payload = {
    "model_type": "Transformer",
    "num_layers": 12,
    "batch_size": 32,
    "seq_length": 128,
    "reuse_probability": 0.3
}

print("Testing /validate-trace endpoint...")
print(f"Request: {json.dumps(payload, indent=2)}\n")

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    print("Response:")
    print(json.dumps(result, indent=2))
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    print(f"Reuse Entropy: {result.get('reuse_entropy', 'N/A')}")
    print(f"  → {result.get('interpretation', {}).get('reuse_entropy', '')}")
    print(f"\nTemporal Locality Score: {result.get('temporal_locality_score', 'N/A')}")
    print(f"  → {result.get('interpretation', {}).get('temporal_locality', '')}")
    print(f"\nTensor Size Distribution (KS-test p-value): {result.get('tensor_size_distribution_p_value (KS-test)', 'N/A')}")
    print(f"  → {result.get('interpretation', {}).get('size_p_value', '')}")
    
    if 'reuse_statistics' in result:
        print(f"\nReuse Statistics:")
        for key, value in result['reuse_statistics'].items():
            print(f"  - {key}: {value}")
    
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
