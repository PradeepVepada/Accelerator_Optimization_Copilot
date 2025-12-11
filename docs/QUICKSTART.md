# Accelerator Optimization Copilot - Quick Start Guide

## 🎯 Current Status

✅ **Backend Server**: Running on http://127.0.0.1:8000
✅ **Frontend UI**: Running on http://localhost:8501
✅ **All Tests**: Passing
✅ **All Endpoints**: Functional

---

## 🚀 Access Your Application

### Open the Streamlit UI
Visit: **http://localhost:8501**

The UI provides:
- Model workload selection (Transformer, CNN, MLP, MoE)
- Cache policy selection (LRU, FIFO, ML)
- Interactive sliders for configuration
- Real-time simulation results
- Statistical validation
- Compiler prediction

### API Documentation
Visit: **http://127.0.0.1:8000/docs**

Interactive Swagger UI for testing all endpoints

---

## 📝 How to Use the UI

1. **Select Model Type**: Choose from Transformer, CNN, MLP, or MoE
2. **Choose Cache Policy**: LRU, FIFO, or ML-based
3. **Configure Parameters**:
   - Layers: 4-48
   - Batch Size: 1-256
   - Sequence Length: 32-1024
   - Reuse Probability: 0.0-1.0
4. **Enable Statistical Validation**: Check the box to run chi-square and KS tests
5. **Click "Run Simulation"**: View results including:
   - Cache metrics (hits, misses, latency, evictions)
   - Eviction scores table
   - Cache occupancy heatmap
   - Statistical validation p-values

### Compiler Prediction
Scroll down to the "Compiler Prediction" section:
1. Paste kernel code
2. Select model type and optimization level
3. Click "Predict Compilation Results"
4. View latency/energy predictions and feature importance

---

## 🧪 Running Tests

### Unit Tests
```bash
cd C:\Users\vprad\.gemini\antigravity\scratch\accelerator_optimization_copilot

# Test workload generator
python backend/tests/test_workload.py

# Test memory simulator
python backend/tests/test_memory_simulator.py

# Test ML scheduler
python backend/tests/test_scheduler.py
```

### API Tests
```bash
python test_api.py
```

---

## 🔧 Restarting Services

### If Backend Stops
```bash
cd C:\Users\vprad\.gemini\antigravity\scratch\accelerator_optimization_copilot
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### If Frontend Stops
```bash
cd C:\Users\vprad\.gemini\antigravity\scratch\accelerator_optimization_copilot
streamlit run ui/app.py --server.port 8501
```

---

## 📊 API Endpoints

### POST /generate-trace
Generate synthetic workload trace
```json
{
  "model_type": "Transformer",
  "num_layers": 12,
  "batch_size": 32,
  "seq_length": 128,
  "reuse_probability": 0.3
}
```

### POST /predict-schedule
Get eviction scores for tensors
```json
{
  "model_type": "Transformer",
  "num_layers": 12,
  "batch_size": 32,
  "seq_length": 128,
  "reuse_probability": 0.3
}
```

### POST /simulate?policy=LRU
Run cache simulation
```json
{
  "model_type": "Transformer",
  "num_layers": 12,
  "batch_size": 32,
  "seq_length": 128,
  "reuse_probability": 0.3
}
```

### POST /validate-trace
Run statistical validation
```json
{
  "model_type": "Transformer",
  "num_layers": 12,
  "batch_size": 32,
  "seq_length": 128,
  "reuse_probability": 0.3
}
```

### POST /predict-compile
Predict compilation metrics
```json
{
  "code": "def matmul(a, b): return a @ b",
  "model_type": "Transformer",
  "opt_level": "O2"
}
```

---

## 📁 Project Location

**Full Path**: `C:\Users\vprad\.gemini\antigravity\scratch\accelerator_optimization_copilot`

### Key Files
- Backend: `backend/main.py`
- Frontend: `ui/app.py`
- Tests: `backend/tests/`
- Documentation: `README.md`

---

## ✅ Verification Checklist

- [x] Backend server running
- [x] Frontend UI running
- [x] All unit tests passing
- [x] All API endpoints responding
- [x] Statistical validation working
- [x] Charts rendering correctly
- [x] No runtime errors
- [x] No import errors

---

## 🎉 You're All Set!

The Accelerator Optimization Copilot is fully functional and ready to use!

**Next Steps**:
1. Open http://localhost:8501 in your browser
2. Try running a simulation with different model types
3. Experiment with different cache policies
4. View the statistical validation results
5. Test the compiler prediction feature

Enjoy exploring your accelerator optimization system! 🚀
