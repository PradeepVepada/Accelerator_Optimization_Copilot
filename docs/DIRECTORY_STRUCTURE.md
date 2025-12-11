# 📂 Project Directory Structure

Here is the complete file structure for the **Accelerator Optimization Copilot**:

```text
accelerator_optimization_copilot/
├── backend/                          # Core logic and API
│   ├── ml/                           # Machine Learning components
│   │   ├── models/
│   │   │   ├── scheduler_opt.py      # ML Eviction Scoring Model
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── simulator/                    # Cache Simulation Engine
│   │   ├── memory_simulator.py       # Discrete-event memory simulator (LRU/FIFO/ML)
│   │   └── __init__.py
│   ├── tests/                        # Unit Tests
│   │   ├── test_memory_simulator.py  # Tests for cache logic
│   │   ├── test_scheduler.py         # Tests for ML scoring
│   │   └── test_workload.py          # Tests for trace generation
│   ├── workload/                     # Workload Generation
│   │   ├── generate_synthetic_workload.py # Synthetic trace generator
│   │   └── __init__.py
│   ├── main.py                       # FastAPI Application Entry Point
│   ├── statistical_validation.py     # Statistical tests (Chi-Square, KS)
│   └── __init__.py
├── ui/                               # Frontend User Interface
│   └── app.py                        # Streamlit Dashboard
├── ENGINEERING_GUIDE.md              # Detailed Senior Engineer's Guide
├── QUICKSTART.md                     # Quick start instructions
├── README.md                         # Project overview
├── requirements.txt                  # Python dependencies
└── test_api.py                       # API integration tests
```

## 🔑 Key Files Description

- **`backend/main.py`**: The brain of the application. Handles API requests from the UI.
- **`backend/workload/generate_synthetic_workload.py`**: Creates realistic memory access patterns for Transformers, CNNs, etc.
- **`backend/simulator/memory_simulator.py`**: Simulates the cache behavior and calculates latency/hits.
- **`backend/ml/models/scheduler_opt.py`**: The "Smart" part. Predicts which data to evict.
- **`ui/app.py`**: The visual dashboard where you control everything.
