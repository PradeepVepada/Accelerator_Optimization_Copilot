# Accelerator Optimization Copilot

A comprehensive system for optimizing accelerator workloads through ML-based cache scheduling, simulation, and compiler prediction.

## Features

- **Synthetic Workload Generation**: Generate realistic tensor access patterns for Transformer, CNN, MLP, and MoE models
- **ML-Based Eviction Scheduling**: Predict optimal cache eviction strategies using machine learning
- **Cache Simulation**: Simulate cache behavior with LRU, FIFO, and ML-based policies
- **Statistical Validation**: Validate workload quality with chi-square and KS tests
- **Compiler Prediction**: Predict latency and energy consumption for kernel code
- **Interactive UI**: Streamlit-based interface for visualization and experimentation

## Project Structure

```
accelerator_optimization_copilot/
├── backend/
│   ├── workload/
│   │   └── generate_synthetic_workload.py
│   ├── simulator/
│   │   └── memory_simulator.py
│   ├── ml/
│   │   └── models/
│   │       └── scheduler_opt.py
│   ├── statistical_validation.py
│   └── main.py
├── ui/
│   └── app.py
├── backend/tests/
│   ├── test_workload.py
│   ├── test_memory_simulator.py
│   └── test_scheduler.py
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Tests

Test the workload generator:
```bash
python backend/tests/test_workload.py
```

Test the memory simulator:
```bash
python backend/tests/test_memory_simulator.py
```

Test the ML scheduler:
```bash
python backend/tests/test_scheduler.py
```

### Running the Backend

Start the FastAPI server:
```bash
cd accelerator_optimization_copilot
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

API Documentation: `http://127.0.0.1:8000/docs`

### Running the Frontend

In a separate terminal, start the Streamlit UI:
```bash
cd accelerator_optimization_copilot
streamlit run ui/app.py
```

The UI will open in your browser at `http://localhost:8501`

## API Endpoints

- `POST /generate-trace`: Generate synthetic workload trace
- `POST /predict-schedule`: Predict eviction scores for tensors
- `POST /simulate`: Run cache simulation with specified policy
- `POST /validate-trace`: Run statistical validation on trace
- `POST /predict-compile`: Predict compilation latency and energy

## Components

### Workload Generator
Generates synthetic tensor access patterns with configurable:
- Model type (Transformer, CNN, MLP, MoE)
- Number of layers
- Batch size and sequence length
- Reuse probability

### Memory Simulator
Simulates cache behavior with:
- Configurable cache size
- Multiple eviction policies (LRU, FIFO, ML)
- Detailed metrics (hits, misses, latency, evictions)

### ML Scheduler
Predicts eviction scores based on:
- Tensor size
- Reuse distance
- Memory access stride

### Statistical Validation
Validates workload quality using:
- Chi-square test for reuse distribution
- Kolmogorov-Smirnov test for size distribution

## License

MIT
