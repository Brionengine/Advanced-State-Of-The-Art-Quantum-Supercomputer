# Quantum OS Installation Guide

## Quick Installation

```bash
cd "/mnt/c/Adv Quantum Supercomputer/quantum-os"

# Install all dependencies
pip install -r requirements.txt

# Or install core dependencies only
pip install cirq cirq-google qiskit qiskit-aer qiskit-ibm-runtime tensorflow tensorflow-quantum
```

## Step-by-Step Installation

### 1. Python Environment

Ensure you have Python 3.9 or higher:
```bash
python --version
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv quantum-env
source quantum-env/bin/activate  # On Windows: quantum-env\Scripts\activate
```

### 3. Install Quantum Frameworks

**Install Cirq (Google Quantum):**
```bash
pip install cirq cirq-google
```

**Install Qiskit (IBM Quantum):**
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime qiskit-ibm-provider
```

**Install TensorFlow Quantum:**
```bash
pip install tensorflow tensorflow-quantum
```

### 4. Install GPU Support (Optional but Recommended)

For GPU acceleration with TensorFlow Quantum:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# Ensure you have:
# - NVIDIA GPU with CUDA support
# - CUDA Toolkit 12.x installed
# - cuDNN library
```

### 5. Configure IBM Quantum Access

```bash
# Save your IBM Quantum token
export IBM_QUANTUM_TOKEN="your_token_here"

# Or save permanently
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN', overwrite=True)"
```

### 6. Verify Installation

```bash
cd "/mnt/c/Adv Quantum Supercomputer/quantum-os"
python -c "from quantum_os import create_quantum_os; qos = create_quantum_os(); print(qos.list_backends())"
```

## Running Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Quantum supercomputer demonstration
python examples/quantum_supercomputer_demo.py
```

## Running Tests

```bash
# Install pytest
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_backends.py -v
```

## Configuration

### Using Environment Variables

```bash
export IBM_QUANTUM_TOKEN="your_ibm_token"
export GOOGLE_CLOUD_PROJECT="your_project_id"
export QUANTUM_OS_CONFIG="/path/to/config.yaml"
```

### Using Configuration File

Create `config.yaml`:

```yaml
backends:
  ibm_brisbane:
    backend_type: qiskit
    execution_mode: real_quantum
    enabled: true
    priority: 10
    credentials:
      api_token: ${IBM_QUANTUM_TOKEN}

error_correction:
  enabled: true
  method: surface_code
  code_distance: 3

resources:
  max_qubits: 100
  gpu_enabled: true
```

Then load it:

```python
from quantum_os import QuantumOSConfig, QuantumOS

config = QuantumOSConfig('/path/to/config.yaml')
qos = QuantumOS(config)
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### IBM Quantum Connection Issues

Verify your token:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print(service.backends())
```

### GPU Not Detected

Check TensorFlow GPU setup:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Next Steps

1. Read the README.md for usage examples
2. Explore examples/ directory
3. Try basic_usage.py to get started
4. Configure your IBM Quantum or Google Quantum credentials
5. Run quantum_supercomputer_demo.py to see advanced features

## Support

- GitHub: https://github.com/Brionengine
- Twitter/X: @Brionengine

**Happy Quantum Computing!**
