"""
Benchmark Suite for Advanced Quantum Supercomputer

Provides comprehensive performance benchmarks and verification tests
for peer review by Google Research and Google Quantum AI
"""

__version__ = "1.0.0"
__author__ = "Brionengine Team"

from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "benchmark_results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)
