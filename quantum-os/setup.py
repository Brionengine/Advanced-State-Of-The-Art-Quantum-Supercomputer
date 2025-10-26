"""
Setup script for Quantum OS
"""

from setuptools import setup, find_packages

setup(
    name="quantum-os",
    version="1.0.0",
    author="Brionengine Team",
    description="Advanced Quantum Supercomputer Operating System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Brionengine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[
        "cirq>=1.3.0",
        "cirq-google>=1.3.0",
        "qiskit>=1.0.0",
        "qiskit-aer>=0.13.0",
        "qiskit-ibm-runtime>=0.17.0",
        "qiskit-ibm-provider>=0.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.2",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "psutil>=5.9.6",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
    ],
    extras_require={
        "tfq": ["tensorflow>=2.15.0", "tensorflow-quantum>=0.7.3"],
        "gpu": ["cupy-cuda12x>=12.3.0", "pycuda>=2022.2"],
        "distributed": ["dask[complete]>=2023.10.0", "ray[default]>=2.8.0"],
        "dev": ["pytest>=7.4.3", "pytest-asyncio>=0.21.1", "black>=23.11.0"],
    },
)
