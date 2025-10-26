"""
Quantum OS Configuration Management

Handles configuration for:
- Backend selection and credentials
- Error correction settings
- Resource management
- Security settings
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class BackendConfig:
    """Configuration for a quantum backend"""
    name: str
    backend_type: str  # 'cirq', 'qiskit', 'tfq'
    execution_mode: str = 'simulation'  # 'simulation' or 'real_quantum'
    enabled: bool = True
    priority: int = 1  # Higher priority backends used first
    credentials: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackendConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ErrorCorrectionConfig:
    """Error correction configuration"""
    enabled: bool = True
    method: str = 'surface_code'  # 'surface_code', 'repetition', 'steane', etc.
    code_distance: int = 3
    error_threshold: float = 0.01  # Target error rate
    mitigation_enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceConfig:
    """Resource management configuration"""
    max_qubits: int = 100
    max_concurrent_jobs: int = 5
    gpu_enabled: bool = True
    distributed_enabled: bool = False
    scheduler_type: str = 'fifo'  # 'fifo', 'priority', 'round_robin'
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration"""
    obfuscation_enabled: bool = True
    obfuscation_level: int = 2  # 0-3
    encryption_enabled: bool = True
    authentication_required: bool = False
    api_key: Optional[str] = None
    allowed_ips: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)


class QuantumOSConfig:
    """
    Main Quantum OS Configuration

    Manages all configuration settings for the quantum OS
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.backends: Dict[str, BackendConfig] = {}
        self.error_correction = ErrorCorrectionConfig()
        self.resources = ResourceConfig()
        self.security = SecurityConfig()

        # Load default configuration
        self._load_defaults()

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            # Try to load from environment variable
            env_config = os.getenv('QUANTUM_OS_CONFIG')
            if env_config and os.path.exists(env_config):
                self.load_from_file(env_config)

    def _load_defaults(self):
        """Load default configuration"""
        # Default backends
        self.backends = {
            'cirq_simulator': BackendConfig(
                name='cirq_simulator',
                backend_type='cirq',
                execution_mode='simulation',
                priority=2,
                options={'simulator_type': 'density_matrix'}
            ),
            'qiskit_aer': BackendConfig(
                name='aer_simulator',
                backend_type='qiskit',
                execution_mode='simulation',
                priority=1,
                options={}
            ),
            'tfq_simulator': BackendConfig(
                name='tfq_simulator',
                backend_type='tfq',
                execution_mode='simulation',
                priority=3,
                options={'use_gpu': True}
            ),
        }

        # Check for IBM Quantum credentials
        ibm_token = os.getenv('IBM_QUANTUM_TOKEN')
        if ibm_token:
            # Add IBM Brisbane
            self.backends['ibm_brisbane'] = BackendConfig(
                name='ibm_brisbane',
                backend_type='qiskit',
                execution_mode='real_quantum',
                enabled=True,
                priority=10,  # Highest priority for real hardware
                credentials={'api_token': ibm_token},
                options={'use_runtime': True}
            )

            # Add IBM Torino
            self.backends['ibm_torino'] = BackendConfig(
                name='ibm_torino',
                backend_type='qiskit',
                execution_mode='real_quantum',
                enabled=True,
                priority=9,
                credentials={'api_token': ibm_token},
                options={'use_runtime': True}
            )

        # Check for Google Quantum credentials
        google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        if google_project:
            self.backends['google_quantum'] = BackendConfig(
                name='rainbow',
                backend_type='cirq',
                execution_mode='real_quantum',
                enabled=False,  # Disabled until credentials verified
                priority=10,
                credentials={'project_id': google_project},
                options={}
            )

    def add_backend(self, backend_config: BackendConfig):
        """Add a backend configuration"""
        self.backends[backend_config.name] = backend_config

    def remove_backend(self, backend_name: str):
        """Remove a backend configuration"""
        if backend_name in self.backends:
            del self.backends[backend_name]

    def get_backend(self, backend_name: str) -> Optional[BackendConfig]:
        """Get backend configuration by name"""
        return self.backends.get(backend_name)

    def get_enabled_backends(self) -> List[BackendConfig]:
        """Get all enabled backends, sorted by priority"""
        enabled = [b for b in self.backends.values() if b.enabled]
        return sorted(enabled, key=lambda x: x.priority, reverse=True)

    def get_primary_backend(self) -> Optional[BackendConfig]:
        """Get the highest priority enabled backend"""
        enabled = self.get_enabled_backends()
        return enabled[0] if enabled else None

    def load_from_file(self, file_path: str):
        """
        Load configuration from YAML file

        Args:
            file_path: Path to YAML configuration file
        """
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Load backends
        if 'backends' in config_data:
            self.backends = {}
            for name, backend_data in config_data['backends'].items():
                backend_data['name'] = name
                self.backends[name] = BackendConfig.from_dict(backend_data)

        # Load error correction
        if 'error_correction' in config_data:
            self.error_correction = ErrorCorrectionConfig(**config_data['error_correction'])

        # Load resources
        if 'resources' in config_data:
            self.resources = ResourceConfig(**config_data['resources'])

        # Load security
        if 'security' in config_data:
            self.security = SecurityConfig(**config_data['security'])

        self.config_path = file_path

    def save_to_file(self, file_path: Optional[str] = None):
        """
        Save configuration to YAML file

        Args:
            file_path: Path to save configuration (uses self.config_path if None)
        """
        if file_path is None:
            file_path = self.config_path

        if not file_path:
            raise ValueError("No file path specified")

        config_data = {
            'backends': {
                name: backend.to_dict()
                for name, backend in self.backends.items()
            },
            'error_correction': asdict(self.error_correction),
            'resources': asdict(self.resources),
            'security': asdict(self.security)
        }

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'backends': {
                name: backend.to_dict()
                for name, backend in self.backends.items()
            },
            'error_correction': asdict(self.error_correction),
            'resources': asdict(self.resources),
            'security': asdict(self.security)
        }

    def __repr__(self) -> str:
        enabled_count = len(self.get_enabled_backends())
        return (
            f"QuantumOSConfig("
            f"backends={len(self.backends)}, "
            f"enabled={enabled_count}, "
            f"error_correction={self.error_correction.enabled})"
        )
