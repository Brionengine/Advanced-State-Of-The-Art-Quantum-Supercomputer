"""
Quantum Resource Manager

Manages quantum computing resources:
- Qubit allocation
- GPU resources
- Distributed computing
- Resource monitoring
"""

import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading


@dataclass
class ResourceAllocation:
    """Resource allocation record"""
    allocation_id: str
    num_qubits: int
    backend_name: str
    allocated_at: float
    released_at: Optional[float] = None


class QuantumResourceManager:
    """
    Quantum resource manager

    Tracks and manages:
    - Qubit allocations
    - System resources (CPU, GPU, memory)
    - Backend availability
    """

    def __init__(self, resource_config=None):
        """
        Initialize resource manager

        Args:
            resource_config: Resource configuration
        """
        self.max_qubits = resource_config.max_qubits if resource_config else 100
        self.gpu_enabled = resource_config.gpu_enabled if resource_config else True

        self.allocations: Dict[str, ResourceAllocation] = {}
        self.backend_usage: Dict[str, int] = {}  # Track usage per backend

        self._lock = threading.Lock()
        self._monitor_thread = None
        self._monitoring = False

    def allocate_qubits(
        self,
        num_qubits: int,
        backend_name: str,
        allocation_id: str
    ) -> bool:
        """
        Allocate qubits for a job

        Args:
            num_qubits: Number of qubits needed
            backend_name: Backend name
            allocation_id: Unique allocation ID

        Returns:
            bool: True if allocation successful
        """
        with self._lock:
            # Check if within limits
            total_allocated = sum(
                alloc.num_qubits
                for alloc in self.allocations.values()
                if alloc.released_at is None
            )

            if total_allocated + num_qubits > self.max_qubits:
                return False

            # Create allocation
            self.allocations[allocation_id] = ResourceAllocation(
                allocation_id=allocation_id,
                num_qubits=num_qubits,
                backend_name=backend_name,
                allocated_at=time.time()
            )

            # Update backend usage
            self.backend_usage[backend_name] = self.backend_usage.get(backend_name, 0) + 1

            return True

    def release_qubits(self, allocation_id: str):
        """Release allocated qubits"""
        with self._lock:
            if allocation_id in self.allocations:
                alloc = self.allocations[allocation_id]
                alloc.released_at = time.time()

                # Update backend usage
                if alloc.backend_name in self.backend_usage:
                    self.backend_usage[alloc.backend_name] = max(
                        0,
                        self.backend_usage[alloc.backend_name] - 1
                    )

    def get_allocated_qubits(self) -> int:
        """Get total number of allocated qubits"""
        return sum(
            alloc.num_qubits
            for alloc in self.allocations.values()
            if alloc.released_at is None
        )

    def get_available_qubits(self) -> int:
        """Get number of available qubits"""
        return self.max_qubits - self.get_allocated_qubits()

    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        }

        # Check GPU if available
        if self.gpu_enabled:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                resources['gpu_count'] = len(gpus)
                resources['gpu_available'] = len(gpus) > 0
            except:
                resources['gpu_count'] = 0
                resources['gpu_available'] = False

        return resources

    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status"""
        return {
            'max_qubits': self.max_qubits,
            'allocated_qubits': self.get_allocated_qubits(),
            'available_qubits': self.get_available_qubits(),
            'active_allocations': len([
                a for a in self.allocations.values()
                if a.released_at is None
            ]),
            'backend_usage': dict(self.backend_usage),
            'system_resources': self.get_system_resources()
        }

    def start_monitoring(self, interval: float = 5.0):
        """
        Start resource monitoring in background

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return

        self._monitoring = True

        def monitor_loop():
            while self._monitoring:
                status = self.get_status()
                # Could log or send to monitoring system
                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
