"""
Quantum Job Scheduler

Manages quantum job scheduling and execution queuing
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
import threading


class JobStatus(Enum):
    """Job status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QuantumJob:
    """Quantum job representation"""
    job_id: str
    circuit: Any
    shots: int
    backend_name: str
    priority: int = 1
    status: JobStatus = JobStatus.PENDING
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        return self.priority > other.priority


class QuantumScheduler:
    """
    Quantum job scheduler

    Supports multiple scheduling strategies:
    - FIFO (First In First Out)
    - Priority-based
    - Round-robin across backends
    """

    def __init__(self, resource_config=None):
        """
        Initialize scheduler

        Args:
            resource_config: Resource configuration
        """
        self.jobs: Dict[str, QuantumJob] = {}
        self.job_queue = PriorityQueue()
        self.scheduler_type = 'priority' if resource_config else 'fifo'
        self.max_concurrent_jobs = resource_config.max_concurrent_jobs if resource_config else 5

        self._running_jobs = 0
        self._lock = threading.Lock()

    def submit_job(
        self,
        job_params: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """
        Submit a quantum job

        Args:
            job_params: Job parameters (circuit, shots, backend, etc.)
            priority: Job priority (higher = more urgent)

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        job = QuantumJob(
            job_id=job_id,
            circuit=job_params.get('circuit'),
            shots=job_params.get('shots', 1024),
            backend_name=job_params.get('backend_name', 'default'),
            priority=priority,
            metadata=job_params.get('kwargs', {})
        )

        with self._lock:
            self.jobs[job_id] = job
            self.job_queue.put(job)

        return job_id

    def get_job(self, job_id: str) -> Optional[QuantumJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def mark_job_running(self, job_id: str):
        """Mark job as running"""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = JobStatus.RUNNING
                job.start_time = time.time()
                self._running_jobs += 1

    def mark_job_complete(
        self,
        job_id: str,
        success: bool = True,
        result: Any = None,
        error: Optional[str] = None
    ):
        """Mark job as complete"""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
                job.end_time = time.time()
                job.result = result
                job.error = error
                self._running_jobs = max(0, self._running_jobs - 1)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    return True
        return False

    def get_pending_jobs(self) -> List[QuantumJob]:
        """Get list of pending jobs"""
        return [
            job for job in self.jobs.values()
            if job.status == JobStatus.PENDING
        ]

    def get_running_jobs(self) -> List[QuantumJob]:
        """Get list of running jobs"""
        return [
            job for job in self.jobs.values()
            if job.status == JobStatus.RUNNING
        ]

    def get_completed_jobs(self) -> List[QuantumJob]:
        """Get list of completed jobs"""
        return [
            job for job in self.jobs.values()
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            'total_jobs': len(self.jobs),
            'pending': len(self.get_pending_jobs()),
            'running': len(self.get_running_jobs()),
            'completed': len(self.get_completed_jobs()),
            'scheduler_type': self.scheduler_type,
            'max_concurrent': self.max_concurrent_jobs
        }

    def wait_for_all_jobs(self, timeout: Optional[float] = None):
        """
        Wait for all jobs to complete

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()

        while True:
            running = len(self.get_running_jobs())
            pending = len(self.get_pending_jobs())

            if running == 0 and pending == 0:
                break

            if timeout and (time.time() - start_time) > timeout:
                break

            time.sleep(0.1)

    def clear_completed_jobs(self):
        """Clear completed jobs from memory"""
        with self._lock:
            completed_ids = [
                job_id for job_id, job in self.jobs.items()
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            ]
            for job_id in completed_ids:
                del self.jobs[job_id]
