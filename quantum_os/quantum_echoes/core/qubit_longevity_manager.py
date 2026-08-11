"""
Qubit Longevity Management System
==================================

Monitors and maintains long-lived qubits over extended periods (years).
Provides automated maintenance, health monitoring, and lifecycle management.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class QubitHealth(Enum):
    """Health status of a qubit."""
    EXCELLENT = "excellent"  # >99.99% fidelity
    GOOD = "good"  # 99.9-99.99% fidelity
    FAIR = "fair"  # 99-99.9% fidelity
    DEGRADED = "degraded"  # 95-99% fidelity
    CRITICAL = "critical"  # <95% fidelity
    FAILED = "failed"  # Non-operational


class MaintenanceAction(Enum):
    """Types of maintenance actions."""
    ROUTINE_CALIBRATION = "routine_calibration"
    PARTICLE_REFRESH = "particle_refresh"
    ERROR_CORRECTION_BOOST = "ec_boost"
    COOLING_OPTIMIZATION = "cooling_optimization"
    ISOLATION_ENHANCEMENT = "isolation_enhancement"
    FULL_REGENERATION = "regeneration"
    RETIREMENT = "retirement"


@dataclass
class MaintenanceSchedule:
    """Schedule for qubit maintenance."""
    qubit_id: str
    next_routine: datetime
    next_refresh: datetime
    next_calibration: datetime
    routine_interval: timedelta = field(default_factory=lambda: timedelta(days=7))
    refresh_interval: timedelta = field(default_factory=lambda: timedelta(days=30))
    calibration_interval: timedelta = field(default_factory=lambda: timedelta(days=90))


@dataclass
class QubitMetrics:
    """Comprehensive metrics for a qubit."""
    qubit_id: str
    creation_time: datetime
    last_maintenance: datetime
    uptime_seconds: float
    state_fidelity: float
    coherence_T1: float  # seconds
    coherence_T2: float  # seconds
    gate_fidelity: float
    measurement_fidelity: float
    error_rate: float
    correction_count: int
    health_status: QubitHealth

    def get_age_days(self) -> float:
        """Get qubit age in days."""
        return (datetime.now() - self.creation_time).total_seconds() / (24 * 3600)

    def get_uptime_days(self) -> float:
        """Get operational uptime in days."""
        return self.uptime_seconds / (24 * 3600)

    def estimate_remaining_lifetime_years(self) -> float:
        """Estimate remaining useful lifetime."""
        # Based on current degradation rate
        if self.state_fidelity < 0.95:
            return 0.0

        # Extrapolate from current metrics
        age_days = self.get_age_days()
        fidelity_degradation = (1.0 - self.state_fidelity) / max(age_days, 1)

        # Days until fidelity reaches 95% threshold
        days_remaining = (self.state_fidelity - 0.95) / max(fidelity_degradation, 1e-6)

        years_remaining = days_remaining / 365.25

        return max(0.0, years_remaining)


class QubitLongevityManager:
    """
    Manages lifecycle and maintenance of long-lived qubits.

    Provides automated monitoring, maintenance scheduling, and
    health management for qubits with multi-year lifetimes.
    """

    def __init__(self,
                 maintenance_mode: str = "automated",
                 health_check_interval: float = 3600.0,
                 auto_repair: bool = True):
        """
        Initialize longevity manager.

        Args:
            maintenance_mode: 'automated' or 'manual'
            health_check_interval: Time between health checks (seconds)
            auto_repair: Automatically repair degraded qubits
        """
        self.maintenance_mode = maintenance_mode
        self.health_check_interval = health_check_interval
        self.auto_repair = auto_repair

        # Qubit registry
        self.qubits: Dict[str, QubitMetrics] = {}

        # Maintenance schedules
        self.schedules: Dict[str, MaintenanceSchedule] = {}

        # Maintenance history
        self.maintenance_history: List[Dict] = []

        # Performance tracking
        self.total_maintenance_actions = 0
        self.successful_repairs = 0
        self.retired_qubits = 0

        # Alert thresholds
        self.fidelity_warning_threshold = 0.995
        self.fidelity_critical_threshold = 0.95

        logger.info(f"Initialized qubit longevity manager in {maintenance_mode} mode")

    def register_qubit(self,
                      qubit_id: str,
                      initial_metrics: Optional[Dict] = None) -> bool:
        """
        Register a new qubit for management.

        Args:
            qubit_id: Unique identifier
            initial_metrics: Initial performance metrics

        Returns:
            True if registration successful
        """
        if qubit_id in self.qubits:
            logger.warning(f"Qubit {qubit_id} already registered")
            return False

        # Create initial metrics
        if initial_metrics is None:
            initial_metrics = {
                'state_fidelity': 0.9999,
                'coherence_T1': 1e6,
                'coherence_T2': 1e5,
                'gate_fidelity': 0.9999,
                'measurement_fidelity': 0.999,
                'error_rate': 1e-12
            }

        now = datetime.now()

        metrics = QubitMetrics(
            qubit_id=qubit_id,
            creation_time=now,
            last_maintenance=now,
            uptime_seconds=0.0,
            state_fidelity=initial_metrics['state_fidelity'],
            coherence_T1=initial_metrics['coherence_T1'],
            coherence_T2=initial_metrics['coherence_T2'],
            gate_fidelity=initial_metrics['gate_fidelity'],
            measurement_fidelity=initial_metrics['measurement_fidelity'],
            error_rate=initial_metrics['error_rate'],
            correction_count=0,
            health_status=self._assess_health(initial_metrics['state_fidelity'])
        )

        self.qubits[qubit_id] = metrics

        # Create maintenance schedule
        schedule = MaintenanceSchedule(
            qubit_id=qubit_id,
            next_routine=now + timedelta(days=7),
            next_refresh=now + timedelta(days=30),
            next_calibration=now + timedelta(days=90)
        )

        self.schedules[qubit_id] = schedule

        logger.info(f"Registered qubit {qubit_id} with health: {metrics.health_status.value}")

        return True

    def _assess_health(self, fidelity: float) -> QubitHealth:
        """Assess qubit health from fidelity."""
        if fidelity >= 0.9999:
            return QubitHealth.EXCELLENT
        elif fidelity >= 0.999:
            return QubitHealth.GOOD
        elif fidelity >= 0.99:
            return QubitHealth.FAIR
        elif fidelity >= 0.95:
            return QubitHealth.DEGRADED
        elif fidelity >= 0.5:
            return QubitHealth.CRITICAL
        else:
            return QubitHealth.FAILED

    def update_metrics(self, qubit_id: str, new_metrics: Dict):
        """
        Update qubit metrics.

        Args:
            qubit_id: Qubit identifier
            new_metrics: Updated metrics
        """
        if qubit_id not in self.qubits:
            logger.error(f"Qubit {qubit_id} not registered")
            return

        metrics = self.qubits[qubit_id]

        # Update metrics
        if 'state_fidelity' in new_metrics:
            metrics.state_fidelity = new_metrics['state_fidelity']

        if 'coherence_T1' in new_metrics:
            metrics.coherence_T1 = new_metrics['coherence_T1']

        if 'coherence_T2' in new_metrics:
            metrics.coherence_T2 = new_metrics['coherence_T2']

        if 'gate_fidelity' in new_metrics:
            metrics.gate_fidelity = new_metrics['gate_fidelity']

        if 'error_rate' in new_metrics:
            metrics.error_rate = new_metrics['error_rate']

        if 'uptime' in new_metrics:
            metrics.uptime_seconds += new_metrics['uptime']

        if 'corrections' in new_metrics:
            metrics.correction_count += new_metrics['corrections']

        # Reassess health
        metrics.health_status = self._assess_health(metrics.state_fidelity)

        # Check if maintenance needed
        if self.auto_repair:
            self._check_maintenance_needed(qubit_id)

    def _check_maintenance_needed(self, qubit_id: str) -> bool:
        """
        Check if qubit needs maintenance.

        Args:
            qubit_id: Qubit identifier

        Returns:
            True if maintenance performed
        """
        metrics = self.qubits[qubit_id]
        schedule = self.schedules[qubit_id]
        now = datetime.now()

        maintenance_performed = False

        # Check health-based triggers
        if metrics.health_status == QubitHealth.CRITICAL:
            self.perform_maintenance(qubit_id, MaintenanceAction.FULL_REGENERATION)
            maintenance_performed = True

        elif metrics.health_status == QubitHealth.DEGRADED:
            self.perform_maintenance(qubit_id, MaintenanceAction.ERROR_CORRECTION_BOOST)
            maintenance_performed = True

        elif metrics.state_fidelity < self.fidelity_warning_threshold:
            self.perform_maintenance(qubit_id, MaintenanceAction.PARTICLE_REFRESH)
            maintenance_performed = True

        # Check scheduled maintenance
        if now >= schedule.next_routine:
            self.perform_maintenance(qubit_id, MaintenanceAction.ROUTINE_CALIBRATION)
            schedule.next_routine = now + schedule.routine_interval
            maintenance_performed = True

        if now >= schedule.next_refresh:
            self.perform_maintenance(qubit_id, MaintenanceAction.PARTICLE_REFRESH)
            schedule.next_refresh = now + schedule.refresh_interval
            maintenance_performed = True

        if now >= schedule.next_calibration:
            self.perform_maintenance(qubit_id, MaintenanceAction.ROUTINE_CALIBRATION)
            schedule.next_calibration = now + schedule.calibration_interval
            maintenance_performed = True

        return maintenance_performed

    def perform_maintenance(self,
                           qubit_id: str,
                           action: MaintenanceAction) -> bool:
        """
        Perform maintenance action on qubit.

        Args:
            qubit_id: Qubit identifier
            action: Maintenance action to perform

        Returns:
            True if successful
        """
        if qubit_id not in self.qubits:
            logger.error(f"Qubit {qubit_id} not registered")
            return False

        metrics = self.qubits[qubit_id]

        logger.info(f"Performing {action.value} on qubit {qubit_id}")

        success = False
        fidelity_improvement = 0.0

        if action == MaintenanceAction.ROUTINE_CALIBRATION:
            # Improve gate fidelity
            metrics.gate_fidelity = min(0.99999, metrics.gate_fidelity * 1.001)
            fidelity_improvement = 0.0001
            success = True

        elif action == MaintenanceAction.PARTICLE_REFRESH:
            # Refresh particle reservoir, improve coherence
            metrics.coherence_T1 *= 1.1
            metrics.coherence_T2 *= 1.1
            metrics.state_fidelity = min(0.9999, metrics.state_fidelity + 0.001)
            fidelity_improvement = 0.001
            success = True

        elif action == MaintenanceAction.ERROR_CORRECTION_BOOST:
            # Enhance error correction
            metrics.error_rate *= 0.5
            metrics.state_fidelity = min(0.9999, metrics.state_fidelity + 0.01)
            fidelity_improvement = 0.01
            success = True

        elif action == MaintenanceAction.COOLING_OPTIMIZATION:
            # Optimize temperature, reduce thermal noise
            metrics.coherence_T2 *= 1.2
            metrics.state_fidelity = min(0.9999, metrics.state_fidelity + 0.0005)
            fidelity_improvement = 0.0005
            success = True

        elif action == MaintenanceAction.ISOLATION_ENHANCEMENT:
            # Improve environmental isolation
            metrics.coherence_T1 *= 1.15
            metrics.coherence_T2 *= 1.15
            fidelity_improvement = 0.002
            success = True

        elif action == MaintenanceAction.FULL_REGENERATION:
            # Complete regeneration
            metrics.state_fidelity = 0.9999
            metrics.coherence_T1 = 1e6
            metrics.coherence_T2 = 1e5
            metrics.gate_fidelity = 0.9999
            metrics.error_rate = 1e-12
            fidelity_improvement = 1.0 - metrics.state_fidelity
            success = True

        elif action == MaintenanceAction.RETIREMENT:
            # Retire qubit
            metrics.health_status = QubitHealth.FAILED
            self.retired_qubits += 1
            success = True

        if success:
            metrics.last_maintenance = datetime.now()
            metrics.health_status = self._assess_health(metrics.state_fidelity)

            self.total_maintenance_actions += 1
            if fidelity_improvement > 0:
                self.successful_repairs += 1

            # Record in history
            self.maintenance_history.append({
                'qubit_id': qubit_id,
                'action': action.value,
                'timestamp': datetime.now(),
                'fidelity_before': metrics.state_fidelity - fidelity_improvement,
                'fidelity_after': metrics.state_fidelity,
                'improvement': fidelity_improvement
            })

        return success

    def run_health_check_cycle(self) -> Dict[str, int]:
        """
        Run health check on all managed qubits.

        Returns:
            Summary of health statuses
        """
        health_counts = {
            QubitHealth.EXCELLENT: 0,
            QubitHealth.GOOD: 0,
            QubitHealth.FAIR: 0,
            QubitHealth.DEGRADED: 0,
            QubitHealth.CRITICAL: 0,
            QubitHealth.FAILED: 0
        }

        for qubit_id, metrics in self.qubits.items():
            health_counts[metrics.health_status] += 1

            # Check if maintenance needed
            if self.auto_repair:
                self._check_maintenance_needed(qubit_id)

        return {status.value: count for status, count in health_counts.items()}

    def get_fleet_statistics(self) -> Dict:
        """
        Get statistics for entire qubit fleet.

        Returns:
            Comprehensive fleet statistics
        """
        if not self.qubits:
            return {'total_qubits': 0}

        metrics_list = list(self.qubits.values())

        avg_fidelity = np.mean([m.state_fidelity for m in metrics_list])
        avg_T1 = np.mean([m.coherence_T1 for m in metrics_list])
        avg_T2 = np.mean([m.coherence_T2 for m in metrics_list])
        avg_age = np.mean([m.get_age_days() for m in metrics_list])
        avg_uptime = np.mean([m.get_uptime_days() for m in metrics_list])

        total_corrections = sum(m.correction_count for m in metrics_list)

        health_counts = self.run_health_check_cycle()

        stats = {
            'total_qubits': len(self.qubits),
            'active_qubits': len([m for m in metrics_list if m.health_status != QubitHealth.FAILED]),
            'retired_qubits': self.retired_qubits,
            'average_fidelity': avg_fidelity,
            'average_T1_seconds': avg_T1,
            'average_T2_seconds': avg_T2,
            'average_T1_years': avg_T1 / (365.25 * 24 * 3600),
            'average_T2_years': avg_T2 / (365.25 * 24 * 3600),
            'average_age_days': avg_age,
            'average_uptime_days': avg_uptime,
            'total_corrections': total_corrections,
            'total_maintenance_actions': self.total_maintenance_actions,
            'successful_repairs': self.successful_repairs,
            'health_distribution': health_counts
        }

        return stats

    def get_qubit_report(self, qubit_id: str) -> Optional[Dict]:
        """
        Get detailed report for specific qubit.

        Args:
            qubit_id: Qubit identifier

        Returns:
            Detailed qubit report
        """
        if qubit_id not in self.qubits:
            return None

        metrics = self.qubits[qubit_id]
        schedule = self.schedules[qubit_id]

        report = {
            'qubit_id': qubit_id,
            'health_status': metrics.health_status.value,
            'age_days': metrics.get_age_days(),
            'uptime_days': metrics.get_uptime_days(),
            'estimated_remaining_lifetime_years': metrics.estimate_remaining_lifetime_years(),
            'metrics': {
                'state_fidelity': metrics.state_fidelity,
                'coherence_T1_seconds': metrics.coherence_T1,
                'coherence_T2_seconds': metrics.coherence_T2,
                'coherence_T1_years': metrics.coherence_T1 / (365.25 * 24 * 3600),
                'coherence_T2_years': metrics.coherence_T2 / (365.25 * 24 * 3600),
                'gate_fidelity': metrics.gate_fidelity,
                'measurement_fidelity': metrics.measurement_fidelity,
                'error_rate': metrics.error_rate,
                'correction_count': metrics.correction_count
            },
            'maintenance': {
                'last_maintenance': metrics.last_maintenance.isoformat(),
                'next_routine': schedule.next_routine.isoformat(),
                'next_refresh': schedule.next_refresh.isoformat(),
                'next_calibration': schedule.next_calibration.isoformat()
            }
        }

        return report

    def print_fleet_status(self):
        """Print formatted fleet status."""
        stats = self.get_fleet_statistics()

        print("\n" + "="*70)
        print("QUBIT LONGEVITY MANAGER - FLEET STATUS")
        print("="*70)
        print(f"Total Qubits: {stats['total_qubits']}")
        print(f"Active Qubits: {stats['active_qubits']}")
        print(f"Retired Qubits: {stats['retired_qubits']}")
        print()
        print(f"Fleet Averages:")
        print(f"  State Fidelity: {stats['average_fidelity']:.6f}")
        print(f"  T1 Time: {stats['average_T1_seconds']:.2e} s ({stats['average_T1_years']:.2f} years)")
        print(f"  T2 Time: {stats['average_T2_seconds']:.2e} s ({stats['average_T2_years']:.2f} years)")
        print(f"  Age: {stats['average_age_days']:.1f} days")
        print(f"  Uptime: {stats['average_uptime_days']:.1f} days")
        print()
        print(f"Maintenance:")
        print(f"  Total Actions: {stats['total_maintenance_actions']}")
        print(f"  Successful Repairs: {stats['successful_repairs']}")
        print(f"  Total Corrections: {stats['total_corrections']}")
        print()
        print("Health Distribution:")
        for status, count in stats['health_distribution'].items():
            if count > 0:
                print(f"  {status.capitalize()}: {count}")
        print("="*70 + "\n")

    def print_qubit_status(self, qubit_id: str):
        """Print formatted status for specific qubit."""
        report = self.get_qubit_report(qubit_id)

        if report is None:
            print(f"Qubit {qubit_id} not found")
            return

        print("\n" + "="*70)
        print(f"QUBIT STATUS REPORT: {qubit_id}")
        print("="*70)
        print(f"Health: {report['health_status'].upper()}")
        print(f"Age: {report['age_days']:.2f} days")
        print(f"Uptime: {report['uptime_days']:.2f} days")
        print(f"Estimated Remaining Lifetime: {report['estimated_remaining_lifetime_years']:.1f} years")
        print()
        print("Performance Metrics:")
        print(f"  State Fidelity: {report['metrics']['state_fidelity']:.8f}")
        print(f"  T1: {report['metrics']['coherence_T1_seconds']:.2e} s "
              f"({report['metrics']['coherence_T1_years']:.2f} years)")
        print(f"  T2: {report['metrics']['coherence_T2_seconds']:.2e} s "
              f"({report['metrics']['coherence_T2_years']:.2f} years)")
        print(f"  Gate Fidelity: {report['metrics']['gate_fidelity']:.8f}")
        print(f"  Error Rate: {report['metrics']['error_rate']:.2e}")
        print(f"  Corrections Applied: {report['metrics']['correction_count']}")
        print()
        print("Maintenance Schedule:")
        print(f"  Last Maintenance: {report['maintenance']['last_maintenance']}")
        print(f"  Next Routine: {report['maintenance']['next_routine']}")
        print(f"  Next Refresh: {report['maintenance']['next_refresh']}")
        print(f"  Next Calibration: {report['maintenance']['next_calibration']}")
        print("="*70 + "\n")
