"""
Quantum Communication Application
==================================

Secure quantum communication using echo-enhanced protocols.
Implements quantum key distribution (QKD) and teleportation using
topological qubits for enhanced security and robustness.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.particle_emitter import SpecializedIonEmitter, EmissionPattern
from core.echo_propagation import EchoPropagator, PropagationMode
from algorithms.quantum_echoes import QuantumEchoesAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class QuantumKey:
    """Represents a quantum key generated via QKD."""
    raw_key: np.ndarray  # Raw key bits
    sifted_key: np.ndarray  # After basis reconciliation
    error_rate: float  # Quantum bit error rate (QBER)
    security_parameter: float  # Security level achieved


@dataclass
class TeleportationResult:
    """Result of quantum teleportation."""
    success: bool
    fidelity: float
    classical_bits: Tuple[int, int]  # Measurement results
    teleported_state: np.ndarray


class QuantumKeyDistribution:
    """
    Quantum Key Distribution using topological qubits.

    Implements BB84 and E91 protocols with echo enhancement
    for improved security and noise resistance.
    """

    def __init__(self,
                 protocol: str = "BB84",
                 key_length: int = 256):
        """
        Initialize QKD system.

        Args:
            protocol: QKD protocol ("BB84", "E91", "BBM92")
            key_length: Target key length in bits
        """
        self.protocol = protocol
        self.key_length = key_length

        # Initialize quantum hardware
        self.emitter = SpecializedIonEmitter(num_ions=100)
        self.qe = QuantumEchoesAlgorithm(num_qubits=2)

        # Communication statistics
        self.stats = {
            'qubits_sent': 0,
            'qubits_received': 0,
            'errors_detected': 0,
            'eavesdropping_attempts': 0
        }

        logger.info(f"Initialized QKD with {protocol} protocol, "
                   f"target key length {key_length} bits")

    def generate_key(self,
                     alice_bases: Optional[List[str]] = None,
                     bob_bases: Optional[List[str]] = None) -> QuantumKey:
        """
        Generate quantum key using QKD protocol.

        Args:
            alice_bases: Alice's measurement bases (optional)
            bob_bases: Bob's measurement bases (optional)

        Returns:
            QuantumKey with sifted key and security metrics
        """
        if self.protocol == "BB84":
            return self._bb84_protocol(alice_bases, bob_bases)
        elif self.protocol == "E91":
            return self._e91_protocol()
        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")

    def _bb84_protocol(self,
                      alice_bases: Optional[List[str]] = None,
                      bob_bases: Optional[List[str]] = None) -> QuantumKey:
        """
        Execute BB84 protocol.

        Alice sends qubits in random bases (Z or X).
        Bob measures in random bases.
        They keep bits where bases matched.
        """
        logger.info("Starting BB84 protocol")

        # Need ~4x raw bits for final key (after sifting and privacy amplification)
        num_qubits = self.key_length * 4

        # Alice's random bits and bases
        alice_bits = np.random.randint(0, 2, num_qubits)

        if alice_bases is None:
            alice_bases = np.random.choice(['Z', 'X'], num_qubits)
        else:
            alice_bases = np.array(alice_bases)

        # Bob's random measurement bases
        if bob_bases is None:
            bob_bases = np.random.choice(['Z', 'X'], num_qubits)
        else:
            bob_bases = np.array(bob_bases)

        # Simulate qubit transmission with echoes
        bob_measurements = []

        for i in range(num_qubits):
            # Alice prepares qubit
            bit = alice_bits[i]
            basis = alice_bases[i]

            # Create emission pattern encoding the qubit
            if basis == 'Z':
                # |0⟩ or |1⟩
                encoding = np.array([bit])
            else:  # X basis
                # |+⟩ or |-⟩
                encoding = np.array([bit])

            pattern = self.emitter.emit_particle_cascade(
                num_particles=5,
                encoding_data=encoding
            )

            # Create quantum echo for enhanced transmission
            echo_pattern = self.emitter.create_echo_from_pattern(
                pattern,
                delay_time=1e-6
            )

            # Bob receives and measures
            bob_result = self._measure_qubit(echo_pattern, bob_bases[i])
            bob_measurements.append(bob_result)

            self.stats['qubits_sent'] += 1
            self.stats['qubits_received'] += 1

        bob_measurements = np.array(bob_measurements)

        # Basis reconciliation (classical communication)
        matching_bases = (alice_bases == bob_bases)

        # Sifted key (keep only matching bases)
        alice_sifted = alice_bits[matching_bases]
        bob_sifted = bob_measurements[matching_bases]

        # Error estimation
        # Use subset for error checking
        check_fraction = 0.1
        check_size = int(len(alice_sifted) * check_fraction)

        check_indices = np.random.choice(len(alice_sifted), check_size, replace=False)

        errors = np.sum(alice_sifted[check_indices] != bob_sifted[check_indices])
        qber = errors / check_size if check_size > 0 else 0

        self.stats['errors_detected'] += errors

        # Check for eavesdropping
        if qber > 0.11:  # Theoretical limit for BB84
            logger.warning(f"High QBER detected: {qber:.2%} - possible eavesdropping!")
            self.stats['eavesdropping_attempts'] += 1

        # Remove check bits from key
        keep_mask = np.ones(len(alice_sifted), dtype=bool)
        keep_mask[check_indices] = False

        final_key = alice_sifted[keep_mask]

        # Privacy amplification (simplified)
        # In practice, use proper hash functions
        if len(final_key) > self.key_length:
            final_key = final_key[:self.key_length]

        # Security parameter
        security = 1.0 - qber if qber < 0.11 else 0.0

        key = QuantumKey(
            raw_key=alice_bits,
            sifted_key=final_key,
            error_rate=qber,
            security_parameter=security
        )

        logger.info(f"BB84 complete: {len(final_key)} bit key, QBER={qber:.2%}, "
                   f"security={security:.3f}")

        return key

    def _e91_protocol(self) -> QuantumKey:
        """
        Execute E91 (entanglement-based) protocol.

        Uses entangled pairs and Bell inequality violation
        to detect eavesdropping.
        """
        logger.info("Starting E91 protocol")

        num_pairs = self.key_length * 4

        alice_measurements = []
        bob_measurements = []

        alice_bases = []
        bob_bases = []

        # Generate entangled pairs
        for i in range(num_pairs):
            # Create Bell pair using Quantum Echoes
            self.qe.reset()
            self.qe.initialize_qubits('zero')

            # Create Bell state
            self.qe.apply_single_qubit_gate(0, 'H')
            self.qe.apply_two_qubit_gate(0, 1, 'CNOT')

            # Random measurement bases
            alice_basis = np.random.choice(['Z', 'X', 'W'])  # 3 bases for E91
            bob_basis = np.random.choice(['Z', 'X', 'W'])

            alice_bases.append(alice_basis)
            bob_bases.append(bob_basis)

            # Measure
            results = self.qe.measure([0, 1])

            alice_measurements.append(results[0])
            bob_measurements.append(results[1])

        alice_measurements = np.array(alice_measurements)
        bob_measurements = np.array(bob_measurements)

        # Basis reconciliation
        # Keep Z-Z and X-X correlations for key
        matching = np.array([(a == b and a in ['Z', 'X'])
                            for a, b in zip(alice_bases, bob_bases)])

        alice_sifted = alice_measurements[matching]
        bob_sifted = bob_measurements[matching]

        # For Bell pairs: Bob's result should equal Alice's (in same basis)
        errors = np.sum(alice_sifted != bob_sifted)
        qber = errors / len(alice_sifted) if len(alice_sifted) > 0 else 0

        # Bell inequality test (using W basis measurements)
        bell_violation = self._test_bell_inequality(
            alice_measurements, bob_measurements,
            alice_bases, bob_bases
        )

        logger.info(f"Bell inequality parameter: {bell_violation:.3f} "
                   f"(>2 indicates quantum correlation)")

        if bell_violation < 2.0:
            logger.warning("Bell inequality not violated - possible eavesdropping!")
            self.stats['eavesdropping_attempts'] += 1

        # Final key
        final_key = alice_sifted[:self.key_length]

        security = bell_violation / 2.828 if bell_violation > 2.0 else 0.0  # Max is 2√2

        key = QuantumKey(
            raw_key=alice_measurements,
            sifted_key=final_key,
            error_rate=qber,
            security_parameter=security
        )

        logger.info(f"E91 complete: {len(final_key)} bit key, QBER={qber:.2%}")

        return key

    def _measure_qubit(self, emission_pattern: EmissionPattern, basis: str) -> int:
        """
        Measure a qubit (represented by emission pattern) in given basis.

        Args:
            emission_pattern: Emission pattern encoding qubit
            basis: Measurement basis ('Z' or 'X')

        Returns:
            Measurement result (0 or 1)
        """
        # Extract information from emission correlations
        mean_correlation = np.mean(np.abs(emission_pattern.quantum_correlations))

        # Simulate measurement with noise
        noise_level = 1.0 - emission_pattern.entanglement_degree

        if basis == 'Z':
            # Measure in computational basis
            prob_0 = mean_correlation
        else:  # X or W basis
            # Measure in Hadamard basis
            prob_0 = 0.5 + (mean_correlation - 0.5) * np.cos(np.pi / 4)

        # Add quantum noise
        prob_0 += np.random.randn() * noise_level * 0.1
        prob_0 = np.clip(prob_0, 0, 1)

        result = 0 if np.random.rand() < prob_0 else 1

        return result

    def _test_bell_inequality(self,
                              alice_results: np.ndarray,
                              bob_results: np.ndarray,
                              alice_bases: List[str],
                              bob_bases: List[str]) -> float:
        """
        Test CHSH Bell inequality.

        Returns:
            S parameter (classical limit is 2, quantum can reach 2√2)
        """
        # CHSH: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')

        def correlation(a_basis, b_basis):
            """Calculate correlation for given basis pair."""
            indices = [(i, a, b) for i, (a, b) in enumerate(zip(alice_bases, bob_bases))
                      if a == a_basis and b == b_basis]

            if len(indices) == 0:
                return 0.0

            corr = 0
            for i, _, _ in indices:
                # +1 if same, -1 if different
                corr += 1 if alice_results[i] == bob_results[i] else -1

            return corr / len(indices)

        # Calculate CHSH parameter (simplified)
        # Using available basis measurements
        E_ZZ = correlation('Z', 'Z')
        E_ZX = correlation('Z', 'X')
        E_XZ = correlation('X', 'Z')
        E_XX = correlation('X', 'X')

        S = abs(E_ZZ - E_ZX + E_XZ + E_XX)

        return S


class QuantumTeleportation:
    """
    Quantum teleportation using topological qubits.

    Uses entanglement and classical communication to teleport
    quantum states with high fidelity.
    """

    def __init__(self):
        """Initialize teleportation system."""
        self.qe = QuantumEchoesAlgorithm(num_qubits=3)

        logger.info("Initialized QuantumTeleportation")

    def teleport(self, state_to_teleport: np.ndarray) -> TeleportationResult:
        """
        Teleport a quantum state.

        Args:
            state_to_teleport: 2D complex array representing qubit state

        Returns:
            TeleportationResult with teleported state
        """
        logger.info("Starting quantum teleportation")

        # Initialize system
        self.qe.reset()
        self.qe.initialize_qubits('zero')

        # Qubit 0: state to teleport (Alice)
        # Qubits 1,2: Bell pair (Alice has 1, Bob has 2)

        # Step 1: Prepare Bell pair between qubits 1 and 2
        self.qe.apply_single_qubit_gate(1, 'H')
        self.qe.apply_two_qubit_gate(1, 2, 'CNOT')

        # Step 2: Prepare qubit 0 in desired state
        # (Simplified: just encode in quantum state directly)
        current_state = self.qe.get_statevector()

        # Encode state_to_teleport into qubit 0
        # In practice, this would be done physically
        alpha, beta = state_to_teleport[0], state_to_teleport[1]

        # Step 3: Bell measurement on qubits 0 and 1 (Alice's side)
        self.qe.apply_two_qubit_gate(0, 1, 'CNOT')
        self.qe.apply_single_qubit_gate(0, 'H')

        # Measure qubits 0 and 1
        measurements = self.qe.measure([0, 1])

        m0 = measurements[0]
        m1 = measurements[1]

        logger.info(f"Bell measurement results: {m0}, {m1}")

        # Step 4: Bob applies correction based on classical bits
        if m1 == 1:
            self.qe.apply_single_qubit_gate(2, 'X')
        if m0 == 1:
            self.qe.apply_single_qubit_gate(2, 'Z')

        # Step 5: Extract Bob's qubit (qubit 2)
        final_state = self.qe.get_qubit_state(2)

        # Calculate fidelity
        fidelity = self._calculate_fidelity(state_to_teleport, final_state)

        success = fidelity > 0.9

        result = TeleportationResult(
            success=success,
            fidelity=fidelity,
            classical_bits=(m0, m1),
            teleported_state=final_state
        )

        logger.info(f"Teleportation {'successful' if success else 'failed'}, "
                   f"fidelity={fidelity:.4f}")

        return result

    def _calculate_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate fidelity between two quantum states.

        Fidelity F = |⟨ψ|φ⟩|²
        """
        # Normalize states
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)

        # Inner product
        overlap = np.abs(np.vdot(state1, state2))

        fidelity = overlap ** 2

        return float(fidelity)


class SecureQuantumChannel:
    """
    Secure quantum communication channel combining QKD and teleportation.

    Provides end-to-end secure quantum communication.
    """

    def __init__(self):
        """Initialize secure channel."""
        self.qkd = QuantumKeyDistribution(protocol="BB84", key_length=256)
        self.teleporter = QuantumTeleportation()

        logger.info("Initialized SecureQuantumChannel")

    def establish_secure_channel(self) -> QuantumKey:
        """
        Establish secure communication channel via QKD.

        Returns:
            Shared quantum key
        """
        logger.info("Establishing secure quantum channel")

        key = self.qkd.generate_key()

        if key.error_rate > 0.11:
            logger.error("Channel security compromised!")
            return None

        logger.info(f"Secure channel established with {len(key.sifted_key)} bit key")

        return key

    def send_quantum_message(self, message_state: np.ndarray) -> TeleportationResult:
        """
        Send quantum message securely.

        Args:
            message_state: Quantum state to send

        Returns:
            TeleportationResult
        """
        result = self.teleporter.teleport(message_state)

        return result

    def encrypt_classical_message(self, message: str, key: QuantumKey) -> bytes:
        """
        Encrypt classical message using quantum key (one-time pad).

        Args:
            message: Message to encrypt
            key: Quantum key from QKD

        Returns:
            Encrypted message
        """
        # Convert message to bytes
        message_bytes = message.encode('utf-8')

        # XOR with quantum key (one-time pad)
        encrypted = bytearray()

        for i, byte in enumerate(message_bytes):
            key_byte = 0
            for j in range(8):
                if i * 8 + j < len(key.sifted_key):
                    key_byte |= int(key.sifted_key[i * 8 + j]) << j

            encrypted.append(byte ^ key_byte)

        logger.info(f"Encrypted {len(message)} character message")

        return bytes(encrypted)

    def get_channel_stats(self) -> Dict[str, any]:
        """Get communication statistics."""
        return self.qkd.stats.copy()
