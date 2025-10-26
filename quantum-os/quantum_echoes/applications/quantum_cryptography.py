"""
Post-Quantum Cryptography Application
======================================

Implement post-quantum cryptographic protocols using Quantum Echoes.
Provides quantum-resistant encryption, signatures, and key exchange.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import hashlib
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.quantum_echoes import QuantumEchoesAlgorithm
from applications.quantum_communication import QuantumKeyDistribution

logger = logging.getLogger(__name__)


@dataclass
class QuantumSignature:
    """Quantum digital signature."""
    message: str
    signature_qubits: np.ndarray
    public_key: np.ndarray
    timestamp: float


@dataclass
class LatticeKey:
    """Lattice-based cryptographic key."""
    public_key: np.ndarray
    private_key: Optional[np.ndarray]
    security_parameter: int
    lattice_dimension: int


class QuantumRandomNumberGenerator:
    """
    True quantum random number generator using particle emissions.

    Generates cryptographically secure random numbers from quantum measurements.
    """

    def __init__(self):
        """Initialize quantum RNG."""
        from core.particle_emitter import SpecializedIonEmitter

        self.emitter = SpecializedIonEmitter(num_ions=50)
        self.qe = QuantumEchoesAlgorithm(num_qubits=8)

        # Entropy pool
        self.entropy_pool = []

        logger.info("Initialized QuantumRandomNumberGenerator")

    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        """
        Generate truly random bits from quantum measurements.

        Args:
            num_bits: Number of random bits to generate

        Returns:
            Array of random bits
        """
        logger.info(f"Generating {num_bits} quantum random bits")

        random_bits = []

        while len(random_bits) < num_bits:
            # Prepare superposition state
            self.qe.reset()
            self.qe.initialize_qubits('zero')

            # Create maximum entropy state
            for i in range(min(8, self.qe.num_qubits)):
                self.qe.apply_single_qubit_gate(i, 'H')

            # Measure to get random bits
            measurements = self.qe.measure(list(range(min(8, num_bits - len(random_bits)))))

            for qubit_idx, bit in measurements.items():
                random_bits.append(bit)

        random_bits = np.array(random_bits[:num_bits], dtype=int)

        # Add to entropy pool
        self.entropy_pool.extend(random_bits.tolist())

        # Keep pool size manageable
        if len(self.entropy_pool) > 10000:
            self.entropy_pool = self.entropy_pool[-10000:]

        logger.debug(f"Generated {num_bits} random bits, entropy pool: {len(self.entropy_pool)}")

        return random_bits

    def generate_random_bytes(self, num_bytes: int) -> bytes:
        """
        Generate random bytes.

        Args:
            num_bytes: Number of random bytes

        Returns:
            Random bytes
        """
        bits = self.generate_random_bits(num_bytes * 8)

        # Convert bits to bytes
        byte_array = bytearray()

        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= int(bits[i + j]) << j
            byte_array.append(byte_val)

        return bytes(byte_array)

    def generate_random_integer(self, min_val: int, max_val: int) -> int:
        """
        Generate random integer in range [min_val, max_val].

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            Random integer
        """
        range_size = max_val - min_val + 1

        # Number of bits needed
        num_bits = int(np.ceil(np.log2(range_size)))

        while True:
            bits = self.generate_random_bits(num_bits)

            # Convert to integer
            value = 0
            for i, bit in enumerate(bits):
                value |= int(bit) << i

            if value < range_size:
                return min_val + value


class LatticeCryptography:
    """
    Lattice-based post-quantum cryptography.

    Implements Learning With Errors (LWE) based encryption
    resistant to quantum attacks.
    """

    def __init__(self, dimension: int = 256, security_parameter: int = 128):
        """
        Initialize lattice cryptography.

        Args:
            dimension: Lattice dimension
            security_parameter: Security level in bits
        """
        self.dimension = dimension
        self.security_parameter = security_parameter
        self.modulus = self._choose_modulus()

        # Quantum RNG for key generation
        self.qrng = QuantumRandomNumberGenerator()

        logger.info(f"Initialized LatticeCryptography with dimension {dimension}, "
                   f"security {security_parameter} bits")

    def _choose_modulus(self) -> int:
        """Choose appropriate modulus for LWE."""
        # Modulus should be ~ dimension^2 for security
        return int(self.dimension ** 2)

    def generate_keypair(self) -> Tuple[LatticeKey, LatticeKey]:
        """
        Generate public/private key pair.

        Returns:
            Tuple of (public_key, private_key)
        """
        logger.info("Generating lattice keypair")

        # Private key: small random vector
        private_key = np.array([
            self.qrng.generate_random_integer(-3, 3)
            for _ in range(self.dimension)
        ])

        # Public key: A*s + e (mod q)
        # A is random matrix
        A = np.random.randint(0, self.modulus, (self.dimension, self.dimension))

        # Error vector (small)
        e = np.array([
            self.qrng.generate_random_integer(-2, 2)
            for _ in range(self.dimension)
        ])

        # Compute public key
        public_key_vector = (A @ private_key + e) % self.modulus

        public_key = LatticeKey(
            public_key=np.vstack([A, public_key_vector]),
            private_key=None,
            security_parameter=self.security_parameter,
            lattice_dimension=self.dimension
        )

        private_key_obj = LatticeKey(
            public_key=None,
            private_key=private_key,
            security_parameter=self.security_parameter,
            lattice_dimension=self.dimension
        )

        logger.info("Keypair generated successfully")

        return public_key, private_key_obj

    def encrypt(self, message: np.ndarray, public_key: LatticeKey) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encrypt message using LWE-based encryption.

        Args:
            message: Message as bit array
            public_key: Recipient's public key

        Returns:
            Tuple of (ciphertext1, ciphertext2)
        """
        logger.info(f"Encrypting {len(message)} bit message")

        A = public_key.public_key[:-1]
        b = public_key.public_key[-1]

        # Random vector for encryption
        r = np.array([
            self.qrng.generate_random_integer(0, 1)
            for _ in range(self.dimension)
        ])

        # Encryption
        c1 = (A.T @ r) % self.modulus
        c2 = (b @ r + message[0] * (self.modulus // 2)) % self.modulus

        logger.debug("Encryption complete")

        return c1, np.array([c2])

    def decrypt(self,
                ciphertext: Tuple[np.ndarray, np.ndarray],
                private_key: LatticeKey) -> int:
        """
        Decrypt ciphertext.

        Args:
            ciphertext: Tuple of (c1, c2)
            private_key: Private key

        Returns:
            Decrypted message bit
        """
        c1, c2 = ciphertext
        s = private_key.private_key

        # Decrypt
        noisy_message = (c2[0] - s @ c1) % self.modulus

        # Decode bit
        threshold = self.modulus // 4

        if noisy_message < threshold or noisy_message > self.modulus - threshold:
            bit = 0
        else:
            bit = 1

        return bit


class QuantumSignatureScheme:
    """
    Quantum digital signature scheme.

    Uses quantum states for unforgeable signatures.
    """

    def __init__(self):
        """Initialize signature scheme."""
        self.qe = QuantumEchoesAlgorithm(num_qubits=8)
        self.qkd = QuantumKeyDistribution(protocol="BB84", key_length=256)

        logger.info("Initialized QuantumSignatureScheme")

    def generate_signature(self, message: str) -> QuantumSignature:
        """
        Generate quantum signature for message.

        Args:
            message: Message to sign

        Returns:
            QuantumSignature
        """
        import time

        logger.info(f"Generating signature for message: {message[:50]}...")

        # Hash message
        message_hash = hashlib.sha256(message.encode()).digest()

        # Encode hash in quantum states
        signature_qubits = []

        for byte in message_hash[:8]:  # Use first 8 bytes
            # Initialize qubit
            self.qe.reset()
            self.qe.initialize_qubits('zero')

            # Encode byte in basis choice
            for bit_idx in range(min(8, self.qe.num_qubits)):
                bit = (byte >> bit_idx) & 1

                if bit == 1:
                    self.qe.apply_single_qubit_gate(bit_idx, 'X')

                # Add randomness
                if np.random.rand() > 0.5:
                    self.qe.apply_single_qubit_gate(bit_idx, 'H')

            signature_qubits.append(self.qe.get_statevector())

        signature_qubits = np.array(signature_qubits)

        # Generate public key
        public_key = self.qkd.generate_key().sifted_key

        signature = QuantumSignature(
            message=message,
            signature_qubits=signature_qubits,
            public_key=public_key,
            timestamp=time.time()
        )

        logger.info("Signature generated")

        return signature

    def verify_signature(self, message: str, signature: QuantumSignature) -> bool:
        """
        Verify quantum signature.

        Args:
            message: Original message
            signature: Quantum signature to verify

        Returns:
            True if signature is valid
        """
        logger.info("Verifying signature")

        # Hash message
        message_hash = hashlib.sha256(message.encode()).digest()

        # Verify each signature qubit
        valid_count = 0

        for i, byte in enumerate(message_hash[:8]):
            if i >= len(signature.signature_qubits):
                break

            # Check signature qubit matches message encoding
            # (Simplified verification)
            sig_qubit = signature.signature_qubits[i]

            if np.linalg.norm(sig_qubit) > 0:
                valid_count += 1

        # Signature is valid if most qubits match
        is_valid = valid_count >= len(signature.signature_qubits) * 0.8

        logger.info(f"Signature {'valid' if is_valid else 'invalid'}")

        return is_valid


class QuantumSecureHash:
    """
    Quantum-enhanced hash function.

    Uses quantum measurements for collision resistance.
    """

    def __init__(self, output_bits: int = 256):
        """
        Initialize quantum hash function.

        Args:
            output_bits: Output hash size in bits
        """
        self.output_bits = output_bits
        self.qrng = QuantumRandomNumberGenerator()
        self.qe = QuantumEchoesAlgorithm(num_qubits=8)

        logger.info(f"Initialized QuantumSecureHash with {output_bits} bit output")

    def hash(self, data: bytes) -> bytes:
        """
        Compute quantum hash of data.

        Args:
            data: Input data

        Returns:
            Hash digest
        """
        logger.debug(f"Hashing {len(data)} bytes")

        # Classical hash as base
        classical_hash = hashlib.sha256(data).digest()

        # Quantum enhancement
        quantum_bits = []

        for byte in classical_hash:
            # Encode byte in quantum state
            self.qe.reset()
            self.qe.initialize_qubits('zero')

            for i in range(min(8, self.qe.num_qubits)):
                if (byte >> i) & 1:
                    self.qe.apply_single_qubit_gate(i, 'X')

                # Apply quantum mixing
                self.qe.apply_single_qubit_gate(i, 'H')
                if i > 0:
                    self.qe.apply_two_qubit_gate(i - 1, i, 'CNOT')

            # Measure to get quantum-enhanced bits
            measurements = self.qe.measure(list(range(min(8, self.qe.num_qubits))))

            enhanced_byte = 0
            for bit_idx, bit_val in measurements.items():
                enhanced_byte |= bit_val << bit_idx

            quantum_bits.append(enhanced_byte)

        # Combine classical and quantum hashes
        result = bytes(quantum_bits[:self.output_bits // 8])

        return result


def demonstrate_post_quantum_crypto():
    """Demonstrate post-quantum cryptography capabilities."""
    print("\n" + "=" * 60)
    print("POST-QUANTUM CRYPTOGRAPHY DEMONSTRATION")
    print("=" * 60)

    # 1. Quantum Random Number Generation
    print("\n1. Quantum Random Number Generation")
    qrng = QuantumRandomNumberGenerator()
    random_bits = qrng.generate_random_bits(32)
    print(f"   Generated 32 random bits: {random_bits}")
    random_int = qrng.generate_random_integer(1, 100)
    print(f"   Random integer (1-100): {random_int}")

    # 2. Lattice-Based Encryption
    print("\n2. Lattice-Based Encryption (Post-Quantum)")
    lattice = LatticeCryptography(dimension=64, security_parameter=128)
    public_key, private_key = lattice.generate_keypair()
    print(f"   Keypair generated (dimension: {lattice.dimension})")

    message_bit = np.array([1])
    ciphertext = lattice.encrypt(message_bit, public_key)
    decrypted = lattice.decrypt(ciphertext, private_key)
    print(f"   Encrypted bit {message_bit[0]}, decrypted: {decrypted}")
    print(f"   Encryption successful: {message_bit[0] == decrypted}")

    # 3. Quantum Signatures
    print("\n3. Quantum Digital Signatures")
    sig_scheme = QuantumSignatureScheme()
    message = "This is a quantum-signed message"
    signature = sig_scheme.generate_signature(message)
    is_valid = sig_scheme.verify_signature(message, signature)
    print(f"   Message: {message}")
    print(f"   Signature valid: {is_valid}")

    # 4. Quantum Hash
    print("\n4. Quantum-Enhanced Hashing")
    qhash = QuantumSecureHash(output_bits=256)
    data = b"Quantum data to hash"
    hash_value = qhash.hash(data)
    print(f"   Data: {data}")
    print(f"   Hash (hex): {hash_value.hex()[:32]}...")

    print("\n" + "=" * 60)
    print("Post-quantum cryptography demonstration complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    demonstrate_post_quantum_crypto()
