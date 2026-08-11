#!/usr/bin/env python3
"""
Quantum Key Distribution Example
=================================

Demonstrates secure key generation using BB84 and E91 protocols.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from applications.quantum_communication import (
    QuantumKeyDistribution,
    SecureQuantumChannel
)


def main():
    print("\n" + "=" * 60)
    print("QUANTUM KEY DISTRIBUTION EXAMPLE")
    print("=" * 60 + "\n")

    # BB84 Protocol
    print("1. BB84 Protocol (Prepare & Measure)")
    print("   " + "-" * 56)

    qkd_bb84 = QuantumKeyDistribution(protocol="BB84", key_length=256)

    print("   Generating 256-bit quantum key...")
    key_bb84 = qkd_bb84.generate_key()

    print(f"\n   Results:")
    print(f"   - Key length: {len(key_bb84.sifted_key)} bits")
    print(f"   - Quantum Bit Error Rate (QBER): {key_bb84.error_rate:.2%}")
    print(f"   - Security parameter: {key_bb84.security_parameter:.4f}")
    print(f"   - Qubits sent: {qkd_bb84.stats['qubits_sent']}")
    print(f"   - Errors detected: {qkd_bb84.stats['errors_detected']}")

    if key_bb84.error_rate > 0.11:
        print(f"   ⚠ WARNING: QBER exceeds security threshold!")
        print(f"   Possible eavesdropping detected!")
    else:
        print(f"   ✓ Channel secure - no eavesdropping detected")

    # E91 Protocol
    print("\n2. E91 Protocol (Entanglement-Based)")
    print("   " + "-" * 56)

    qkd_e91 = QuantumKeyDistribution(protocol="E91", key_length=256)

    print("   Generating key using entangled pairs...")
    key_e91 = qkd_e91.generate_key()

    print(f"\n   Results:")
    print(f"   - Key length: {len(key_e91.sifted_key)} bits")
    print(f"   - QBER: {key_e91.error_rate:.2%}")
    print(f"   - Security (Bell violation): {key_e91.security_parameter:.4f}")

    if key_e91.security_parameter < 0.7:
        print(f"   ⚠ WARNING: Bell inequality not strongly violated!")
    else:
        print(f"   ✓ Quantum correlations verified")

    # Secure Channel
    print("\n3. Secure Quantum Channel")
    print("   " + "-" * 56)

    channel = SecureQuantumChannel()

    print("   Establishing secure channel...")
    key = channel.establish_secure_channel()

    if key:
        print(f"   ✓ Secure channel established")
        print(f"   - Shared key length: {len(key.sifted_key)} bits")

        # Encrypt message
        message = "This is a secret quantum message!"
        print(f"\n   Encrypting message: '{message}'")

        encrypted = channel.encrypt_classical_message(message, key)
        print(f"   - Encrypted (hex): {encrypted.hex()[:40]}...")
        print(f"   - Encryption method: One-time pad (perfect secrecy)")

    print("\n4. Security Analysis")
    print("   " + "-" * 56)

    stats = channel.get_channel_stats()

    print(f"   Total qubits transmitted: {stats['qubits_sent']}")
    print(f"   Errors detected: {stats['errors_detected']}")
    print(f"   Eavesdropping attempts: {stats['eavesdropping_attempts']}")

    print("\n   Security guarantees:")
    print("   - Information-theoretic security")
    print("   - Unconditional security against eavesdropping")
    print("   - Eavesdropping detection via quantum mechanics")

    print("\n" + "=" * 60)
    print("QKD example complete!")
    print("\nApplications:")
    print("  - Secure government communications")
    print("  - Banking and financial transactions")
    print("  - Military and defense networks")
    print("  - Critical infrastructure protection")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
