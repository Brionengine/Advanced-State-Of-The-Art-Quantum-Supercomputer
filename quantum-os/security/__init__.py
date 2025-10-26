"""
Quantum OS Security Module

Provides code protection and security features
"""

from .obfuscator import CodeObfuscator
from .auth import SecurityManager

__all__ = ['CodeObfuscator', 'SecurityManager']
