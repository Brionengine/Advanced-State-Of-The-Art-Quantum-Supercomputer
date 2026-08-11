"""
Quantum OS Security Module

Provides code protection and security features
"""

from .obfuscator import CodeObfuscator

# NOTE: this package previously also exported `SecurityManager` from a `.auth`
# module that does not exist in the tree, which made importing anything under
# quantum_os fail. Nothing referenced the name; if an auth layer is added later,
# re-export it here.
__all__ = ['CodeObfuscator']
