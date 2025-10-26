"""
Quantum OS Plugin System

Allows importing and integrating existing quantum modules
"""

from .loader import PluginLoader
from .registry import PluginRegistry

__all__ = ['PluginLoader', 'PluginRegistry']
