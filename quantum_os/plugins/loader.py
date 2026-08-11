"""
Plugin Loader

Dynamically loads quantum algorithms and modules from existing projects
"""

import os
import sys
import importlib.util
from typing import Dict, Any, List, Optional
from pathlib import Path


class PluginLoader:
    """
    Plugin loader for quantum algorithms

    Enables importing algorithms from existing L.L.M.A and other quantum projects
    """

    def __init__(self):
        """Initialize plugin loader"""
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_paths: List[str] = []

    def add_plugin_path(self, path: str):
        """
        Add a directory to search for plugins

        Args:
            path: Directory path
        """
        if os.path.isdir(path) and path not in self.plugin_paths:
            self.plugin_paths.append(path)
            # Add to Python path
            if path not in sys.path:
                sys.path.insert(0, path)

    def load_plugin(
        self,
        plugin_name: str,
        plugin_path: Optional[str] = None
    ) -> Any:
        """
        Load a plugin module

        Args:
            plugin_name: Name of the plugin
            plugin_path: Specific path to plugin file

        Returns:
            Loaded plugin module
        """
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name]

        try:
            if plugin_path:
                # Load from specific path
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.loaded_plugins[plugin_name] = module
                    return module
            else:
                # Try to import normally
                module = importlib.import_module(plugin_name)
                self.loaded_plugins[plugin_name] = module
                return module

        except Exception as e:
            print(f"Error loading plugin '{plugin_name}': {e}")
            return None

    def load_llma_algorithms(self, llma_path: str) -> Dict[str, Any]:
        """
        Load quantum algorithms from L.L.M.A project

        Args:
            llma_path: Path to L.L.M.A project directory

        Returns:
            Dictionary of loaded algorithms
        """
        algorithms = {}

        if not os.path.isdir(llma_path):
            print(f"L.L.M.A path not found: {llma_path}")
            return algorithms

        self.add_plugin_path(llma_path)

        # Common algorithm files in L.L.M.A
        algorithm_files = [
            'quantum_algorithms.py',
            'quantum_error_correction_bit_flip.py',
            'quantum_fusion_core.py',
            'vqe_energy_minimization.py',
        ]

        for algo_file in algorithm_files:
            # Search for file
            for root, dirs, files in os.walk(llma_path):
                if algo_file in files:
                    full_path = os.path.join(root, algo_file)
                    module_name = algo_file.replace('.py', '')

                    module = self.load_plugin(module_name, full_path)
                    if module:
                        algorithms[module_name] = module
                    break

        return algorithms

    def get_plugin_functions(self, plugin_name: str) -> List[str]:
        """
        Get list of functions in a plugin

        Args:
            plugin_name: Plugin name

        Returns:
            List of function names
        """
        if plugin_name not in self.loaded_plugins:
            return []

        module = self.loaded_plugins[plugin_name]
        return [
            name for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith('_')
        ]

    def get_plugin_classes(self, plugin_name: str) -> List[str]:
        """
        Get list of classes in a plugin

        Args:
            plugin_name: Plugin name

        Returns:
            List of class names
        """
        if plugin_name not in self.loaded_plugins:
            return []

        module = self.loaded_plugins[plugin_name]
        import inspect
        return [
            name for name, obj in inspect.getmembers(module)
            if inspect.isclass(obj) and obj.__module__ == module.__name__
        ]

    def list_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return list(self.loaded_plugins.keys())
