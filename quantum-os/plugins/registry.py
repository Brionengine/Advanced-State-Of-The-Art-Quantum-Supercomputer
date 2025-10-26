"""
Plugin Registry

Maintains registry of quantum algorithms and their metadata
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    description: str
    author: str = "Unknown"
    category: str = "general"
    requires: list = field(default_factory=list)


class PluginRegistry:
    """
    Registry for quantum algorithm plugins

    Allows registration and discovery of quantum algorithms
    """

    def __init__(self):
        """Initialize plugin registry"""
        self.plugins: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, list] = {}

    def register(
        self,
        name: str,
        plugin_callable: Callable,
        metadata: Optional[PluginMetadata] = None
    ):
        """
        Register a plugin

        Args:
            name: Plugin name
            plugin_callable: Callable (function or class)
            metadata: Plugin metadata
        """
        if metadata is None:
            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                description="No description provided"
            )

        self.plugins[name] = {
            'callable': plugin_callable,
            'metadata': metadata
        }

        # Add to category
        category = metadata.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)

    def get(self, name: str) -> Optional[Callable]:
        """Get a registered plugin"""
        if name in self.plugins:
            return self.plugins[name]['callable']
        return None

    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata"""
        if name in self.plugins:
            return self.plugins[name]['metadata']
        return None

    def list_plugins(self, category: Optional[str] = None) -> list:
        """
        List registered plugins

        Args:
            category: Filter by category (None for all)

        Returns:
            List of plugin names
        """
        if category:
            return self.categories.get(category, [])
        return list(self.plugins.keys())

    def list_categories(self) -> list:
        """List all categories"""
        return list(self.categories.keys())
