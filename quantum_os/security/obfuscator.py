"""
Code Obfuscation

Provides code obfuscation to protect intellectual property
"""

import os
import subprocess
from typing import List, Optional
from pathlib import Path


class CodeObfuscator:
    """
    Code obfuscator for protecting quantum algorithms

    Uses PyArmor for Python code obfuscation
    """

    def __init__(self, obfuscation_level: int = 2):
        """
        Initialize obfuscator

        Args:
            obfuscation_level: 0=none, 1=light, 2=medium, 3=heavy
        """
        self.obfuscation_level = obfuscation_level

    def obfuscate_file(
        self,
        source_file: str,
        output_dir: str,
        license_key: Optional[str] = None
    ) -> bool:
        """
        Obfuscate a Python file

        Args:
            source_file: Source Python file
            output_dir: Output directory
            license_key: Optional license key for runtime validation

        Returns:
            bool: True if successful
        """
        if self.obfuscation_level == 0:
            # No obfuscation
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(source_file, output_dir)
            return True

        try:
            # Use PyArmor for obfuscation
            cmd = ['pyarmor', 'gen']

            # Set obfuscation mode based on level
            if self.obfuscation_level >= 2:
                cmd.extend(['-O', 'rft'])  # Restrict mode
            if self.obfuscation_level >= 3:
                cmd.extend(['--private'])  # Private mode

            # Add output directory
            cmd.extend(['-O', output_dir])

            # Add source file
            cmd.append(source_file)

            # Run obfuscation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"Obfuscated: {source_file} -> {output_dir}")
                return True
            else:
                print(f"Obfuscation failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error during obfuscation: {e}")
            return False

    def obfuscate_directory(
        self,
        source_dir: str,
        output_dir: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> bool:
        """
        Obfuscate all Python files in a directory

        Args:
            source_dir: Source directory
            output_dir: Output directory
            exclude_patterns: Patterns to exclude

        Returns:
            bool: True if successful
        """
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', '.pytest_cache', 'tests']

        success = True

        for root, dirs, files in os.walk(source_dir):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns]

            for file in files:
                if file.endswith('.py'):
                    source_path = os.path.join(root, file)
                    rel_path = os.path.relpath(source_path, source_dir)
                    output_path = os.path.join(output_dir, os.path.dirname(rel_path))

                    if not self.obfuscate_file(source_path, output_path):
                        success = False

        return success

    def create_license(
        self,
        output_file: str,
        expire_date: Optional[str] = None,
        bind_data: Optional[str] = None
    ) -> bool:
        """
        Create a runtime license for obfuscated code

        Args:
            output_file: License file path
            expire_date: Expiration date (YYYY-MM-DD)
            bind_data: Hardware binding data (MAC address, etc.)

        Returns:
            bool: True if successful
        """
        try:
            cmd = ['pyarmor', 'gen', 'key']

            if expire_date:
                cmd.extend(['-e', expire_date])

            if bind_data:
                cmd.extend(['-b', bind_data])

            cmd.extend(['-O', output_file])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            return result.returncode == 0

        except Exception as e:
            print(f"Error creating license: {e}")
            return False

    @staticmethod
    def is_pyarmor_available() -> bool:
        """Check if PyArmor is installed"""
        try:
            result = subprocess.run(
                ['pyarmor', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
