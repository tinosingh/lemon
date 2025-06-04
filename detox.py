#!/usr/bin/env python3
"""
🍋 Dependency Detox - Python Environment Cleanse Tool 🍋

Give your Python environments a refreshing cleanse!

Analyzes Python environments to identify redundancies, conflicts, and opportunities
to consolidate similar virtual environments for clarity and resource efficiency
and suggests optimizations using UV.

Usage:
    detox [venv_path1] [venv_path2] ...
    detox --search-dir /path/to/search
    detox --use-uv --full-detox

Author: The Lemon Squad / Tino Singh
License: MIT
"""

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

# Attempt to import the packaging library for robust version and specifier handling
try:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, parse as parse_version

    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False

    # Provide dummy classes if 'packaging' is not available to avoid runtime errors
    # in type hints and allow basic fallback functionality.
    class _BaseFallbackException(Exception):
        pass

    class InvalidVersion(_BaseFallbackException):  # type: ignore
        pass

    class InvalidSpecifier(_BaseFallbackException):  # type: ignore
        pass

    def parse_version(version_string: str) -> Any:  # type: ignore
        # Simplified parser for fallback, less robust
        match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version_string)
        if match:
            parts = [int(p) for p in match.groups() if p is not None]
            return tuple(parts)
        raise InvalidVersion(f"Cannot parse version: {version_string}")

    class SpecifierSet:  # type: ignore
        def __init__(self, specifiers: str = ""):
            self.specifiers_str = specifiers
            # Extremely simplified specifier parsing for fallback
            self.parsed_specifiers = []
            if specifiers:
                for spec in specifiers.split(","):
                    spec = spec.strip()
                    match = re.match(r"([<>=!~]+)\s*(\d+\.\d+(?:\.\d+)?)", spec)
                    if match:
                        op, ver_str = match.groups()
                        try:
                            ver = parse_version(ver_str)
                            self.parsed_specifiers.append((op, ver))
                        except InvalidVersion:
                            pass  # Ignore unparseable specifiers in fallback

        def __contains__(self, version_str: str) -> bool:
            if not self.parsed_specifiers:  # No specifiers, assume compatible
                return True
            try:
                item_version = parse_version(version_str)
                for op, req_ver in self.parsed_specifiers:
                    if op == ">=":
                        return item_version >= req_ver
                    if op == "<=":
                        return item_version <= req_ver
                    if op == ">":
                        return item_version > req_ver
                    if op == "<":
                        return item_version < req_ver
                    if op == "==":
                        return item_version == req_ver
                    if op == "!=":
                        return item_version != req_ver
                    # Add more operators if needed for fallback (e.g., ~=)
                return True  # If specific checks pass or no restrictive specifiers
            except InvalidVersion:
                return False  # Cannot compare if version is invalid
            return False


# --- Configuration & Constants ---
# Using a function to initialize logger to allow level changes more easily if needed elsewhere
def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Initializes and returns a logger instance."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],  # Explicitly use stdout
    )
    return logging.getLogger("DependencyDetox")


logger = setup_logger()

LEMON = "🍋"
PYTHON_VERSIONS_TO_CHECK = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]
CONSOLIDATION_THRESHOLD = 0.8  # Jaccard similarity

# These could be loaded from a config file (e.g., pyproject.toml, JSON)
# for more flexibility
IMPORTANT_PACKAGES_FOR_PYTHON_VERSION_CHECK: Set[str] = {
    "numpy",
    "pandas",
    "scipy",
    "tensorflow",
    "torch",
    "django",
    "flask",
    "requests",
    "matplotlib",
    "scikit-learn",
    "fastapi",
    "httpx",
    "pydantic",
    "sqlalchemy",  # Added some more modern/common ones
}
DEFAULT_RECOMMENDED_PYTHON_VERSION = "3.11"  # Fallback if no requirements found


# --- Data Models ---
@dataclass
class PackageInfo:
    """Information about a Python package."""

    name: str
    version: str
    raw_version: str  # Store the original version string for accurate display
    parsed_version: Any = field(
        default=None
    )  # Stores result of packaging.version.parse
    python_requires: Optional[str] = None
    dependencies: List[str] = field(
        default_factory=list
    )  # Placeholder for future dep analysis

    def __post_init__(self):
        try:
            self.parsed_version = parse_version(self.version)
        except InvalidVersion:
            logger.debug(f"Could not parse version for {self.name}: {self.version}")
            self.parsed_version = self.version  # Fallback to raw string if unparseable


@dataclass
class ConflictInfo:
    """Information about a dependency conflict."""

    package: str
    env1_name: str
    env1_version: str
    env2_name: str
    env2_version: str
    severity: str  # 'major', 'minor', 'patch', 'other' (for non-semver or complex diffs)


@dataclass
class PythonVersionRecommendation:
    """Recommendation for optimal Python version."""

    recommended_version: str
    compatible_packages_count: int
    total_analyzed_package_requirements: int
    incompatible_packages: List[Tuple[str, str]]  # (package_name, python_requires_spec)
    warnings: List[str]


# --- Core Components ---


class SubprocessRunner:
    """Helper for running subprocess commands consistently."""

    @staticmethod
    def run(
        cmd: List[str], timeout: int = 30, suppress_output: bool = False, **kwargs: Any
    ) -> subprocess.CompletedProcess:
        """Runs a subprocess command and handles common exceptions."""
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                errors="ignore",  # Handles potential decoding errors in output
                **kwargs,
            )
            if not suppress_output and process.stdout.strip():
                logger.debug(f"CMD Output ({' '.join(cmd)}):\n{process.stdout.strip()}")
            if process.stderr.strip():
                logger.debug(f"CMD Stderr ({' '.join(cmd)}):\n{process.stderr.strip()}")
            return process
        except subprocess.TimeoutExpired:
            logger.error(f"  ⏳ Command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(
                f"  ❌ Command failed: {' '.join(cmd)}\n"
                f"     Return code: {e.returncode}\n"
                f"     Stderr: {e.stderr.strip() if e.stderr else 'N/A'}\n"
                f"     Stdout: {e.stdout.strip() if e.stdout else 'N/A'}"
            )
            raise
        except FileNotFoundError:
            logger.error(f"  🔍 Command not found: {cmd[0]}. Is it installed and in PATH?")
            raise
        except Exception as e:  # Catch other potential OS or permission errors
            logger.error(
                f"  ⚠️ Unexpected error running command {' '.join(cmd)}: "
                f"{type(e).__name__}: {e}"
            )
            raise


class PythonEnvironment:
    """Represents a Python environment with its metadata."""

    def __init__(self, path_str: str, name: str, is_global: bool = False):
        self.path = Path(path_str).resolve()  # Resolve to absolute path
        self.name = name
        self.is_global = is_global
        self._python_version: Optional[str] = None
        self.packages: Dict[str, PackageInfo] = {}
        self._python_executable: Optional[Path] = None

    @property
    def python_executable(self) -> Path:
        """Get the Python executable path for this environment. Lazily determined."""
        if self._python_executable is None:
            if self.is_global:
                exe_path = Path(sys.executable)  # Current interpreter for global
            else:
                # Standard venv paths
                script_dir = "Scripts" if platform.system() == "Windows" else "bin"
                exe_path = self.path / script_dir / (
                    "python.exe" if platform.system() == "Windows" else "python"
                )

            # Verify executable exists and is a file
            if exe_path.exists() and exe_path.is_file():
                self._python_executable = exe_path
            else:
                # Fallback for less standard layouts, e.g. some conda envs might
                # place it directly in env root
                fallback_exe_path = self.path / (
                    "python.exe" if platform.system() == "Windows" else "python"
                )
                if fallback_exe_path.exists() and fallback_exe_path.is_file():
                    self._python_executable = fallback_exe_path
                else:
                    logger.warning(
                        f"  ⚠️ Python executable not found at expected paths for "
                        f"{self.name} ({self.path})."
                    )
                    # Return a dummy path to avoid None propagation issues,
                    # subsequent calls will fail gracefully.
                    self._python_executable = self.path / "nonexistent_python"
        return self._python_executable

    def is_valid(self) -> bool:
        """Check if this is a valid Python environment."""
        if self.is_global:
            # Global is valid if current interpreter is found
            return self.python_executable.exists()
        # For venvs, check for pyvenv.cfg and a Python executable
        return (self.path / "pyvenv.cfg").exists() and self.python_executable.exists()

    def get_python_version(self) -> Optional[str]:
        """Determine the Python version of the environment. Lazily evaluated."""
        if self._python_version is None and self.python_executable.exists():
            try:
                result = SubprocessRunner.run(
                    [str(self.python_executable), "--version"], timeout=5
                )
                match = re.search(r"Python (\d+\.\d+\.\d+)", result.stdout)
                if match:
                    self._python_version = match.group(1)
                else:
                    logger.debug(f"Could not parse Python version from: {result.stdout}")
            except (subprocess.SubprocessError, OSError, ValueError, FileNotFoundError) as e:
                logger.warning(
                    f"  ⚠️ Could not determine Python version for {self.name} at "
                    f"{self.path}: {e}"
                )
        return self._python_version


class EnvironmentDiscovery:
    """Handles discovery of Python environments."""

    @staticmethod
    def find_global_environment_path() -> Optional[str]:
        """Find the path to the directory of the global Python installation."""
        logger.info(f"{LEMON} Locating your main Python environment...")
        try:
            # sys.prefix is usually reliable for the currently running Python
            global_env_path = Path(sys.prefix)
            if (global_env_path / "pyvenv.cfg").exists():  # This is a venv, not global
                logger.debug(f"  sys.prefix points to a venv: {global_env_path}. " "Trying sysconfig.")
                # Fallback to sysconfig which might point to system install
                stdlib_path = sysconfig.get_path("stdlib")
                if stdlib_path:
                    # Go up to site-packages, then to pythonX.Y folder
                    python_base = Path(stdlib_path).parent.parent
                    logger.info(f"  ✅ Found system Python via sysconfig: {python_base}")
                    return str(python_base)
            else:
                logger.info(
                    f"  ✅ Found current Python (assumed global or base) via "
                    f"sys.prefix: {global_env_path}"
                )
                return str(global_env_path)

        except Exception as exc:
            logger.debug(f"  Error using sys.prefix or sysconfig: {exc}")

        # Fallback if sys.prefix was a venv and sysconfig failed
        cmd = ["where", "python"] if platform.system() == "Windows" else ["which", "python3"]  # Prefer python3
        try:
            result = SubprocessRunner.run(cmd, timeout=5)
            # Get the first line, resolve symlinks, and get the directory
            python_exe_path = Path(result.stdout.strip().splitlines()[0]).resolve()
            python_dir = python_exe_path.parent.parent  # executable is in bin/ or Scripts/
            logger.info(f"  ✅ Found via {cmd[0]}: {python_dir}")
            return str(python_dir)
        except (subprocess.SubprocessError, OSError, FileNotFoundError, IndexError) as exc:
            logger.debug(f"  {cmd[0]} command failed or no output: {exc}")

        logger.warning("  ⚠️ Could not reliably locate a global/system Python environment path.")
        return None

    @staticmethod
    def find_virtual_environments(search_dir_str: str) -> List[str]:
        """Search for virtual environments in a directory."""
        search_path = Path(search_dir_str).resolve()
        logger.info(f"\n{LEMON} Searching for environments in: {search_path}")
        venvs: List[str] = []

        if not search_path.is_dir():
            logger.warning(f"  ⚠️ Search path is not a directory or does not exist: {search_path}")
            return venvs

        try:
            # Look for 'pyvenv.cfg' which is standard for venv and virtualenv
            for pyvenv_file in search_path.rglob("pyvenv.cfg"):
                venv_dir = pyvenv_file.parent
                # Basic check to ensure it looks like an environment structure
                bin_dir = "Scripts" if platform.system() == "Windows" else "bin"
                if (venv_dir / bin_dir / "python").exists() or (
                    venv_dir / bin_dir / "python.exe"
                ).exists():
                    venvs.append(str(venv_dir))
                    logger.info(f"  🔍 Found potential environment: {venv_dir.name} at {venv_dir}")
                else:
                    logger.debug(
                        f"  Skipping {venv_dir}: missing Python executable in "
                        "expected bin/Scripts folder."
                    )
        except PermissionError:
            logger.warning(f"  ⚠️ Permission denied when searching in: {search_path}")
        except Exception as exc:  # Catch other potential errors during file system traversal
            logger.error(f"  ⚠️ Error during environment search in {search_path}: {exc}")
        return venvs


class PackageExtractor(Protocol):
    """Protocol for package extraction strategies."""

    def extract(self, env: PythonEnvironment) -> Dict[str, PackageInfo]:
        """Extract packages from an environment."""
        ...

    def _enrich_package_metadata(
        self, packages: Dict[str, PackageInfo], env: PythonEnvironment
    ) -> None:
        """Enrich package info with python_requires for important packages.
        (Optional based on tool)
        """
        pass  # Default no-op


class BasePackageExtractor:
    """Base class for common package extraction logic."""

    def _parse_package_list_json(self, json_output: str, env_name: str) -> Dict[str, PackageInfo]:
        packages: Dict[str, PackageInfo] = {}
        try:
            raw_packages = json.loads(json_output)
            for pkg_data in raw_packages:
                name = pkg_data.get("name", "").lower()
                version = pkg_data.get("version", "0.0.0")  # Default if missing
                if name:
                    packages[name] = PackageInfo(
                        name=name, version=version, raw_version=version
                    )
            logger.info(f"  ✅ Found {len(packages)} packages in {env_name}!")
        except json.JSONDecodeError:
            logger.error(
                f"  ⚠️ Invalid JSON from package listing for {env_name}. "
                f"Raw output snippet: {json_output[:200]}"
            )
        return packages

    def _get_python_requires(
        self, package_name: str, python_exe: str, tool_cmd: List[str]
    ) -> Optional[str]:
        """Helper to get 'Requires-Python' using pip or uv show."""
        try:
            # SubprocessRunner is used by the calling extract method, so direct call here
            result = subprocess.run(
                tool_cmd + [package_name],  # e.g., [python, -m, pip, show] or [uv, pip, show]
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
                errors="ignore",
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.lower().startswith("requires-python:"):
                        return line.split(":", 1)[1].strip()
        except (subprocess.SubprocessError, OSError):
            logger.debug(
                f"Failed to get 'Requires-Python' for {package_name} using "
                f"{' '.join(tool_cmd)}"
            )
        return None


class PipExtractor(BasePackageExtractor, PackageExtractor):
    """Extracts packages using pip."""

    def extract(self, env: PythonEnvironment) -> Dict[str, PackageInfo]:
        logger.info(f"\n{LEMON} Extracting packages from {env.name} (using pip)...")
        if not env.python_executable.exists():
            logger.error(f"  ⚠️ Python executable not found for {env.name}: {env.python_executable}")
            return {}

        pip_cmd_base = [str(env.python_executable), "-m", "pip"]
        try:
            # Removed --not-required to get a complete list of installed packages
            result = SubprocessRunner.run(pip_cmd_base + ["list", "--format=json"], timeout=60)
            packages = self._parse_package_list_json(result.stdout, env.name)
            if packages:
                self._enrich_package_metadata(packages, env)
            return packages
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            logger.error(
                f"  ⚠️ Pip extraction failed for {env.name}. "
                "Check pip installation in this environment."
            )
            return {}

    def _enrich_package_metadata(self, packages: Dict[str, PackageInfo], env: PythonEnvironment) -> None:
        """Enrich with python_requires using pip show."""
        logger.info(f"  🔍 Enriching metadata for important packages in {env.name} (pip)...")
        pip_cmd_base = [str(env.python_executable), "-m", "pip", "show"]
        for pkg_name in IMPORTANT_PACKAGES_FOR_PYTHON_VERSION_CHECK:
            if pkg_name in packages:
                python_req = self._get_python_requires(
                    pkg_name, str(env.python_executable), pip_cmd_base
                )
                if python_req:
                    packages[pkg_name].python_requires = python_req


class UvExtractor(BasePackageExtractor, PackageExtractor):
    """Extracts packages using UV."""

    def extract(self, env: PythonEnvironment) -> Dict[str, PackageInfo]:
        logger.info(f"\n{LEMON} Extracting packages from {env.name} (UV Turbo Mode)...")
        if not env.python_executable.exists() and not env.is_global:
            logger.error(
                f"  ⚠️ Python executable not found for {env.name}: {env.python_executable}. "
                "UV might still try global."
            )
            # Allow UV to try, it might use the active env if one is active where detox runs

        uv_cmd_base = ["uv", "pip", "list", "--format", "json"]
        # If the environment's python executable is known, explicitly tell UV which Python to inspect
        if env.python_executable.exists():
            uv_cmd_base.extend(["--python", str(env.python_executable)])

        try:
            result = SubprocessRunner.run(uv_cmd_base, timeout=45)
            packages = self._parse_package_list_json(result.stdout, env.name)
            if packages:
                self._enrich_package_metadata(packages, env)
            return packages
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            logger.error(f"  ⚠️ UV extraction failed for {env.name}. Is UV installed and in PATH?")
            return {}

    def _enrich_package_metadata(self, packages: Dict[str, PackageInfo], env: PythonEnvironment) -> None:
        """Enrich with python_requires using uv pip show."""
        logger.info(f"  🔍 Enriching metadata for important packages in {env.name} (UV)...")

        uv_show_cmd_base = ["uv", "pip", "show"]
        # Provide context if the environment's python executable is known
        if env.python_executable.exists():
            uv_show_cmd_base.extend(["--python", str(env.python_executable)])

        for pkg_name in IMPORTANT_PACKAGES_FOR_PYTHON_VERSION_CHECK:
            if pkg_name in packages:
                python_req = self._get_python_requires(
                    pkg_name, str(env.python_executable), uv_show_cmd_base
                )
                if python_req:
                    packages[pkg_name].python_requires = python_req

    @staticmethod
    def is_available() -> bool:
        """Checks if UV is installed and accessible."""
        try:
            result = SubprocessRunner.run(["uv", "--version"], timeout=5, suppress_output=True)
            version = result.stdout.strip()
            logger.info(f"{LEMON} UV Turbo Mode: AVAILABLE! ({version})")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("UV not found or --version command failed.")
            return False


class ConflictAnalyzer:
    """Analyzes dependency conflicts between environments."""

    @staticmethod
    def detect_conflicts(environments: List[PythonEnvironment]) -> List[ConflictInfo]:
        """Detects version conflicts for shared packages across environments."""
        conflicts: List[ConflictInfo] = []
        unique_environments = [env for env in environments if env.packages]  # Only consider envs with packages

        for i in range(len(unique_environments)):
            for j in range(i + 1, len(unique_environments)):
                env1, env2 = unique_environments[i], unique_environments[j]
                common_packages = set(env1.packages.keys()) & set(env2.packages.keys())

                for pkg_name in common_packages:
                    pkg_info1 = env1.packages[pkg_name]
                    pkg_info2 = env2.packages[pkg_name]

                    # Compare parsed versions if available and valid, otherwise raw strings
                    are_versions_different = False
                    if (
                        PACKAGING_AVAILABLE
                        and pkg_info1.parsed_version
                        and pkg_info2.parsed_version
                        and not isinstance(pkg_info1.parsed_version, str)
                        and not isinstance(pkg_info2.parsed_version, str)
                    ):
                        if pkg_info1.parsed_version != pkg_info2.parsed_version:
                            are_versions_different = True
                    elif pkg_info1.version != pkg_info2.version:  # Fallback to string comparison
                        are_versions_different = True

                    if are_versions_different:
                        severity = ConflictAnalyzer._get_severity(pkg_info1, pkg_info2)
                        conflicts.append(
                            ConflictInfo(
                                package=pkg_name,
                                env1_name=env1.name,
                                env1_version=pkg_info1.raw_version,  # Display raw version
                                env2_name=env2.name,
                                env2_version=pkg_info2.raw_version,  # Display raw version
                                severity=severity,
                            )
                        )
        return conflicts

    @staticmethod
    def _get_severity(pkg_info1: PackageInfo, pkg_info2: PackageInfo) -> str:
        """Determines the severity of a version conflict."""
        if (
            not PACKAGING_AVAILABLE
            or isinstance(pkg_info1.parsed_version, str)
            or isinstance(pkg_info2.parsed_version, str)
        ):
            # Fallback if packaging is not available or versions weren't parsed
            # (e.g. "1.2.3.dev0")
            # Simple string comparison for major/minor/patch if format is X.Y.Z
            v1_parts_str = pkg_info1.version.split(".")
            v2_parts_str = pkg_info2.version.split(".")
            try:
                v1_p = [int(p) for p in v1_parts_str[:3]]
                v2_p = [int(p) for p in v2_parts_str[:3]]
                # Pad with zeros if necessary
                v1_p.extend([0] * (3 - len(v1_p)))
                v2_p.extend([0] * (3 - len(v2_p)))

                if v1_p[0] != v2_p[0]:
                    return "major"
                if v1_p[1] != v2_p[1]:
                    return "minor"
                if v1_p[2] != v2_p[2]:
                    return "patch"
                return "other"  # Same prefix, but strings differ (e.g. post release)
            except ValueError:
                return "other"  # Non-integer parts

        # Use packaging.version for robust comparison
        v1 = pkg_info1.parsed_version
        v2 = pkg_info2.parsed_version

        if v1.epoch != v2.epoch or v1.release[0] != v2.release[0]:  # Major version (or epoch) differs
            return "major"
        if len(v1.release) < 2 or len(v2.release) < 2 or v1.release[1] != v2.release[1]:  # Minor version differs
            return "minor"
        if len(v1.release) < 3 or len(v2.release) < 3 or v1.release[2] != v2.release[2]:  # Patch version differs
            return "patch"
        # Covers pre-releases, post-releases, dev releases if major.minor.patch are same
        return "other"


class PythonVersionAnalyzer:
    """Analyzes Python version compatibility."""

    @staticmethod
    def get_recommendation(environments: List[PythonEnvironment]) -> PythonVersionRecommendation:
        """Recommends an optimal Python version based on package requirements."""
        all_pkg_python_requirements: List[Tuple[str, str]] = []
        for env in environments:
            for pkg in env.packages.values():
                if pkg.python_requires:  # e.g., ">=3.7, <3.12"
                    all_pkg_python_requirements.append((pkg.name, pkg.python_requires))

        if not all_pkg_python_requirements:
            return PythonVersionRecommendation(
                recommended_version=DEFAULT_RECOMMENDED_PYTHON_VERSION,
                compatible_packages_count=0,  # No requirements to check against
                total_analyzed_package_requirements=0,
                incompatible_packages=[],
                warnings=[
                    "No explicit Python version requirements found for important packages. "
                    f"Suggesting default: {DEFAULT_RECOMMENDED_PYTHON_VERSION}."
                ],
            )

        if not PACKAGING_AVAILABLE:
            logger.warning(
                "⚠️ 'packaging' library not found. Python version compatibility checks "
                "will be basic and may be less accurate."
            )

        compatibility_scores: Dict[str, Tuple[int, List[Tuple[str, str]]]] = {}
        target_python_versions = PYTHON_VERSIONS_TO_CHECK

        for target_py_ver_str in target_python_versions:  # e.g., "3.10"
            compatible_count = 0
            incompatible_list: List[Tuple[str, str]] = []  # (package_name, requirement_spec)

            # We use the major.minor for testing against specifiers like ">=3.9"
            # For specifiers like "==3.9.5", our target_py_ver_str "3.9" would still be checked.
            # A full compatibility would involve specific patch versions, but this is a good
            # heuristic. `packaging` library handles this well.

            for pkg_name, req_spec_str in all_pkg_python_requirements:
                if PythonVersionAnalyzer._is_compatible(target_py_ver_str, req_spec_str):
                    compatible_count += 1
                else:
                    incompatible_list.append((pkg_name, req_spec_str))
            compatibility_scores[target_py_ver_str] = (compatible_count, incompatible_list)

        # Find the version that maximizes compatible packages.
        # If ties, prefer newer versions.
        best_version = DEFAULT_RECOMMENDED_PYTHON_VERSION  # Default
        max_compatible = -1

        for py_ver in reversed(target_python_versions):  # Prefer newer versions in case of tie
            if compatibility_scores[py_ver][0] >= max_compatible:
                max_compatible = compatibility_scores[py_ver][0]
                best_version = py_ver

        compatible_count_for_best, incompatible_for_best = compatibility_scores[best_version]
        warnings = []
        if incompatible_for_best:
            warnings.append(
                f"Some packages have Python requirements not met by {best_version}."
            )
        if not PACKAGING_AVAILABLE:
            warnings.append("Results may be less accurate as 'packaging' library is not installed.")

        return PythonVersionRecommendation(
            recommended_version=best_version,
            compatible_packages_count=compatible_count_for_best,
            total_analyzed_package_requirements=len(all_pkg_python_requirements),
            incompatible_packages=incompatible_for_best,
            warnings=warnings,
        )

    @staticmethod
    def _is_compatible(python_version_str: str, requirement_spec_str: str) -> bool:
        """Checks if a given Python version satisfies a requirement string."""
        if not PACKAGING_AVAILABLE:  # Fallback if 'packaging' is not available
            # Very basic check for fallback, e.g. ">=3.7"
            match = re.search(r"([<>=!~]+)\s*(\d+\.\d+(?:\.\d+)?)", requirement_spec_str)
            if not match:
                return True  # No parsable specifier, assume compatible
            op, req_ver_str = match.groups()
            try:
                # Compare major.minor tuples
                py_ver_tuple = tuple(map(int, python_version_str.split(".")[:2]))
                req_ver_tuple = tuple(map(int, req_ver_str.split(".")[:2]))

                if op == ">=":
                    return py_ver_tuple >= req_ver_tuple
                if op == "<=":
                    return py_ver_tuple <= req_ver_tuple
                if op == ">":
                    return py_ver_tuple > req_ver_tuple
                if op == "<":
                    return py_ver_tuple < req_ver_tuple
                if op == "==":
                    return py_ver_tuple == req_ver_tuple
                # More ops (e.g. !=, ~=) could be added for fallback
                return True  # Default to compatible if operator is not simple
            except ValueError:
                return False  # Cannot parse version parts

        try:
            # Create a SpecifierSet from the requirement string (e.g., ">=3.7, <3.12")
            spec_set = SpecifierSet(requirement_spec_str)
            # Check if the target Python version (e.g., "3.10") is in the specifier set.
            # We use major.minor of the target python version for checking broader compatibility.
            # For exact matches like "==3.9.5", `packaging` handles it if `python_version_str`
            # is "3.9.5". Here, `python_version_str` is "3.8", "3.9", etc. so it checks if that
            # series is allowed.
            return python_version_str in spec_set
        except InvalidSpecifier:
            logger.debug(f"Invalid Python requirement specifier: '{requirement_spec_str}'")
            return False  # Treat invalid specifiers as incompatible to be safe
        except Exception as e:  # Catch any other error from packaging lib
            logger.debug(
                f"Error checking compatibility for Py {python_version_str} with spec "
                f"'{requirement_spec_str}': {e}"
            )
            return False


class ReportGenerator:
    """Generates analysis reports for Dependency Detox."""

    def __init__(self, use_uv_flag: bool = False):
        self.use_uv_flag = use_uv_flag

    def print_header(self) -> None:
        """Prints the main header of the report."""
        logger.info("\n" + "=" * 80)
        logger.info(f"  {LEMON} DEPENDENCY DETOX RESULTS {LEMON}".center(80))
        logger.info("=" * 80)
        logger.info("\n⚠️  REMEMBER: We only diagnose, never operate!")
        logger.info("    Your packages are safe - this is just a health check.\n")
        mode = "UV Turbo" if self.use_uv_flag else "Standard pip"
        logger.info(f"🔧 Analysis Mode: {mode}")
        if not PACKAGING_AVAILABLE and not self.use_uv_flag:  # UV might have its own robust parsing
            logger.warning(
                "  ❗ 'packaging' library not installed. Some version comparisons "
                "might be less accurate."
            )

    def print_environments_summary(self, environments: List[PythonEnvironment]) -> None:
        """Prints a summary of all analyzed environments."""
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 Environments Analyzed ({len(environments)}):")
        logger.info("=" * 80)
        if not environments:
            logger.info("  No environments were successfully analyzed.")
            return
        for env in environments:
            icon = "🏠 Global" if env.is_global else "📦 Virtual"
            version_str = env.get_python_version() or "Unknown Python"
            pkg_count = len(env.packages)
            logger.info(
                f"  {icon:<10} | {env.name:<25} | Python {version_str:<10} | "
                f"{pkg_count:>4} packages | Path: {env.path}"
            )

    def print_redundancies(self, global_env: Optional[PythonEnvironment], venvs: List[PythonEnvironment]) -> bool:
        """Reports on packages found in both global and virtual environments.
        Returns True if redundancies found.
        """
        logger.info("\n" + "-" * 80)
        logger.info(f"{LEMON} REDUNDANCY CHECK (vs Global Environment)")
        logger.info("-" * 80)

        if not global_env or not global_env.packages:
            logger.info("  ℹ️ Global environment packages not available. Skipping redundancy checks.")
            return False
        if not venvs:
            logger.info("  ℹ️ No virtual environments to compare against global. Skipping redundancy checks.")
            return False

        total_redundant_found = False
        for venv in venvs:
            if not venv.packages:
                continue

            redundant_packages_info = []  # List of (name, global_ver_str, venv_ver_str, match_status)
            for pkg_name, venv_pkg_info in venv.packages.items():
                if pkg_name in global_env.packages:
                    global_pkg_info = global_env.packages[pkg_name]

                    match_status = "❓ Unknown (parsing error)"
                    if (
                        PACKAGING_AVAILABLE
                        and not isinstance(global_pkg_info.parsed_version, str)
                        and not isinstance(venv_pkg_info.parsed_version, str)
                    ):
                        match_status = (
                            "✅ Match"
                            if global_pkg_info.parsed_version == venv_pkg_info.parsed_version
                            else "⚠️ Mismatch"
                        )
                    elif global_pkg_info.raw_version == venv_pkg_info.raw_version:  # Fallback
                        match_status = "✅ Match"
                    else:
                        match_status = "⚠️ Mismatch (basic)"

                    redundant_packages_info.append(
                        (
                            pkg_name,
                            global_pkg_info.raw_version,
                            venv_pkg_info.raw_version,
                            match_status,
                        )
                    )

            if redundant_packages_info:
                total_redundant_found = True
                logger.info(f"\n  Virtual Environment: {venv.name} (Path: {venv.path})")
                header = (
                    f"    {'Package':<30} {'Global Version':<18} {'VEnv Version':<18} "
                    f"{'Status':<15}"
                )
                logger.info(header)
                logger.info(f"    {'-' * 30} {'-' * 18} {'-' * 18} {'-' * 15}")
                for name, g_ver, v_ver, status in sorted(redundant_packages_info):
                    logger.info(f"    {name:<30} {g_ver:<18} {v_ver:<18} {status:<15}")

        if not total_redundant_found:
            logger.info(
                "  ✅ No redundant packages (present in global) found in analyzed virtual "
                "environments."
            )
        return total_redundant_found

    def print_conflicts(self, conflicts: List[ConflictInfo]) -> None:
        """Displays detected package version conflicts between environments."""
        logger.info("\n" + "-" * 80)
        logger.info(f"{LEMON} INTER-ENVIRONMENT CONFLICT ANALYSIS")
        logger.info("-" * 80)

        if not conflicts:
            logger.info("  ✅ No version conflicts detected between different environments for common packages!")
            return

        conflicts.sort(
            key=lambda c: ({"major": 0, "minor": 1, "patch": 2, "other": 3}[c.severity], c.package)
        )

        severity_map = {
            "major": ("🔴 Major", []),
            "minor": ("🟡 Minor", []),
            "patch": ("🔵 Patch", []),
            "other": ("⚪ Other", []),
        }
        for conflict in conflicts:
            severity_map[conflict.severity][1].append(conflict)

        for sev_key, (sev_display_name, conflict_list) in severity_map.items():
            if conflict_list:
                logger.info(f"\n  {sev_display_name} Conflicts ({len(conflict_list)} found):")
                for c in conflict_list:
                    logger.info(
                        f"    • {c.package:<25} | {c.env1_name} ({c.env1_version}) vs. "
                        f"{c.env2_name} ({c.env2_version})"
                    )

    def print_python_recommendation(self, rec: PythonVersionRecommendation) -> None:
        """Presents the recommended optimal Python version."""
        logger.info("\n" + "-" * 80)
        logger.info(f"{LEMON} PYTHON VERSION RECOMMENDATION (based on important package needs)")
        logger.info("-" * 80)

        if not rec.total_analyzed_package_requirements and not rec.warnings:
            # No data and no specific warning means it's likely the default
            logger.info(
                f"  🐍 Recommended Python Version: {rec.recommended_version} (default due to "
                "no specific package requirements found)"
            )
        else:
            logger.info(f"\n  🐍 Recommended Python Version: {rec.recommended_version}")
            logger.info(
                f"     (Based on {rec.total_analyzed_package_requirements} 'Requires-Python' "
                "analyzed from important packages)"
            )
            logger.info(f"  ✅ {rec.compatible_packages_count} analyzed package requirements are compatible.")

        if rec.incompatible_packages:
            logger.info(
                f"  ⚠️ Incompatible package requirements with {rec.recommended_version} "
                f"({len(rec.incompatible_packages)} found):"
            )
            for pkg, req_spec in rec.incompatible_packages[:5]:  # Show top 5
                logger.info(f"    • {pkg} (requires: Python {req_spec})")
            if len(rec.incompatible_packages) > 5:
                logger.info(f"    ...and {len(rec.incompatible_packages) - 5} more.")

        if rec.warnings:
            for warn_msg in rec.warnings:
                logger.warning(f"  ❗ Warning: {warn_msg}")

    def print_consolidation_suggestions(self, venvs: List[PythonEnvironment]) -> None:
        """Suggests virtual environments that could potentially be consolidated."""
        logger.info("\n" + "-" * 80)
        logger.info(f"{LEMON} ENVIRONMENT CONSOLIDATION SUGGESTIONS (Jaccard Similarity)")
        logger.info("-" * 80)

        eligible_venvs = [env for env in venvs if env.packages]  # Only venvs with packages
        if len(eligible_venvs) < 2:
            logger.info("  ℹ️ Not enough virtual environments with package data to suggest consolidation.")
            return

        suggestions_made = False
        # Avoid re-calculating for pairs by using a set of checked pairs
        checked_pairs = set()

        for i in range(len(eligible_venvs)):
            for j in range(i + 1, len(eligible_venvs)):
                env1, env2 = eligible_venvs[i], eligible_venvs[j]

                # Create a canonical representation for the pair to store in checked_pairs
                pair_key = tuple(sorted((env1.name, env2.name)))
                if pair_key in checked_pairs:
                    continue

                similarity = self._compute_jaccard_similarity(env1, env2)
                checked_pairs.add(pair_key)

                if similarity >= CONSOLIDATION_THRESHOLD:
                    suggestions_made = True
                    percent = round(similarity * 100, 1)
                    logger.info(
                        f"  • Consider consolidating '{env1.name}' and '{env2.name}' "
                        f"(Similarity: {percent}%)"
                    )
                    logger.info(f"    Env1 ({len(env1.packages)} pkgs): {env1.path}")
                    logger.info(f"    Env2 ({len(env2.packages)} pkgs): {env2.path}")

        if not suggestions_made:
            logger.info(
                "  ✅ No strong consolidation candidates found based on current package "
                "similarity threshold."
            )

    @staticmethod
    def _compute_jaccard_similarity(env1: PythonEnvironment, env2: PythonEnvironment) -> float:
        """Computes Jaccard similarity between two environments based on package names."""
        set1 = set(env1.packages.keys())
        set2 = set(env2.packages.keys())
        if not set1 or not set2:  # Avoid division by zero if one set is empty
            return 0.0

        intersection_len = len(set1.intersection(set2))
        union_len = len(set1.union(set2))

        return intersection_len / union_len if union_len > 0 else 0.0

    def print_final_summary(self, has_conflicts: bool, has_redundancies: bool, use_uv: bool) -> None:
        """Prints final recommendations and closing remarks."""
        logger.info("\n" + "-" * 80)
        logger.info(f"{LEMON} KEY TAKEAWAYS & RECOMMENDATIONS")
        logger.info("-" * 80)

        recommendations = []
        if has_redundancies:
            recommendations.append(
                "Reduce clutter by removing packages from virtual environments if "
                "identical versions exist globally and are appropriate for the project."
            )
            recommendations.append(
                "For managing isolated command-line tools, consider using `pipx`."
            )
        if has_conflicts:
            recommendations.append(
                "Address 'major' version conflicts first, as these are most likely to "
                "cause issues. Ensure projects with such conflicts use dedicated "
                "virtual environments."
            )
        if not use_uv and UvExtractor.is_available():  # If UV available but not used
            recommendations.append(
                "Speed up analysis! UV is available. Try running with `--use-uv` for "
                "faster package processing."
            )
        elif not use_uv and not UvExtractor.is_available():  # If UV not used and not available
            recommendations.append(
                "For potentially faster analysis in the future, consider installing `uv` "
                "(e.g., `pip install uv` or from astral.sh) and using the `--use-uv` flag."
            )

        if not recommendations:
            logger.info("  ✨ Your Python environments look generally well-maintained based on the checks performed!")
        else:
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")

        logger.info(
            "\nRegularly reviewing your Python setup can help maintain a clean and "
            "efficient development workflow. Happy coding! 🚀"
        )
        logger.info("\n" + "=" * 80)
        logger.info(f"  Thanks for using {LEMON} Dependency Detox!".center(80))
        logger.info("=" * 80 + "\n")


class DependencyDetox:
    """Main application orchestrator."""

    def __init__(self, use_uv_arg: bool, analyze_conflicts_arg: bool, suggest_python_arg: bool):
        self.use_uv = use_uv_arg
        self.analyze_conflicts = analyze_conflicts_arg
        self.suggest_python = suggest_python_arg

        self.extractor: PackageExtractor = self._select_extractor()
        self.reporter = ReportGenerator(use_uv_flag=isinstance(self.extractor, UvExtractor))

    def _select_extractor(self) -> PackageExtractor:
        """Determines which package extractor to use (UV or pip)."""
        if self.use_uv:
            if UvExtractor.is_available():
                return UvExtractor()
            else:
                logger.warning(f"{LEMON} UV was requested but is not available or functional. " "Falling back to pip.")
        return PipExtractor()

    def _load_and_analyze_environments(
        self, venv_paths: List[str], search_dirs: List[str]
    ) -> Tuple[List[PythonEnvironment], Optional[PythonEnvironment]]:
        """Discovers, loads, and performs initial analysis on environments."""
        processed_environments: List[PythonEnvironment] = []
        global_env: Optional[PythonEnvironment] = None

        # 1. Discover and analyze global environment
        global_env_path_str = EnvironmentDiscovery.find_global_environment_path()
        if global_env_path_str:
            env_obj = PythonEnvironment(global_env_path_str, "Global System Python", is_global=True)
            if env_obj.is_valid():
                env_obj.get_python_version()  # Populate version
                env_obj.packages = self.extractor.extract(env_obj)
                if env_obj.packages:  # Only add if packages were found
                    processed_environments.append(env_obj)
                    global_env = env_obj
                else:
                    logger.warning(
                        "  ⚠️ Could not extract packages from the identified global "
                        f"environment ({env_obj.name}). It will be excluded from "
                        "detailed analysis."
                    )
            else:
                logger.warning(
                    f"  ⚠️ Identified global environment at {global_env_path_str} "
                    "seems invalid or inaccessible."
                )
        else:
            logger.warning(
                "  ⚠️ Global Python environment could not be found. Skipping analyses "
                "that require it (like redundancy checks)."
            )

        # 2. Discover and analyze virtual environments
        # Combine explicitly provided paths and discovered paths, ensuring uniqueness
        unique_venv_paths: Set[str] = set(map(lambda p: str(Path(p).resolve()), venv_paths))
        for s_dir in search_dirs:
            found_paths = EnvironmentDiscovery.find_virtual_environments(s_dir)
            unique_venv_paths.update(map(lambda p: str(Path(p).resolve()), found_paths))

        sorted_venv_paths = sorted(list(unique_venv_paths))  # Process in a consistent order

        for venv_p_str in sorted_venv_paths:
            venv_path = Path(venv_p_str)
            env_obj = PythonEnvironment(str(venv_path), venv_path.name, is_global=False)

            if not env_obj.is_valid():
                logger.warning(f"  ⚠️ Skipping invalid or inaccessible virtual environment: {venv_path}")
                continue

            env_obj.get_python_version()
            env_obj.packages = self.extractor.extract(env_obj)

            if env_obj.packages:  # Only add if packages were found
                # Check if this venv is actually the same as the identified global_env
                if global_env and env_obj.path == global_env.path:
                    logger.info(
                        f"  ℹ️ Virtual environment '{env_obj.name}' is the same as the "
                        "active global/base. Merging details."
                    )
                    # Potentially merge/update global_env if this one has more info,
                    # or just skip adding. For now, we assume the global discovery
                    # was sufficient if paths match.
                else:
                    processed_environments.append(env_obj)
            else:
                logger.warning(f"  ⚠️ No packages found or extraction failed for {env_obj.name}. Skipping.")

        return processed_environments, global_env

    def run(self, venv_paths: List[str], search_dirs: List[str]) -> None:
        """Executes the environment analysis and generates reports."""
        self.reporter.print_header()

        all_analyzed_environments, global_env_obj = self._load_and_analyze_environments(
            venv_paths, search_dirs
        )

        if not all_analyzed_environments:
            logger.error("❌ No valid Python environments with package data found to analyze. Exiting.")
            return  # No sys.exit here, let main handle exit codes

        self.reporter.print_environments_summary(all_analyzed_environments)

        # Separate virtual environments for specific analyses
        virtual_envs_only = [e for e in all_analyzed_environments if not e.is_global]

        # --- Perform Analyses ---
        has_any_redundancies = False
        if global_env_obj and virtual_envs_only:  # Need global and some venvs to check redundancy
            has_any_redundancies = self.reporter.print_redundancies(
                global_env_obj, virtual_envs_only
            )

        conflict_list: List[ConflictInfo] = []
        if self.analyze_conflicts and len(all_analyzed_environments) >= 2:  # Need at least two envs
            conflict_list = ConflictAnalyzer.detect_conflicts(all_analyzed_environments)
            self.reporter.print_conflicts(conflict_list)
        elif self.analyze_conflicts:
            logger.info("\nℹ️ Conflict analysis skipped: requires at least two environments with package data.")

        if self.suggest_python:
            py_version_rec = PythonVersionAnalyzer.get_recommendation(all_analyzed_environments)
            self.reporter.print_python_recommendation(py_version_rec)

        if len(virtual_envs_only) >= 2:  # Need at least two venvs to suggest consolidation
            self.reporter.print_consolidation_suggestions(virtual_envs_only)
        else:
            logger.info(
                "\nℹ️ Consolidation suggestions skipped: requires at least two virtual "
                "environments with package data."
            )

        self.reporter.print_final_summary(
            has_conflicts=bool(conflict_list),
            has_redundancies=has_any_redundancies,
            use_uv=isinstance(self.extractor, UvExtractor),
        )


def main() -> None:
    """Main entry point for the Dependency Detox CLI tool."""
    parser = argparse.ArgumentParser(
        description=f"{LEMON} Dependency Detox - Python Environment Cleanse Tool {LEMON}",
        epilog="Tidy up your Python spaces! We diagnose, you decide. No packages are altered.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "venvs",
        nargs="*",
        metavar="VENV_PATH",
        help="Direct paths to virtual environment directories to analyze.",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        dest="search_dirs",
        metavar="DIR_PATH",
        help=(
            "Directory to recursively search for virtual environments (e.g., '~/projects'). "
            "Can be used multiple times."
        ),
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help="Prefer `uv` for faster package operations. Falls back to `pip` if `uv` is unavailable.",
    )
    parser.add_argument(
        "--analyze-conflicts",
        action="store_true",
        help="Enable analysis of package version conflicts across all analyzed environments.",
    )
    parser.add_argument(
        "--suggest-python-version",
        action="store_true",
        help=(
            "Suggest an optimal Python version based on 'Requires-Python' metadata "
            "from important packages."
        ),
    )
    parser.add_argument(
        "--full-detox",
        action="store_true",
        help=(
            "Run all analyses: conflict detection, Python version suggestion, redundancy "
            "checks, and consolidation suggestions."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0",  # Example version
        help="Show program's version number and exit.",
    )

    args = parser.parse_args()

    # Update logger level based on verbosity argument
    logger.setLevel(args.verbose)
    if args.verbose == logging.DEBUG:
        logger.info("Debug logging enabled.")
        if not PACKAGING_AVAILABLE:
            logger.debug(
                "Note: 'packaging' library is not installed. Version comparisons "
                "will use a fallback mechanism."
            )
        else:
            logger.debug("'packaging' library is available. Using it for robust version handling.")

    if args.full_detox:
        args.analyze_conflicts = True
        args.suggest_python_version = True
        logger.info("🧪 Full Detox mode enabled: all analysis features are active.")

    if not args.venvs and not args.search_dirs:
        current_dir = os.getcwd()
        logger.info(
            f"{LEMON} No specific environments or search directories provided. "
            f"Defaulting to search for venvs in the current directory: {current_dir}"
        )
        args.search_dirs.append(current_dir)

    try:
        detox_app = DependencyDetox(
            use_uv_arg=args.use_uv,
            analyze_conflicts_arg=args.analyze_conflicts,
            suggest_python_arg=args.suggest_python_version,
        )
        detox_app.run(args.venvs, args.search_dirs)
    except KeyboardInterrupt:
        logger.info("\n🚫 Detox process interrupted by user. Exiting gracefully.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"\n💥 An unexpected critical error occurred: {type(e).__name__} - {e}")
        logger.debug("Traceback:", exc_info=True)  # Log full traceback in debug mode
        sys.exit(1)


if __name__ == "__main__":
    main()
