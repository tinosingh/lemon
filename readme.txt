================================================================================
🍋 Lemon – The Dependency Detox - Python Environment Cleanse Tool 🍋
================================================================================

--------------------------------------------------------------------------------
1. Overview / Functional Description
--------------------------------------------------------------------------------

Dependency Detox is a command-line tool designed to help Python developers analyze
and understand their Python environments. It scans specified virtual environments
and the global Python installation to identify potential issues, redundancies,
and opportunities for optimization.

Key Features:
* Environment Discovery: Automatically locates the global Python installation
    and searches specified directories for virtual environments.
* Package Listing: Extracts lists of installed packages and their versions
    from each environment.
* UV Support: Can utilize 'uv' (if installed) for significantly faster
    package extraction and analysis, falling back to 'pip' if uv is not
    available or not requested.
* Redundancy Checks: Identifies packages present in virtual environments
    that also exist in the global Python installation, highlighting version
    matches or mismatches.
* Conflict Analysis: Detects package version conflicts for common libraries
    across different analyzed environments, categorizing them by severity
    (major, minor, patch, other).
* Python Version Recommendation: Suggests an optimal Python version
    (e.g., 3.9, 3.10, 3.11) based on the 'Requires-Python' metadata of
    important packages found in the analyzed environments.
* Consolidation Suggestions: Proposes potential consolidation of virtual
    environments that have a high degree of package similarity (based on
    Jaccard index).
* Informative Reporting: Provides a clear, console-based report summarizing
    its findings and recommendations.

Disclaimer:
Dependency Detox is a diagnostic tool. It analyzes your environments and
provides insights but *NEVER* makes any changes to your environments or
packages. Your files and installations are safe.

--------------------------------------------------------------------------------
2. User Guide
--------------------------------------------------------------------------------

2.1. Prerequisites
-------------------
* Python: Version 3.8 or newer is recommended.
* pip: Must be available in your Python environments for package extraction
    if 'uv' is not used.
* (Optional but Recommended for speed) uv: Install 'uv' for faster analysis.
    You can get it from Astral (astral.sh) or via pip: `pip install uv`.
* (Optional but Recommended for accuracy) `packaging` library:
    For the most accurate version parsing and compatibility checks. If not
    installed (`pip install packaging`), the tool will use a less robust
    fallback mechanism.

2.2. Installation
------------------
1.  Save the script: Download or save the Python script code as a file,
    for example, `detox.py`.
2.  (Optional) Make it executable (Linux/macOS):
    `chmod +x detox.py`
    You can then run it as `./detox.py`. Otherwise, use `python detox.py`.

2.3. Usage (Command-Line Interface)
-----------------------------------
Run the script from your terminal:

Basic syntax:
`python detox.py [OPTIONS] [VENV_PATH_1 VENV_PATH_2 ...]`

Or if executable:
`./detox.py [OPTIONS] [VENV_PATH_1 VENV_PATH_2 ...]`

Arguments and Options:
* `VENV_PATH ...`: (Positional) Space-separated direct paths to virtual
    environment directories you want to analyze.
* `--search-dir DIR_PATH`: Recursively search `DIR_PATH` for virtual
    environments (identified by `pyvenv.cfg`). Can be used multiple times.
    If no VENV_PATHs or --search-dirs are given, it defaults to searching
    the current working directory.
* `--use-uv`: Attempt to use `uv` for faster package extraction. If `uv`
    is not found or fails, it falls back to `pip`.
* `--analyze-conflicts`: Enable analysis of package version conflicts across
    all analyzed environments.
* `--suggest-python-version`: Suggest an optimal Python version based on
    'Requires-Python' metadata from important packages.
* `--full-detox`: A shortcut to enable all analysis features:
    `--analyze-conflicts` and `--suggest-python-version`.
* `-v`, `--verbose`: Enable verbose debug logging for more detailed output,
    useful for troubleshooting.
* `--version`: Show the script's version number and exit.
* `-h`, `--help`: Show the help message and exit.

Examples:
* Analyze two specific virtual environments:
    `python detox.py /path/to/myenv1 /path/to/project_env`
* Search for environments in your projects folder and use UV:
    `python detox.py --search-dir ~/projects --search-dir ~/other_venvs --use-uv`
* Perform a full detox analysis on venvs in the current directory:
    `python detox.py --full-detox`
* Analyze a specific venv and get Python version suggestions with verbose output:
    `python detox.py /path/to/special_env --suggest-python-version -v`

2.4. Understanding the Output
-----------------------------
The tool generates a report directly to your console. Key sections include:

* Header & Analysis Mode: Shows the tool name and whether it's using 'pip'
    or 'uv'.
* Environments Analyzed: A summary list of all Python environments (global
    and virtual) that were found and analyzed, including their type, name,
    Python version, package count, and path.
* Redundancy Check: Lists packages found in your virtual environments that
    are also present in your global Python installation. It indicates if the
    versions match or mismatch.
* Inter-Environment Conflict Analysis: (If enabled) Highlights packages that
    have different versions across any two analyzed environments. Conflicts are
    categorized by severity (Major, Minor, Patch, Other) to help prioritize.
* Python Version Recommendation: (If enabled) Suggests a Python version (e.g.,
    "3.11") that aims to be compatible with the 'Requires-Python' metadata of
    important packages (like numpy, pandas, Django, etc.) found across your
    environments. It also lists any packages whose requirements might not be
    met by the recommended version.
* Environment Consolidation Suggestions: (If virtual environments are analyzed)
    Suggests pairs of virtual environments that have a high similarity in their
    installed packages (above a certain threshold), indicating they might be
    candidates for consolidation.
* Key Takeaways & Recommendations: Provides a brief summary of actionable advice
    based on the findings, such as addressing major conflicts or considering `pipx`
    for tools.

--------------------------------------------------------------------------------
3. Technical Description
--------------------------------------------------------------------------------

3.1. Core Architecture
----------------------
Dependency Detox is structured modularly:
* Orchestrator (`DependencyDetox`): The main class that coordinates the entire
    analysis process, from environment discovery to report generation.
* Environment Representation (`PythonEnvironment`): A dataclass that holds
    information about a single Python environment (path, name, Python version,
    installed packages).
* Discovery (`EnvironmentDiscovery`): Contains static methods to find the
    global Python environment and search for virtual environments on the
    filesystem.
* Package Extraction (`PackageExtractor` Protocol): Defines an interface for
    extracting package information.
    * `PipExtractor`: Implements extraction using `pip list --format=json` and
        `pip show`.
    * `UvExtractor`: Implements extraction using `uv pip list --format json`
        and `uv pip show`.
    * `BasePackageExtractor`: A base class providing common JSON parsing and
        metadata enrichment logic for extractors.
* Analysis Components:
    * `ConflictAnalyzer`: Detects and categorizes version conflicts between
        environments.
    * `PythonVersionAnalyzer`: Recommends an optimal Python version based on
        package metadata.
* Reporting (`ReportGenerator`): Responsible for formatting and printing the
    analysis results to the console.
* Subprocess Management (`SubprocessRunner`): A utility class to run external
    commands (like pip, uv, python --version) consistently, with built-in
    error handling and timeout management.

3.2. Key Mechanisms
-------------------
* Environment Discovery:
    * Global Python: Primarily uses `sys.prefix` and `sysconfig`, with
        fallbacks to `which python3` (Unix-like) or `where python` (Windows).
    * Virtual Environments: Scans directories for `pyvenv.cfg` files, then
        validates the presence of a Python executable in standard `bin/` or
        `Scripts/` subdirectories.
* Package Listing: Executes `pip list --format=json --not-required` or
    `uv pip list --format json [--python <path>]` in a subprocess for each
    environment to get a list of installed packages.
* Metadata Enrichment: For "important" packages (a predefined list), it runs
    `pip show <package>` or `uv pip show <package>` to fetch metadata,
    specifically the `Requires-Python` field.
* Version Parsing & Comparison:
    * Utilizes the `packaging` library (if installed) for robust parsing of
        version strings (PEP 440) and evaluation of Python version specifiers
        (PEP 440, e.g., ">=3.7, <3.12").
    * If `packaging` is not available, it falls back to a simpler regex-based
        parsing method, which is less comprehensive.
* Conflict Severity: Compares parsed versions (if available) to determine if
    differences are major, minor, patch, or other (e.g., pre-releases, different
    build metadata).
* Jaccard Similarity: Calculates the Jaccard index (intersection over union)
    of package sets between pairs of virtual environments to suggest candidates
    for consolidation.

3.3. Dependencies
-----------------
* Python Standard Libraries: `argparse`, `json`, `logging`, `os`, `platform`,
    `pathlib`, `re`, `subprocess`, `sys`, `sysconfig`.
* External Python Libraries (Optional but Recommended):
    * `packaging`: Used for accurate PEP 440 version parsing and specifier
        evaluation. The tool includes a basic fallback if this library is not
        present but will be more reliable with it. Install via
        `pip install packaging`.

3.4. Potential Future Enhancements
---------------------------------
* Support for Conda environments.
* Configuration file for settings like "important packages" or thresholds.
* More detailed dependency tree analysis within environments.
* Interactive HTML report generation.
* A Text User Interface (TUI) or a graphical user interface (GUI).

--------------------------------------------------------------------------------
4. License
--------------------------------------------------------------------------------

This tool is licensed under the MIT License. See the license text in the script
file or a separate LICENSE file if provided.

--------------------------------------------------------------------------------
5. Author
--------------------------------------------------------------------------------

Dependency Detox is brought to you by:
The Lemon Squad / Tino Singh