"""Standalone environment and repository health check for GenPy-Mini Phase 1.

Run directly with ``python scripts/validate_environment.py``. The script
bootstraps its own import path so it works even before an editable install
has been performed.

No CUDA or GPU is required for Phase 1: a missing NVIDIA GPU is never
treated as a failure. Only actual project setup problems (a missing or
invalid model configuration, or missing required directories) produce a
non-zero exit code.
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (_PROJECT_ROOT, _SRC_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from config.settings import MODEL_CONFIG_PATH, PROJECT_ROOT  # noqa: E402
from genpy.config import ConfigError, load_model_config  # noqa: E402

_REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "config",
    "data/raw",
    "data/interim",
    "data/cleaned",
    "data/tokenized",
    "data/instructions",
    "data/evaluation",
    "data/reports",
    "docs",
    "notebooks",
    "scripts",
    "src/genpy",
    "tests",
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single environment or repository check."""

    name: str
    passed: bool
    detail: str
    critical: bool = False

    def is_blocking(self) -> bool:
        """A check only fails the run if it is both unsuccessful and critical."""
        return self.critical and not self.passed


def _check_python_version() -> CheckResult:
    version = platform.python_version()
    ok = sys.version_info >= (3, 11)
    return CheckResult("Python version", ok, f"{version} (>= 3.11 required)", critical=True)


def _check_operating_system() -> CheckResult:
    detail = f"{platform.system()} {platform.release()} ({platform.machine()})"
    return CheckResult("Operating system", True, detail)


def _check_git_available() -> CheckResult:
    git_path = shutil.which("git")
    return CheckResult("Git available", git_path is not None, git_path or "git not found on PATH")


def _detect_windows_total_ram_gb() -> float | None:
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    stat = _MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return float(stat.ullTotalPhys) / (1024**3)


def _detect_linux_total_ram_gb() -> float | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.is_file():
        return None
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return kib / (1024**2)
    return None


def _detect_total_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            return _detect_windows_total_ram_gb()
        if sys.platform.startswith("linux"):
            return _detect_linux_total_ram_gb()
    except (OSError, ValueError, AttributeError):
        return None
    return None


def _check_total_ram() -> CheckResult:
    ram_gb = _detect_total_ram_gb()
    if ram_gb is None:
        return CheckResult(
            "Total RAM", True, "could not be detected without additional dependencies"
        )
    return CheckResult("Total RAM", True, f"{ram_gb:.1f} GB")


def _check_disk_space() -> CheckResult:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
    except OSError as exc:
        return CheckResult("Available disk space", False, f"could not be determined: {exc}")
    free_gb = usage.free / (1024**3)
    return CheckResult("Available disk space", True, f"{free_gb:.1f} GB free")


def _check_model_config_exists() -> CheckResult:
    exists = MODEL_CONFIG_PATH.is_file()
    return CheckResult("Model config file exists", exists, str(MODEL_CONFIG_PATH), critical=True)


def _check_model_config_loads() -> CheckResult:
    try:
        config = load_model_config(MODEL_CONFIG_PATH)
    except ConfigError as exc:
        return CheckResult("Model config loads and validates", False, str(exc), critical=True)
    detail = f"{config.model_name}: d_model={config.d_model}, layers={config.num_layers}"
    return CheckResult("Model config loads and validates", True, detail, critical=True)


def _check_required_directories() -> CheckResult:
    missing = [d for d in _REQUIRED_DIRECTORIES if not (PROJECT_ROOT / d).is_dir()]
    if missing:
        detail = f"missing: {', '.join(missing)}"
        return CheckResult("Required project directories", False, detail, critical=True)
    detail = f"{len(_REQUIRED_DIRECTORIES)} directories present"
    return CheckResult("Required project directories", True, detail, critical=True)


def run_all_checks() -> list[CheckResult]:
    """Run every environment and repository check and return the results."""
    return [
        _check_python_version(),
        _check_operating_system(),
        _check_git_available(),
        _check_total_ram(),
        _check_disk_space(),
        _check_model_config_exists(),
        _check_model_config_loads(),
        _check_required_directories(),
    ]


def _print_report(results: list[CheckResult]) -> None:
    print("GenPy-Mini environment validation")
    print("=" * 60)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    print("=" * 60)
    print("No GPU/CUDA is required for Phase 1; a missing NVIDIA GPU is not a failure.")


def main() -> int:
    """Run all checks, print a human-readable report, and return an exit code."""
    results = run_all_checks()
    _print_report(results)
    return 1 if any(result.is_blocking() for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
