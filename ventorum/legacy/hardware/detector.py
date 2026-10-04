# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Hardware capability and machine fingerprint detection module for Ventorum.

Scans the local machine's compute architecture (CPU cores, architecture, RAM,
GPU accelerators, CUDA capabilities) and generates a cryptographic machine
fingerprint. This allows Ventorum to verify whether a stored configuration was
calibrated for the current hardware or if the installation was moved to a
different machine.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareInfo:
    """Immutable snapshot of the local machine's hardware capabilities."""

    # OS / Platform
    system: str
    release: str
    machine: str
    node_name: str

    # CPU
    cpu_processor: str
    logical_cores: int
    physical_cores: int

    # RAM
    total_ram_gb: float

    # GPU
    gpu_available: bool
    gpu_name: str | None
    gpu_count: int
    gpu_vram_gb: float | None
    cuda_version: str | None
    cuda_compute_capability: tuple[int, int] | None

    # Cryptographic Hardware Fingerprint
    machine_guid: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Convert hardware info to a JSON-serializable dictionary."""
        d = asdict(self)
        if self.cuda_compute_capability is not None:
            d["cuda_compute_capability"] = list(self.cuda_compute_capability)
        return d


def _get_windows_machine_guid() -> str:
    """Retrieve the Windows MachineGuid from the registry."""
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography"
        ) as key:
            val, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(val).strip()
    except Exception:
        return ""


def _get_linux_machine_id() -> str:
    """Retrieve Linux machine-id from /etc/machine-id or /var/lib/dbus/machine-id."""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
        except Exception:
            continue
    return ""


def _get_macos_uuid() -> str:
    """Retrieve macOS IOPlatformUUID via ioreg."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        for line in out.splitlines():
            if "IOPlatformUUID" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    return parts[1].strip().strip('"')
    except Exception:
        pass
    return ""


def _get_system_machine_id() -> str:
    """Cross-platform unique persistent machine identifier."""
    sys_name = platform.system().lower()
    if sys_name == "windows":
        guid = _get_windows_machine_guid()
        if guid:
            return f"win:{guid}"
    elif sys_name == "linux":
        mid = _get_linux_machine_id()
        if mid:
            return f"linux:{mid}"
    elif sys_name == "darwin":
        muid = _get_macos_uuid()
        if muid:
            return f"darwin:{muid}"

    # Stable fallback: combination of MAC node and node name
    node_mac = hex(uuid.getnode())
    return f"fallback:{platform.node()}:{node_mac}"


def _get_total_ram_gb() -> float:
    """Cross-platform physical RAM detection in gigabytes without external dependencies."""
    sys_name = platform.system().lower()
    if sys_name == "windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024**3), 2)
        except Exception:
            pass

    elif sys_name == "linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return round(kb / (1024**2), 2)
        except Exception:
            pass

    elif sys_name == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            bytes_val = float(out.strip())
            return round(bytes_val / (1024**3), 2)
        except Exception:
            pass

    return 8.0  # conservative baseline fallback


def _get_physical_cpu_cores(logical_cores: int) -> int:
    """Estimate or query physical CPU core count."""
    sys_name = platform.system().lower()
    if sys_name == "windows":
        try:
            cmd = "Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum | Select-Object -ExpandProperty Sum"
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            val = int(out.strip())
            if val > 0:
                return val
        except Exception:
            pass
    elif sys_name == "linux":
        try:
            cores = set()
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                physical_id = "0"
                for line in f:
                    if line.startswith("physical id"):
                        physical_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                        cores.add(f"{physical_id}:{core_id}")
            if cores:
                return len(cores)
        except Exception:
            pass
    elif sys_name == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            val = int(out.strip())
            if val > 0:
                return val
        except Exception:
            pass

    # Standard heuristic if hardware queries fail: logical // 2 if hyperthreaded, else logical
    if logical_cores >= 4:
        return max(1, logical_cores // 2)
    return max(1, logical_cores)


def _detect_gpu_capabilities() -> dict[str, Any]:
    """Detect NVIDIA CUDA GPU presence and compute specifications."""
    gpu_info: dict[str, Any] = {
        "available": False,
        "name": None,
        "count": 0,
        "vram_gb": None,
        "cuda_version": None,
        "compute_capability": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info["vram_gb"] = round(props.total_memory / (1024**3), 2)
            gpu_info["cuda_version"] = str(torch.version.cuda)
            cap = torch.cuda.get_device_capability(0)
            gpu_info["compute_capability"] = (int(cap[0]), int(cap[1]))
            return gpu_info
    except Exception:
        pass

    # Check via nvidia-smi if torch CUDA failed or not installed
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        lines = out.strip().splitlines()
        if lines:
            parts = lines[0].split(",")
            name = parts[0].strip()
            mem_mb = float(parts[1].strip()) if len(parts) > 1 else 0.0
            gpu_info["available"] = True
            gpu_info["name"] = name
            gpu_info["count"] = len(lines)
            gpu_info["vram_gb"] = round(mem_mb / 1024.0, 2)
    except Exception:
        pass

    return gpu_info


def generate_fingerprint(
    system: str,
    machine_id: str,
    cpu_processor: str,
    logical_cores: int,
    total_ram_gb: float,
    gpu_name: str | None,
) -> str:
    """Generate a canonical SHA-256 fingerprint from hardware attributes.

    Parameters
    ----------
    system : str
        Operating system ('Windows', 'Linux', 'Darwin').
    machine_id : str
        System hardware identifier (MachineGuid, machine-id, etc.).
    cpu_processor : str
        CPU processor identifier string.
    logical_cores : int
        Number of logical CPU threads.
    total_ram_gb : float
        Total system RAM. Bucketed to nearest whole GB to prevent false
        mismatches from dynamic OS memory reservations.
    gpu_name : str or None
        Primary GPU name, or 'none' if no GPU is active.

    Returns
    -------
    str
        Deterministic 32-character hexadecimal fingerprint.
    """
    # Round RAM to nearest whole GB for stability
    ram_bucket = round(total_ram_gb)
    canonical = {
        "system": system.strip().lower(),
        "machine_id": machine_id.strip(),
        "cpu_processor": cpu_processor.strip(),
        "logical_cores": int(logical_cores),
        "ram_gb": int(ram_bucket),
        "gpu": (gpu_name or "none").strip().lower(),
    }
    raw_str = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:32]


def scan_hardware() -> HardwareInfo:
    """Scan the local machine hardware and return a comprehensive HardwareInfo snapshot."""
    sys_name = platform.system()
    sys_release = platform.release()
    sys_machine = platform.machine()
    node_name = platform.node()

    cpu_proc = platform.processor() or "Unknown CPU"
    log_cores = os.cpu_count() or 1
    phys_cores = _get_physical_cpu_cores(log_cores)

    ram_gb = _get_total_ram_gb()
    gpu_data = _detect_gpu_capabilities()

    raw_machine_id = _get_system_machine_id()
    fingerprint = generate_fingerprint(
        system=sys_name,
        machine_id=raw_machine_id,
        cpu_processor=cpu_proc,
        logical_cores=log_cores,
        total_ram_gb=ram_gb,
        gpu_name=gpu_data["name"],
    )

    return HardwareInfo(
        system=sys_name,
        release=sys_release,
        machine=sys_machine,
        node_name=node_name,
        cpu_processor=cpu_proc,
        logical_cores=log_cores,
        physical_cores=phys_cores,
        total_ram_gb=ram_gb,
        gpu_available=gpu_data["available"],
        gpu_name=gpu_data["name"],
        gpu_count=gpu_data["count"],
        gpu_vram_gb=gpu_data["vram_gb"],
        cuda_version=gpu_data["cuda_version"],
        cuda_compute_capability=gpu_data["compute_capability"],
        machine_guid=raw_machine_id,
        fingerprint=fingerprint,
    )


def get_machine_fingerprint() -> str:
    """Quickly compute and return the current machine's hardware fingerprint."""
    return scan_hardware().fingerprint
