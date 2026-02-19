"""
Hardware profiler for The Inference Difference.

Detects available compute resources (GPU, CPU, RAM, VRAM) and determines
which local models can actually run on this hardware.

Non-invasive: only reads system info, never modifies anything.
Graceful degradation: missing tools (nvidia-smi, etc.) result in
zeroed-out fields, not errors.

Changelog (Grok audit response, 2026-02-19):
- ADDED: AMD GPU detection via rocm-smi (audit: "NVIDIA-centrism").
  Intel discrete GPUs via clinfo deferred — too varied in output format
  and low adoption for local LLM inference. Will revisit when Intel Arc
  gains ollama support.
- ADDED: VRAM bounds clamping to [0, 512] GB (audit: "no validation").
  512 GB is a generous upper bound covering multi-GPU HPC nodes.
- KEPT: Ollama stdout parsing as-is (audit: "naive split"). The format
  is "NAME  ID  SIZE  MODIFIED" with tab/space separation. The split()[0]
  approach has been stable across ollama versions. CSV parsing would add
  complexity for no gain since ollama has no CSV output mode.
- KEPT: subprocess calls with list args (audit: "shell=True risk"). The
  code already uses shell=False (subprocess.run default) with list args.
  The audit's concern is a misread — no shell=True exists in this file.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("inference_difference.hardware")


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int = 0
    name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    compute_capability: str = ""
    driver_version: str = ""


@dataclass
class HardwareProfile:
    """Complete hardware profile of this machine.

    Used by the router to determine which local models can run
    and to estimate performance characteristics.

    Attributes:
        cpu_count: Number of logical CPU cores.
        cpu_name: CPU model name.
        ram_total_gb: Total system RAM in GB.
        ram_available_gb: Currently available RAM in GB.
        gpus: List of detected GPUs.
        total_vram_gb: Sum of all GPU VRAM.
        available_vram_gb: Sum of all free GPU VRAM.
        has_gpu: Whether any usable GPU was detected.
        os_name: Operating system name.
        platform_arch: CPU architecture (x86_64, arm64, etc.).
        ollama_available: Whether ollama CLI is accessible.
        ollama_models: List of locally available ollama model tags.
    """

    cpu_count: int = 0
    cpu_name: str = ""
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpus: List[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0
    available_vram_gb: float = 0.0
    has_gpu: bool = False
    os_name: str = ""
    platform_arch: str = ""
    ollama_available: bool = False
    ollama_models: List[str] = field(default_factory=list)

    def can_run_model(self, min_vram_gb: float, min_ram_gb: float) -> bool:
        """Whether this hardware can run a model with given requirements.

        For GPU models: checks VRAM.
        For CPU-only models: checks RAM.
        If VRAM requirement is 0, only checks RAM.
        """
        if min_vram_gb > 0 and self.has_gpu:
            return self.available_vram_gb >= min_vram_gb
        return self.ram_available_gb >= min_ram_gb

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and stats."""
        return {
            "cpu_count": self.cpu_count,
            "cpu_name": self.cpu_name,
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_available_gb": round(self.ram_available_gb, 1),
            "gpu_count": len(self.gpus),
            "gpus": [
                {"name": g.name, "vram_gb": g.vram_total_gb}
                for g in self.gpus
            ],
            "total_vram_gb": round(self.total_vram_gb, 1),
            "has_gpu": self.has_gpu,
            "os": self.os_name,
            "arch": self.platform_arch,
            "ollama_available": self.ollama_available,
            "ollama_model_count": len(self.ollama_models),
        }


def detect_hardware() -> HardwareProfile:
    """Detect available hardware resources.

    Safe to call at startup — never hangs, never modifies system state.
    Missing tools result in zeroed fields, not errors.
    """
    profile = HardwareProfile()

    # Platform basics
    profile.os_name = platform.system()
    profile.platform_arch = platform.machine()
    profile.cpu_count = os.cpu_count() or 1
    profile.cpu_name = _detect_cpu_name()

    # RAM
    profile.ram_total_gb, profile.ram_available_gb = _detect_ram()

    # GPU
    profile.gpus = _detect_gpus()
    profile.has_gpu = len(profile.gpus) > 0
    profile.total_vram_gb = sum(g.vram_total_gb for g in profile.gpus)
    profile.available_vram_gb = sum(g.vram_free_gb for g in profile.gpus)

    # Ollama
    profile.ollama_available = _detect_ollama()
    if profile.ollama_available:
        profile.ollama_models = _detect_ollama_models()

    logger.info(
        "Hardware detected: %d CPUs, %.1f GB RAM, %d GPUs (%.1f GB VRAM), "
        "ollama=%s (%d models)",
        profile.cpu_count,
        profile.ram_total_gb,
        len(profile.gpus),
        profile.total_vram_gb,
        profile.ollama_available,
        len(profile.ollama_models),
    )

    return profile


def _detect_cpu_name() -> str:
    """Get CPU model name."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception as e:
        logger.debug("CPU name detection failed: %s", e)
    return platform.processor() or "unknown"


def _detect_ram() -> tuple:
    """Detect total and available RAM in GB.

    Returns (total_gb, available_gb).
    """
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        value_kb = int(parts[1])
                        meminfo[key] = value_kb

                total = meminfo.get("MemTotal", 0) / (1024 * 1024)
                available = meminfo.get("MemAvailable", 0) / (1024 * 1024)
                return total, available

        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                total = int(result.stdout.strip()) / (1024 ** 3)
                # macOS doesn't have a clean "available" metric
                return total, total * 0.5  # Conservative estimate
    except Exception as e:
        logger.debug("RAM detection failed: %s", e)

    return 0.0, 0.0


def _clamp_vram(value_gb: float) -> float:
    """Clamp VRAM to sane bounds [0, 512] GB.

    512 GB covers multi-GPU HPC nodes (e.g., 8x A100 80GB).
    Negative or inf values from malformed nvidia-smi output are clamped.
    """
    if not (0.0 <= value_gb <= 512.0):
        logger.debug("VRAM value out of bounds (%.2f GB), clamping", value_gb)
    return max(0.0, min(value_gb, 512.0))


def _detect_gpus() -> List[GPUInfo]:
    """Detect GPUs via nvidia-smi (NVIDIA) and rocm-smi (AMD)."""
    gpus: List[GPUInfo] = []
    gpus.extend(_detect_nvidia_gpus())
    gpus.extend(_detect_amd_gpus())
    return gpus


def _detect_nvidia_gpus() -> List[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_gb=_clamp_vram(float(parts[2]) / 1024),
                    vram_free_gb=_clamp_vram(float(parts[3]) / 1024),
                    driver_version=parts[4],
                ))
        return gpus

    except Exception as e:
        logger.debug("NVIDIA GPU detection failed: %s", e)
        return []


def _detect_amd_gpus() -> List[GPUInfo]:
    """Detect AMD GPUs via rocm-smi.

    rocm-smi is the AMD equivalent of nvidia-smi for ROCm-supported GPUs.
    Falls back gracefully if rocm-smi is not installed or fails.
    """
    if not shutil.which("rocm-smi"):
        return []

    try:
        # Get GPU names and VRAM via rocm-smi --showmeminfo vram --json
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        import json
        data = json.loads(result.stdout)

        gpus = []
        idx = 0
        for card_key, card_data in data.items():
            if not card_key.startswith("card"):
                continue

            # rocm-smi reports VRAM in bytes
            total_bytes = int(card_data.get("VRAM Total Memory (B)", 0))
            used_bytes = int(card_data.get("VRAM Total Used Memory (B)", 0))
            total_gb = _clamp_vram(total_bytes / (1024 ** 3))
            free_gb = _clamp_vram((total_bytes - used_bytes) / (1024 ** 3))

            # Get GPU name from a separate call
            name = f"AMD GPU {idx}"
            try:
                id_result = subprocess.run(
                    ["rocm-smi", "--showproductname", "--json"],
                    capture_output=True, text=True, timeout=5,
                )
                if id_result.returncode == 0:
                    id_data = json.loads(id_result.stdout)
                    card_id_data = id_data.get(card_key, {})
                    name = card_id_data.get(
                        "Card Series", card_id_data.get("Card Model", name)
                    )
            except Exception:
                pass

            gpus.append(GPUInfo(
                index=idx,
                name=name,
                vram_total_gb=total_gb,
                vram_free_gb=free_gb,
                driver_version="rocm",
            ))
            idx += 1

        return gpus

    except Exception as e:
        logger.debug("AMD GPU detection failed: %s", e)
        return []


def _detect_ollama() -> bool:
    """Check if ollama is available."""
    return shutil.which("ollama") is not None


def _detect_ollama_models() -> List[str]:
    """List locally available ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                # Format: "NAME    ID    SIZE    MODIFIED"
                model_name = line.split()[0]
                if model_name:
                    models.append(model_name)
        return models

    except Exception as e:
        logger.debug("Ollama model listing failed: %s", e)
        return []
