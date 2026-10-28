"""
Detecção de hardware e escolha de perfil (preset).

Estratégia em camadas, da fonte mais confiável para a menos confiável:
  1. PyTorch (CUDA/ROCm/XPU)   -> VRAM exata e compute capability
  2. nvidia-smi / rocm-smi     -> funciona mesmo sem PyTorch instalado
  3. Windows (PowerShell/WMI)  -> pega Intel Arc / AMD sem ROCm
  4. vulkaninfo                -> qualquer GPU com driver Vulkan (caminho NCNN)
  5. ONNX Runtime providers    -> o que o runtime realmente consegue usar
Nada aqui é obrigatório: cada sonda falha em silêncio e o resultado final
cai para CPU se nada for encontrado.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum


class Vendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    CPU = "cpu"


class Backend(str, Enum):
    TENSORRT = "tensorrt"      # NVIDIA, engine compilado (FP16/FP8)
    CUDA = "cuda"              # PyTorch/ORT CUDA
    ROCM = "rocm"              # PyTorch ROCm / ORT MIGraphX
    DIRECTML = "directml"      # ORT DirectML (Windows, qualquer GPU DX12)
    OPENVINO = "openvino"      # Intel
    NCNN_VULKAN = "ncnn_vulkan"  # qualquer GPU com Vulkan, leve
    CPU = "cpu"


class Preset(str, Enum):
    ULTRA = "ultra"            # TensorRT, FP8/FP16, modelos pesados, tiles grandes
    EQUILIBRADO = "equilibrado"  # FP16, modelos médios
    ECONOMICO = "economico"    # NCNN/ONNX leve, tiling agressivo
    CPU = "cpu"                # filtros clássicos FFmpeg + modelo compact


@dataclass
class GPUInfo:
    name: str
    vendor: Vendor
    vram_mb: int = 0
    compute_capability: tuple[int, int] | None = None  # só NVIDIA
    driver: str = ""
    source: str = ""  # qual sonda encontrou

    @property
    def arch(self) -> str:
        """Geração NVIDIA a partir do compute capability."""
        if self.vendor != Vendor.NVIDIA or not self.compute_capability:
            return "desconhecida"
        major, minor = self.compute_capability
        return {
            (12, 0): "Blackwell (RTX 50)",
            (8, 9): "Ada (RTX 40)",
            (8, 6): "Ampere (RTX 30)",
            (8, 0): "Ampere (A100)",
            (7, 5): "Turing (RTX 20 / GTX 16)",
        }.get((major, minor), f"sm_{major}{minor}")

    @property
    def supports_fp8(self) -> bool:
        # Tensor Cores FP8: Ada (8.9) e Blackwell (10.x/12.x)
        cc = self.compute_capability
        return bool(cc and cc >= (8, 9))

    @property
    def supports_fp16_tensor(self) -> bool:
        cc = self.compute_capability
        if self.vendor == Vendor.NVIDIA:
            return bool(cc and cc >= (7, 0))
        return self.vendor in (Vendor.AMD, Vendor.INTEL, Vendor.APPLE)


@dataclass
class HardwareProfile:
    os: str
    cpu: str
    cpu_threads: int
    ram_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)
    torch_backend: str | None = None
    ort_providers: list[str] = field(default_factory=list)
    vulkan_available: bool = False
    ffmpeg_hwaccels: list[str] = field(default_factory=list)

    @property
    def best_gpu(self) -> GPUInfo | None:
        return max(self.gpus, key=lambda g: g.vram_mb, default=None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: float = 8.0) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _vendor_from_name(name: str) -> Vendor:
    n = name.lower()
    if any(k in n for k in ("nvidia", "geforce", "rtx", "quadro", "tesla")):
        return Vendor.NVIDIA
    if any(k in n for k in ("amd", "radeon", "instinct")):
        return Vendor.AMD
    if any(k in n for k in ("intel", " arc", "iris", "uhd")):
        return Vendor.INTEL
    if "apple" in n:
        return Vendor.APPLE
    return Vendor.CPU


def _ram_mb() -> int:
    try:
        import psutil  # opcional
        return psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        pass
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            m = re.search(r"MemTotal:\s+(\d+)", f.read())
            return int(m.group(1)) // 1024 if m else 0
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        return int(out.strip()) // (1024 * 1024) if out.strip().isdigit() else 0
    return 0


def _merge(gpus: list[GPUInfo], new: GPUInfo) -> None:
    """Evita duplicar a mesma placa vinda de sondas diferentes; mantém o dado mais rico."""
    for g in gpus:
        if g.vendor == new.vendor and (g.name.lower() in new.name.lower()
                                       or new.name.lower() in g.name.lower()):
            g.vram_mb = max(g.vram_mb, new.vram_mb)
            g.compute_capability = g.compute_capability or new.compute_capability
            g.driver = g.driver or new.driver
            return
    gpus.append(new)


# --------------------------------------------------------------------------- #
# sondas
# --------------------------------------------------------------------------- #
def probe_torch(gpus: list[GPUInfo]) -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            _merge(gpus, GPUInfo(
                name=p.name,
                vendor=Vendor.AMD if backend == "rocm" else Vendor.NVIDIA,
                vram_mb=p.total_memory // (1024 * 1024),
                compute_capability=(p.major, p.minor) if backend == "cuda" else None,
                source=f"torch-{backend}"))
        return backend
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            p = torch.xpu.get_device_properties(i)
            _merge(gpus, GPUInfo(p.name, Vendor.INTEL, p.total_memory // 2**20, source="torch-xpu"))
        return "xpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _merge(gpus, GPUInfo("Apple GPU", Vendor.APPLE, _ram_mb() // 2, source="torch-mps"))
        return "mps"
    return None


def probe_nvidia_smi(gpus: list[GPUInfo]) -> None:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        cc = None
        if len(parts) >= 4 and re.match(r"^\d+\.\d+$", parts[3]):
            cc = tuple(int(x) for x in parts[3].split("."))
        _merge(gpus, GPUInfo(parts[0], Vendor.NVIDIA, int(float(parts[1])), cc, parts[2],
                             source="nvidia-smi"))


def probe_rocm_smi(gpus: list[GPUInfo]) -> None:
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return
    for card in data.values():
        if not isinstance(card, dict):
            continue
        name = card.get("Card series") or card.get("Card SKU") or "AMD GPU"
        total = card.get("VRAM Total Memory (B)") or 0
        _merge(gpus, GPUInfo(name, Vendor.AMD, int(total) // 2**20, source="rocm-smi"))


def probe_windows(gpus: list[GPUInfo]) -> None:
    if platform.system() != "Windows":
        return
    # AdapterRAM do WMI é uint32 (trava em 4 GB). O registro tem o valor real em qwMemorySize.
    ps = r"""
    Get-ItemProperty 'HKLM:\SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue |
      Where-Object { $_.DriverDesc } |
      ForEach-Object { [pscustomobject]@{ name=$_.DriverDesc; mem=$_.'HardwareInformation.qwMemorySize'; drv=$_.DriverVersion } } |
      ConvertTo-Json -Compress
    """
    out = _run(["powershell", "-NoProfile", "-Command", ps], timeout=15)
    if not out.strip():
        return
    try:
        items = json.loads(out)
    except json.JSONDecodeError:
        return
    for it in items if isinstance(items, list) else [items]:
        name = it.get("name") or ""
        vendor = _vendor_from_name(name)
        if vendor == Vendor.CPU or "basic" in name.lower():
            continue
        mem = it.get("mem") or 0
        _merge(gpus, GPUInfo(name, vendor, int(mem) // 2**20 if mem else 0,
                             driver=it.get("drv") or "", source="windows-registry"))


def probe_vulkan(gpus: list[GPUInfo]) -> bool:
    out = _run(["vulkaninfo", "--summary"])
    if not out:
        return False
    for m in re.finditer(r"deviceName\s*=\s*(.+)", out):
        name = m.group(1).strip()
        vendor = _vendor_from_name(name)
        if vendor != Vendor.CPU:  # ignora llvmpipe/SwiftShader
            _merge(gpus, GPUInfo(name, vendor, source="vulkan"))
    return True


def probe_onnxruntime() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except ImportError:
        return []


def probe_ffmpeg_hwaccels() -> list[str]:
    out = _run(["ffmpeg", "-hide_banner", "-hwaccels"])
    return [l.strip() for l in out.splitlines()[1:] if l.strip()]


def detect_hardware() -> HardwareProfile:
    gpus: list[GPUInfo] = []
    torch_backend = probe_torch(gpus)
    probe_nvidia_smi(gpus)
    probe_rocm_smi(gpus)
    probe_windows(gpus)
    vulkan = probe_vulkan(gpus)
    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
        cpu_threads=os.cpu_count() or 1,
        ram_mb=_ram_mb(),
        gpus=gpus,
        torch_backend=torch_backend,
        ort_providers=probe_onnxruntime(),
        vulkan_available=vulkan,
        ffmpeg_hwaccels=probe_ffmpeg_hwaccels(),
    )


# --------------------------------------------------------------------------- #
# decisão de perfil
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionPlan:
    preset: Preset
    backend: Backend
    precision: str           # fp8 | fp16 | fp32 | int8
    tile_size: int           # 0 = sem tiling
    batch_size: int
    sr_model: str
    face_model: str | None
    interp_model: str | None
    denoise: str
    nvenc: bool
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)


def choose_plan(hw: HardwareProfile, force: Preset | None = None) -> ExecutionPlan:
    g = hw.best_gpu
    why: list[str] = []
    ort = set(hw.ort_providers)
    vram = g.vram_mb if g else 0

    # ---------- escolha do preset ----------
    if force:
        preset = force
        why.append(f"preset forçado pelo usuário: {force.value}")
    elif g is None:
        preset = Preset.CPU
        why.append("nenhuma GPU encontrada")
    elif g.vendor == Vendor.NVIDIA and g.supports_fp8 and vram >= 12_000:
        preset = Preset.ULTRA
        why.append(f"{g.arch} com {vram} MB e FP8")
    elif vram >= 8_000 and g.supports_fp16_tensor:
        preset = Preset.EQUILIBRADO
        why.append(f"{vram} MB com FP16")
    elif vram >= 2_000 or hw.vulkan_available:
        preset = Preset.ECONOMICO
        why.append(f"VRAM limitada ({vram} MB) ou só Vulkan disponível")
    else:
        preset = Preset.CPU
        why.append("GPU fraca demais")

    # ---------- escolha do backend (o que realmente está instalado) ----------
    if g and g.vendor == Vendor.NVIDIA and "TensorrtExecutionProvider" in ort and preset == Preset.ULTRA:
        backend = Backend.TENSORRT
    elif g and g.vendor == Vendor.NVIDIA and (hw.torch_backend == "cuda" or "CUDAExecutionProvider" in ort):
        backend = Backend.CUDA
    elif g and g.vendor == Vendor.AMD and (hw.torch_backend == "rocm" or "MIGraphXExecutionProvider" in ort):
        backend = Backend.ROCM
    elif g and g.vendor == Vendor.INTEL and "OpenVINOExecutionProvider" in ort:
        backend = Backend.OPENVINO
    elif "DmlExecutionProvider" in ort:
        backend = Backend.DIRECTML
    elif hw.vulkan_available and g:
        backend = Backend.NCNN_VULKAN
    else:
        backend = Backend.CPU
        if preset != Preset.CPU:
            why.append(f"GPU existe mas nenhum runtime acelerado instalado -> rebaixado para CPU")
            preset = Preset.CPU

    if preset == Preset.ULTRA and backend != Backend.TENSORRT:
        why.append("TensorRT ausente: Ultra roda em FP16 via CUDA")

    # ---------- parâmetros por preset ----------
    table = {
        Preset.ULTRA: dict(precision="fp8" if backend == Backend.TENSORRT else "fp16",
                           tile=0 if vram >= 16_000 else 768, batch=8,
                           sr="realbasicvsr_x4", face="codeformer", interp="rife_v4",
                           denoise="scunet"),
        Preset.EQUILIBRADO: dict(precision="fp16", tile=512, batch=4,
                                 sr="realesrgan_x4plus", face="gfpgan_v1.4",
                                 interp="rife_v4_lite", denoise="ffmpeg_nlmeans"),
        Preset.ECONOMICO: dict(precision="fp16" if backend != Backend.CPU else "fp32",
                               tile=256, batch=1, sr="realesr_general_x4v3",
                               face=None, interp="rife_ncnn", denoise="ffmpeg_hqdn3d"),
        Preset.CPU: dict(precision="int8", tile=192, batch=1, sr="realesr_general_x4v3",
                         face=None, interp=None, denoise="ffmpeg_hqdn3d"),
    }[preset]

    # tile ajustado pela VRAM livre real (heurística: ~ VRAM/12 px de lado para x4)
    tile = table["tile"]
    if tile and vram:
        tile = min(tile, max(128, (vram // 12) // 64 * 64))

    nvenc = any(h in hw.ffmpeg_hwaccels for h in ("cuda", "nvdec")) and bool(g and g.vendor == Vendor.NVIDIA)

    return ExecutionPlan(preset, backend, table["precision"], tile, table["batch"],
                         table["sr"], table["face"], table["interp"], table["denoise"],
                         nvenc, why)


if __name__ == "__main__":
    hw = detect_hardware()
    print("=== Hardware ===")
    print(hw.to_json())
    print("\n=== Plano escolhido ===")
    print(choose_plan(hw).to_json())
