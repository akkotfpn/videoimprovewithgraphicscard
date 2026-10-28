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
    def integrated(self) -> bool:
        """Vídeo integrado (usa RAM do sistema): Intel UHD/Iris/Arc Graphics, AMD Radeon Graphics/780M."""
        n = self.name.lower()
        if self.vendor == Vendor.INTEL:
            # Arc dedicada tem modelo com letra+3 dígitos (A770, B580); a integrada é só "Arc Graphics"
            return not re.search(r"arc\S*\s+(pro\s+)?[ab]\d{3}", n)
        if self.vendor == Vendor.AMD:
            dedicated = re.search(r"\brx\b|radeon pro w|instinct|radeon vii|\br9\b|\br7\b", n)
            return not dedicated
        return False

    @property
    def is_gtx16(self) -> bool:
        return self.vendor == Vendor.NVIDIA and "gtx 16" in self.name.lower()

    @property
    def has_tensor_cores(self) -> bool:
        """Unidades de matriz para FP16 rápido. A GTX 16 é Turing mas NÃO tem Tensor Cores."""
        if self.vendor == Vendor.NVIDIA:
            cc = self.compute_capability
            return bool(cc and cc >= (7, 0)) and not self.is_gtx16
        if self.vendor == Vendor.INTEL:
            return not self.integrated  # XMX só nas Arc dedicadas
        if self.vendor == Vendor.AMD:
            return not self.integrated  # WMMA a partir de RDNA 3
        return self.vendor == Vendor.APPLE

    @property
    def supports_fp8(self) -> bool:
        # Tensor Cores FP8: Ada (8.9) e Blackwell (10.x/12.x)
        cc = self.compute_capability
        return self.vendor == Vendor.NVIDIA and bool(cc and cc >= (8, 9))

    def effective_vram_mb(self, ram_mb: int) -> int:
        """Integradas usam RAM compartilhada: o Windows libera até metade da RAM; usamos uma fatia segura."""
        if self.integrated:
            return max(self.vram_mb, min(ram_mb // 4, 4096))
        return self.vram_mb


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
    notes: list[str] = field(default_factory=list)   # avisos da detecção (vão para os motivos do plano)

    @property
    def best_gpu(self) -> GPUInfo | None:
        # dedicada sempre ganha da integrada (notebook com Iris Xe + GTX 1650, por exemplo)
        return max(self.gpus, key=lambda g: (not g.integrated, g.effective_vram_mb(self.ram_mb)),
                   default=None)

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


def probe_tensorrt() -> bool:
    """
    O ONNX Runtime lista o TensorrtExecutionProvider mesmo sem as DLLs do TensorRT instaladas;
    aí a sessão falha ou cai para CUDA. Aqui conferimos se a biblioteca carrega de verdade.
    Aceita TensorRT no PATH ou o pacote pip `tensorrt` (pasta tensorrt_libs).
    """
    import ctypes
    import ctypes.util
    try:
        import tensorrt_libs  # pacote pip tensorrt-cu12/cu13
        libdir = os.path.dirname(tensorrt_libs.__file__)
        os.environ["PATH"] = libdir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(libdir)
    except ImportError:
        pass
    names = ["nvinfer_10", "nvinfer"] if platform.system() == "Windows" else ["nvinfer"]
    for n in names:
        path = ctypes.util.find_library(n)
        if not path:
            continue
        try:
            ctypes.CDLL(path)
            return True
        except OSError:
            continue
    return False


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
    providers = probe_onnxruntime()
    notes: list[str] = []
    if "TensorrtExecutionProvider" in providers and not probe_tensorrt():
        providers.remove("TensorrtExecutionProvider")
        notes.append("TensorRT não instalado (só o provider do ONNX Runtime): usando CUDA")
    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
        cpu_threads=os.cpu_count() or 1,
        ram_mb=_ram_mb(),
        gpus=gpus,
        torch_backend=torch_backend,
        ort_providers=providers,
        vulkan_available=vulkan,
        ffmpeg_hwaccels=probe_ffmpeg_hwaccels(),
        notes=notes,
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


def choose_plan(hw: HardwareProfile, force: Preset | None = None, gpu_index: int | None = None) -> ExecutionPlan:
    """
    Meta do projeto: qualquer PC mediano tem que conseguir restaurar (GTX 1660, Intel Iris Xe,
    AMD Radeon 680M/780M). Por isso o Econômico é um caminho de primeira classe, não um "modo de pena":
    mesmo modelo de IA (compact), só que com tiles menores e sem os estágios caros.
    """
    why: list[str] = list(hw.notes)
    if gpu_index is not None:
        if not 0 <= gpu_index < len(hw.gpus):
            raise ValueError(f"GPU {gpu_index} não existe; detectadas: {len(hw.gpus)}")
        g = hw.gpus[gpu_index]
        why.append(f"GPU escolhida pelo usuário: {g.name}")
    else:
        g = hw.best_gpu
    ort = set(hw.ort_providers)
    vram = g.effective_vram_mb(hw.ram_mb) if g else 0
    if g and g.integrated:
        why.append(f"vídeo integrado: usando ~{vram} MB da RAM compartilhada")

    # ---------- escolha do preset ----------
    if force:
        preset = force
        why.append(f"preset forçado pelo usuário: {force.value}")
    elif g is None:
        preset = Preset.CPU
        why.append("nenhuma GPU encontrada")
    elif g.supports_fp8 and vram >= 11_500:   # RTX 4070+/5070+ (12 GB reportam ~11,9-12,2 GB)
        preset = Preset.ULTRA
        why.append(f"{g.arch} com {vram} MB e FP8")
    elif g.has_tensor_cores and vram >= 6_000:
        preset = Preset.EQUILIBRADO
        why.append(f"{vram} MB com unidades de matriz FP16")
    elif not g.integrated and vram >= 3_000:
        preset = Preset.ECONOMICO
        why.append(f"GPU dedicada sem Tensor Cores ou com pouca VRAM ({vram} MB)")
    elif g.integrated:
        preset = Preset.ECONOMICO
        why.append("integrada roda o modelo compact com tiles pequenos")
    elif hw.vulkan_available:
        preset = Preset.ECONOMICO
        why.append(f"VRAM baixa ({vram} MB), caminho Vulkan")
    else:
        preset = Preset.CPU
        why.append("GPU fraca demais")

    # ---------- escolha do backend (o que realmente está instalado) ----------
    nvidia = g is not None and g.vendor == Vendor.NVIDIA
    if nvidia and "TensorrtExecutionProvider" in ort and preset == Preset.ULTRA:
        backend = Backend.TENSORRT
    elif nvidia and (hw.torch_backend == "cuda" or "CUDAExecutionProvider" in ort):
        backend = Backend.CUDA
    elif g and g.vendor == Vendor.AMD and (hw.torch_backend == "rocm" or "MIGraphXExecutionProvider" in ort):
        backend = Backend.ROCM
    elif g and g.vendor == Vendor.INTEL and "OpenVINOExecutionProvider" in ort:
        backend = Backend.OPENVINO
    elif g and "DmlExecutionProvider" in ort:
        # DirectML: o caminho universal no Windows (GTX 1660, Intel UHD/Iris/Arc, AMD integrada ou RX)
        backend = Backend.DIRECTML
    elif hw.vulkan_available and g:
        backend = Backend.NCNN_VULKAN
        why.append("estágio NCNN-Vulkan ainda não implementado: a IA roda na CPU. "
                   "No Windows, instale o runtime directml para usar esta GPU")
    else:
        backend = Backend.CPU
        if g and preset != Preset.CPU:
            why.append("GPU existe mas nenhum runtime acelerado instalado -> CPU "
                       "(instale onnxruntime-directml no Windows)")
        preset = Preset.CPU

    if preset == Preset.ULTRA and backend not in (Backend.TENSORRT, Backend.CUDA):
        preset = Preset.EQUILIBRADO
        why.append("Ultra precisa de CUDA/TensorRT; com este runtime fica no Equilibrado")
    elif preset == Preset.ULTRA and backend != Backend.TENSORRT:
        why.append("TensorRT ausente: Ultra roda em FP16 via CUDA")

    # ---------- parâmetros por preset ----------
    table = {
        # FP8 é meta (precisa de quantização calibrada e validação de qualidade); hoje roda FP16
        Preset.ULTRA: dict(precision="fp16",
                           tile=0 if vram >= 16_000 else 768, batch=8,
                           sr="realbasicvsr_x4", face="codeformer", interp="rife_v4",
                           denoise="scunet"),
        Preset.EQUILIBRADO: dict(precision="fp16", tile=512, batch=4,
                                 sr="realesrgan_x4plus", face="gfpgan_v1.4",
                                 interp="rife_v4_lite", denoise="ffmpeg_nlmeans"),
        # Econômico: mesmo modelo compact do CPU, acelerado; rosto e RIFE desligados por padrão
        # (o usuário pode ligar; bwdif mode=1 já entrega 59,94 fps reais sem custo de IA)
        Preset.ECONOMICO: dict(precision="fp16" if backend != Backend.CPU else "fp32",
                               tile=384 if vram >= 3_000 else 256, batch=1,
                               sr="realesr_general_x4v3", face=None, interp=None,
                               denoise="ffmpeg_hqdn3d"),
        # CPU: sem blocos (RAM sobra e blocos só somam sobreposição); o pipeline tem 1 frame por vez
        # IA na CPU leva dezenas de horas por hora de fita: padrão é filtro clássico,
        # IA só se o usuário pedir (--model), para trechos curtos
        Preset.CPU: dict(precision="fp32", tile=0, batch=1, sr="classico",
                         face=None, interp=None, denoise="ffmpeg_hqdn3d"),
    }[preset]

    # tile limitado pela VRAM (heurística: ~VRAM/12 px de lado para x4)
    tile = table["tile"]
    if tile and vram:
        tile = min(tile, max(128, (vram // 12) // 64 * 64))

    nvenc = any(h in hw.ffmpeg_hwaccels for h in ("cuda", "nvdec")) and nvidia

    return ExecutionPlan(preset, backend, table["precision"], tile, table["batch"],
                         table["sr"], table["face"], table["interp"], table["denoise"],
                         nvenc, why)


if __name__ == "__main__":
    hw = detect_hardware()
    print("=== Hardware ===")
    print(hw.to_json())
    print("\n=== Plano escolhido ===")
    print(choose_plan(hw).to_json())
