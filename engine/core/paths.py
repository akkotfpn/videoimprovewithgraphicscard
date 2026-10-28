"""
Onde ficam as coisas, tanto rodando do código-fonte quanto no app portátil (PyInstaller).

App portátil (pasta extraída do zip):
    VideoImprove/
      VideoImprove.exe         interface
      videoimprove-cli.exe     engine (processo filho da interface; também serve de CLI)
      bin/                     ffmpeg.exe, ffprobe.exe e DLLs
      models/                  modelos ONNX
      _internal/               Python, Qt, ONNX Runtime e (versão NVIDIA) CUDA/cuDNN

Escrita (configurações, cache do TensorRT, arquivos temporários): na própria pasta do app se der
para escrever nela (modo portátil de verdade, ex.: pendrive); senão em %LOCALAPPDATA%\\VideoImprove.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

APP_NAME = "VideoImprove"


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


@lru_cache(maxsize=None)
def app_dir() -> Path:
    if frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]  # raiz do repositório


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


@lru_cache(maxsize=None)
def data_dir() -> Path:
    """Pasta gravável para configurações e caches."""
    local = app_dir() / "data" if frozen() else app_dir() / ".data"
    if _writable(local):
        return local
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def bin_dir() -> Path:
    return app_dir() / "bin"


def models_dir() -> Path:
    env = os.environ.get("VIDEOIMPROVE_MODELS")
    if env:
        return Path(env)
    bundled = app_dir() / "models"
    if bundled.exists() or not frozen():
        return bundled
    return data_dir() / "models"


def cache_dir() -> Path:
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cli_command() -> list[str]:
    """Como a interface chama a engine (processo separado: um travamento do driver não fecha a janela)."""
    if frozen():
        exe = app_dir() / ("videoimprove-cli.exe" if os.name == "nt" else "videoimprove-cli")
        return [str(exe)]
    return [sys.executable, "-m", "engine"]


_env_ready = False


def setup_environment() -> None:
    """Coloca o FFmpeg embutido no PATH e registra as DLLs do CUDA/cuDNN empacotadas (versão NVIDIA)."""
    global _env_ready
    if _env_ready:
        return
    _env_ready = True
    extra: list[Path] = []
    if bin_dir().is_dir():
        extra.append(bin_dir())
    # wheels nvidia-* ficam em site-packages/nvidia/<lib>/bin; no app, dentro de _internal/nvidia
    roots = [Path(getattr(sys, "_MEIPASS", ""))] if frozen() else []
    for root in roots:
        nv = root / "nvidia"
        if nv.is_dir():
            extra += sorted({p.parent for p in nv.rglob("*.dll")})
        ort_capi = root / "onnxruntime" / "capi"
        if ort_capi.is_dir():
            extra.append(ort_capi)
    if not extra:
        return
    os.environ["PATH"] = os.pathsep.join(str(p) for p in extra) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        for p in extra:
            try:
                os.add_dll_directory(str(p))
            except OSError:
                pass


def no_window_flags() -> int:
    """Evita abrir janelas de console ao chamar ffmpeg/ffprobe a partir da interface no Windows."""
    import subprocess
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)
