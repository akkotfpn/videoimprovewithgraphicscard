# -*- mode: python ; coding: utf-8 -*-
"""
Gera a pasta portátil dist/VideoImprove com dois executáveis que compartilham _internal/:
  VideoImprove.exe       interface (sem console)
  videoimprove-cli.exe   engine (console; a interface chama sem abrir janela)

Variante pela variável VIDEOIMPROVE_FLAVOR:
  universal  onnxruntime-directml  (qualquer GPU DirectX 12: Intel, AMD, NVIDIA)
  nvidia     onnxruntime-gpu + CUDA/cuDNN dos pacotes pip nvidia-*

FFmpeg (bin/) e modelos (models/) são copiados pelo workflow depois do PyInstaller.
Uso:  pyinstaller packaging/videoimprove.spec --noconfirm
"""
import importlib.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).parent
FLAVOR = os.environ.get("VIDEOIMPROVE_FLAVOR", "universal")
ICON = str(ROOT / "packaging" / "icon.ico")

# ---- binários do ONNX Runtime (inclui DirectML.dll ou os providers CUDA/TensorRT) ----
engine_binaries = collect_dynamic_libs("onnxruntime")

# ---- CUDA/cuDNN dos wheels nvidia-* (mantém o layout nvidia/<pacote>/bin) ----
if FLAVOR == "nvidia":
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("pacotes nvidia-* não encontrados: instale .[nvidia]")
    for base in spec.submodule_search_locations:
        base = Path(base)
        for dll in base.rglob("*.dll"):
            engine_binaries.append((str(dll), str(dll.parent.relative_to(base.parent))))
    print(f"[spec] DLLs NVIDIA: {sum(1 for b in engine_binaries if 'nvidia' in b[1])}")

COMMON_EXCLUDES = ["tkinter", "_tkinter", "matplotlib", "torch", "IPython", "pytest", "_pytest",
                   "setuptools", "pip"]

a_gui = Analysis(
    [str(ROOT / "packaging" / "gui_entry.py")],
    pathex=[str(ROOT)],
    excludes=COMMON_EXCLUDES + ["onnxruntime",
                                "PySide6.QtNetwork", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtPdf",
                                "PySide6.QtOpenGL", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDBus",
                                "PySide6.QtConcurrent", "PySide6.QtXml", "PySide6.QtSvg"],
    noarchive=False,
)
a_cli = Analysis(
    [str(ROOT / "packaging" / "cli_entry.py")],
    pathex=[str(ROOT)],
    binaries=engine_binaries,
    hiddenimports=["onnxruntime", "PIL.Image"],
    excludes=COMMON_EXCLUDES + ["PySide6", "shiboken6", "gui"],
    noarchive=False,
)

pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)

exe_gui = EXE(
    pyz_gui, a_gui.scripts, [],
    exclude_binaries=True, name="VideoImprove", console=False, icon=ICON,
    upx=False, version=str(ROOT / "packaging" / "version_info.txt"),
)
exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True, name="videoimprove-cli", console=True, icon=ICON,
    upx=False, version=str(ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe_gui, a_gui.binaries, a_gui.datas,
    exe_cli, a_cli.binaries, a_cli.datas,
    upx=False, name="VideoImprove",
)
