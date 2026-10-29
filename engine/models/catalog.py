"""
Catálogo de modelos: baixa os pesos oficiais, confere o SHA-256 e converte para ONNX
(FP32 + FP16) sem PyTorch. Os nomes batem com ExecutionPlan.sr_model.
"""
from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import realesrgan
from .pth_loader import state_dict

log = logging.getLogger("restaura.models")
_RE = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    description: str
    license: str
    sources: tuple[Source, ...]
    blend_default: float | None = None   # DNI entre as duas fontes (força do denoise)
    scale: int = 4                       # fator nativo
    gmac_per_px: float = 1.0             # custo: bilhões de multiplicações por pixel de ENTRADA / 1000
    mem_per_px: int = 2000               # memória de ativação estimada (bytes/pixel de entrada, FP16)


CATALOG: dict[str, ModelSpec] = {
    "realesr_general_x4v3": ModelSpec(
        "realesr_general_x4v3",
        "Real-ESRGAN compact (4x, 1,2 M parâmetros). Econômico/CPU; roda em integradas e GTX 16.",
        "BSD-3-Clause",
        (Source(f"{_RE}/v0.2.5.0/realesr-general-x4v3.pth",
                "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"),
         Source(f"{_RE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca")),
        blend_default=0.5, scale=4, gmac_per_px=1.2, mem_per_px=1200),
    "realesrgan_x2plus": ModelSpec(
        "realesrgan_x2plus",
        "Real-ESRGAN x2plus (2x nativo, 16,7 M parâmetros). Mais detalhe que o compact, ~4x mais "
        "leve que o x4plus para ampliar 2x.",
        "BSD-3-Clause",
        (Source(f"{_RE}/v0.2.1/RealESRGAN_x2plus.pth",
                "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"),),
        scale=2, gmac_per_px=4.5, mem_per_px=3000),
    "realesrgan_x4plus": ModelSpec(
        "realesrgan_x4plus",
        "Real-ESRGAN x4plus (4x, 16,7 M parâmetros). O mais pesado; vale para material SD em 4x.",
        "BSD-3-Clause",
        (Source(f"{_RE}/v0.1.0/RealESRGAN_x4plus.pth",
                "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"),),
        scale=4, gmac_per_px=17.9, mem_per_px=10000),
}


def spec_for(path: str | Path) -> ModelSpec | None:
    """Acha o ModelSpec pelo nome do arquivo (realesrgan_x2plus.fp16.onnx -> realesrgan_x2plus)."""
    stem = Path(path).name.split(".")[0]
    return CATALOG.get(stem)


def default_dir() -> Path:
    from ..core.paths import models_dir
    return models_dir()


def onnx_path(key: str, fp16: bool = False, models_dir: Path | None = None) -> Path:
    return (models_dir or default_dir()) / f"{key}{'.fp16' if fp16 else ''}.onnx"


def find(key_or_path: str, prefer_fp16: bool, models_dir: Path | None = None) -> Path | None:
    """Aceita um caminho .onnx ou um nome do catálogo instalado."""
    p = Path(key_or_path)
    if p.suffix == ".onnx" and p.exists():
        return p
    for fp16 in ((True, False) if prefer_fp16 else (False,)):
        cand = onnx_path(key_or_path, fp16, models_dir)
        if cand.exists():
            return cand
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(src: Source, dest_dir: Path) -> Path:
    dest = dest_dir / src.url.rsplit("/", 1)[-1]
    if dest.exists() and _sha256(dest) == src.sha256:
        return dest
    tmp = dest.with_suffix(".part")
    log.info("baixando %s", src.url)

    def hook(blocks, bs, total):
        if total > 0 and sys.stderr.isatty():
            print(f"\r  {min(100, blocks * bs * 100 // total):3d}%", end="", file=sys.stderr)

    urllib.request.urlretrieve(src.url, tmp, hook)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    digest = _sha256(tmp)
    if digest != src.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 não confere para {dest.name}: {digest}")
    tmp.replace(dest)
    return dest


def install(key: str, models_dir: Path | None = None, denoise: float | None = None,
            weights: list[Path] | None = None) -> list[Path]:
    """
    Baixa (ou usa `weights` locais), converte e grava <key>.onnx e <key>.fp16.onnx.
    denoise: só para o compact. 0 = preserva mais textura/ruído, 1 = limpa mais. Padrão 0,5.
    """
    spec = CATALOG[key]
    out_dir = models_dir or default_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "_weights"
    cache.mkdir(exist_ok=True)

    paths = weights or [_download(s, cache) for s in spec.sources]
    sds = [state_dict(p) for p in paths]
    sd = sds[0]
    if spec.blend_default is not None and len(sds) == 2:
        alpha = spec.blend_default if denoise is None else denoise
        sd = realesrgan.blend(sds[0], sds[1], alpha)
        log.info("denoise (DNI) = %.2f", alpha)

    written = []
    for fp16 in (False, True):
        dest = onnx_path(key, fp16, out_dir)
        realesrgan.export(sd, dest, fp16=fp16)
        written.append(dest)
        log.info("gerado %s (%.1f MB)", dest.name, dest.stat().st_size / 2**20)
    return written
