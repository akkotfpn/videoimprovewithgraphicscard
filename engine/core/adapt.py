"""
Adapta o trabalho a cada vídeo: resolução em que a IA roda e qual modelo usar.

Por que existe: o custo da IA cresce com o número de pixels de ENTRADA. Um modelo pesado que roda
bem num VHS (720x480 = 0,35 MP) fica ~6x mais lento num arquivo 1080p (2,1 MP). E muitos "vídeos 1080p"
de família são fitas digitalizadas e salvas em 1080p: o detalhe real é de SD. Nesses, rodar a IA em
meia resolução (960x540) e ampliar 4x dá o mesmo resultado com 1/4 do trabalho.

1. detail_ratio(): mede quanto detalhe existe acima da metade da resolução. Reduz o quadro pela metade,
   volta ao tamanho original e compara (PSNR). Sem detalhe fino, quase nada se perde (PSNR alto).
   Calibração com vídeos sintéticos: HD nativo ~32 dB, 720p ampliado ~36, 540p ampliado ~41,
   VHS ampliado ~46. Acima de 40 dB, a resolução de trabalho cai pela metade.
2. choose_job(): escolhe o modelo de maior qualidade cuja velocidade estimada atinge a meta do preset,
   a partir do custo de cada modelo (multiplicações por pixel) e do desempenho estimado da GPU.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass, field

import numpy as np

from .hardware import Backend, ExecutionPlan, Preset
from .paths import no_window_flags

log = logging.getLogger("restaura.adapt")

HALF_DETAIL_PSNR = 40.0
# qualidade, do melhor para o mais leve
QUALITY_ORDER = ["realesrgan_x4plus", "realesrgan_x2plus", "realesr_general_x4v3"]
# velocidade mínima aceitável por preset (quadros/s); abaixo disso, usa um modelo mais leve
TARGET_FPS = {Preset.ULTRA: 6.0, Preset.EQUILIBRADO: 6.0, Preset.ECONOMICO: 0.0, Preset.CPU: 0.0}


@dataclass
class JobPlan:
    model: str | None            # chave do catálogo; None = sem IA (redimensionamento clássico)
    work_w: int
    work_h: int
    work_div: int                # 1 = resolução original; 2 = metade
    sr_scale: int                # ampliação aplicada pela IA sobre a resolução de trabalho
    out_w: int
    out_h: int
    est_fps: float | None
    detail_psnr: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _gray_frame(path: str, t: float, w: int, h: int) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
           "-vf", "bwdif=mode=0:deint=interlaced,hqdn3d=4:3:6:4,format=gray", "-f", "rawvideo", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, creationflags=no_window_flags()).stdout
    if len(raw) < w * h:
        return None
    return np.frombuffer(raw[: w * h], np.uint8).reshape(h, w).astype(np.float32)


def half_psnr(gray: np.ndarray) -> float:
    from PIL import Image
    h, w = gray.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    g = gray[:h2, :w2]
    small = g.reshape(h2 // 2, 2, w2 // 2, 2).mean((1, 3))
    up = np.asarray(Image.fromarray(small).resize((w2, h2), Image.BICUBIC))
    mse = float(np.mean((g - up) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def detail_psnr(path: str, width: int, height: int, duration: float, samples: int = 5) -> float | None:
    """Mediana do PSNR de meia resolução em quadros espalhados pelo vídeo."""
    times = [duration * (i + 1) / (samples + 1) for i in range(samples)] if duration > 1 else [0.0]
    vals = [half_psnr(g) for t in times if (g := _gray_frame(path, t, width, height)) is not None]
    return float(np.median(vals)) if vals else None


def estimate_tflops(plan: ExecutionPlan) -> float:
    """Desempenho efetivo aproximado em convoluções (TFLOP/s). Só para escolher modelo e estimar tempo."""
    if plan.backend == Backend.CPU or plan.gpu_name is None:
        return 0.3
    return {Preset.ULTRA: 20.0, Preset.EQUILIBRADO: 8.0, Preset.ECONOMICO: 2.5, Preset.CPU: 0.3}[plan.preset]


def est_fps(model: str, work_px: int, tflops: float) -> float:
    from ..models.catalog import CATALOG
    macs = CATALOG[model].gmac_per_px * 1e6 * work_px
    return tflops * 1e12 / (2 * macs)


def choose_job(plan: ExecutionPlan, width: int, height: int, scale: int, *, path: str | None = None,
               duration: float = 0.0, work: str = "auto", model: str | None = None,
               available: set[str] | None = None) -> JobPlan:
    """
    scale: ampliação final pedida sobre o arquivo original (1, 2 ou 4).
    work: "auto" (mede o detalhe), "original" ou "half".
    model: chave do catálogo forçada pelo usuário, ou None (automático).
    available: modelos instalados (None = todos).
    """
    from ..models.catalog import CATALOG
    notes: list[str] = []
    psnr = None
    div = 1
    if work == "half":
        div = 2
    elif work == "auto" and height >= 700 and path:
        psnr = detail_psnr(path, width, height, duration)
        if psnr is not None and psnr >= HALF_DETAIL_PSNR:
            div = 2
            notes.append(f"o vídeo tem pouco detalhe acima de {width // 2}x{height // 2} "
                         f"(PSNR {psnr:.1f} dB): a IA roda em meia resolução, ~4x mais rápido")
        elif psnr is not None:
            notes.append(f"vídeo com detalhe real de alta resolução (PSNR {psnr:.1f} dB)")
    total = scale * div
    if total > 4:          # 4x sobre meia resolução = 8x: nenhum modelo faz; mantém a original
        div, total = 1, scale
        notes.append("ampliação 4x usa a resolução original")
    work_w, work_h = width // div // 2 * 2, height // div // 2 * 2
    out_w, out_h = work_w * total, work_h * total
    work_px = work_w * work_h
    tflops = estimate_tflops(plan)

    if total == 1:
        return JobPlan(None, work_w, work_h, div, 1, out_w, out_h, None, psnr, notes + ["sem ampliação"])

    def usable(key: str) -> bool:
        return key in CATALOG and CATALOG[key].scale >= total and (available is None or key in available)

    if model:
        if model not in CATALOG:
            raise ValueError(f"modelo desconhecido: {model}")
        if CATALOG[model].scale < total:
            raise ValueError(f"o modelo {model} amplia no máximo {CATALOG[model].scale}x; "
                             f"aqui são necessários {total}x")
        chosen = model
    elif plan.sr_model == "classico":
        return JobPlan(None, work_w, work_h, div, total, out_w, out_h, None, psnr,
                       notes + ["sem GPU acelerada: filtros clássicos"])
    else:
        target = TARGET_FPS.get(plan.preset, 0.0)
        candidates = [k for k in QUALITY_ORDER if usable(k)]
        if plan.preset in (Preset.ECONOMICO, Preset.CPU):
            candidates = [k for k in candidates if k == "realesr_general_x4v3"] or candidates
        if not candidates:
            return JobPlan(None, work_w, work_h, div, total, out_w, out_h, None, psnr,
                           notes + ["nenhum modelo de IA instalado"])
        chosen = next((k for k in candidates if est_fps(k, work_px, tflops) >= target), candidates[-1])
        if chosen != candidates[0]:
            notes.append(f"modelo {chosen} para manter a velocidade em {work_w}x{work_h}")
    fps = est_fps(chosen, work_px, tflops)
    log.info("trabalho %dx%d (÷%d) -> %dx%d com %s (~%.1f q/s estimado)", work_w, work_h, div, out_w, out_h,
             chosen, fps)
    return JobPlan(chosen, work_w, work_h, div, total, out_w, out_h, round(fps, 2), psnr, notes)
