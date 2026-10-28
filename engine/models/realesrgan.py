"""
Conversão Real-ESRGAN (.pth) -> ONNX sem PyTorch.

Arquiteturas (iguais às do BasicSR):
  SRVGGNetCompact  realesr-general-x4v3: 32 convs 3x3 + PReLU, PixelShuffle, + upsample nearest da entrada.
                   ~1,2 M parâmetros. É o modelo dos presets Econômico e CPU.
  RRDBNet          RealESRGAN_x4plus: 23 blocos RRDB. ~16,7 M parâmetros. Preset Equilibrado.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .onnx_writer import GraphBuilder


def detect_arch(sd: dict[str, np.ndarray]) -> str:
    if "conv_first.weight" in sd and any(k.startswith("body.0.rdb1") for k in sd):
        return "rrdbnet"
    if "body.0.weight" in sd:
        return "srvgg_compact"
    raise ValueError("arquitetura não reconhecida")


def blend(sd_a: dict, sd_b: dict, alpha: float) -> dict:
    """
    Interpolação de pesos (DNI), como o --denoise_strength do Real-ESRGAN:
    alpha=1 -> só realesr-general-x4v3 (denoise forte); alpha=0 -> só o wdn (denoise fraco).
    """
    return {k: alpha * sd_a[k] + (1 - alpha) * sd_b[k] for k in sd_a}


# ---------------- SRVGGNetCompact ----------------
def _compact_layers(sd):
    idx = sorted({int(m.group(1)) for k in sd if (m := re.match(r"body\.(\d+)\.", k))})
    last = idx[-1]
    scale = int(np.sqrt(sd[f"body.{last}.weight"].shape[0] // 3))
    return idx, last, scale


def export_compact(sd: dict, path: str | Path, fp16: bool = False) -> int:
    idx, last, scale = _compact_layers(sd)
    g = GraphBuilder(fp16)
    x = "input"
    h = x
    for i in idx:
        w = sd[f"body.{i}.weight"]
        if w.ndim == 4:
            h = g.conv(h, w, sd[f"body.{i}.bias"], f"body.{i}")
        else:
            h = g.prelu(h, w, f"body.{i}.weight")
    h = g.pixel_shuffle(h, scale)
    y = g.add(h, g.upsample_nearest(x, scale))
    g.save(path, x, y)
    return scale


def run_compact_numpy(sd: dict, img: np.ndarray) -> np.ndarray:
    """Referência em NumPy puro (lenta) para validar o grafo ONNX. img: CHW float32."""
    idx, last, scale = _compact_layers(sd)

    def conv(x, w, b):
        c, hh, ww = x.shape
        p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        cols = np.stack([p[:, i:i + hh, j:j + ww] for i in range(3) for j in range(3)], 1)  # C,9,H,W
        return np.einsum("ocij,cijhw->ohw", w.reshape(w.shape[0], c, 3, 3),
                         cols.reshape(c, 3, 3, hh, ww)) + b[:, None, None]

    h = img
    for i in idx:
        w = sd[f"body.{i}.weight"]
        if w.ndim == 4:
            h = conv(h, w, sd[f"body.{i}.bias"])
        else:
            h = np.where(h >= 0, h, h * w[:, None, None])
    c, hh, ww = h.shape
    oc = c // (scale * scale)
    h = h.reshape(oc, scale, scale, hh, ww).transpose(0, 3, 1, 4, 2).reshape(oc, hh * scale, ww * scale)
    return h + img.repeat(scale, 1).repeat(scale, 2)


# ---------------- RRDBNet ----------------
def export_rrdbnet(sd: dict, path: str | Path, fp16: bool = False) -> int:
    g = GraphBuilder(fp16)
    c = lambda x, name: g.conv(x, sd[name + ".weight"], sd[name + ".bias"], name)  # noqa: E731
    x = "input"
    feat = c(x, "conv_first")
    n_blocks = 1 + max(int(m.group(1)) for k in sd if (m := re.match(r"body\.(\d+)\.rdb", k)))
    body = feat
    for b in range(n_blocks):
        block_in = body
        for r in (1, 2, 3):
            p = f"body.{b}.rdb{r}"
            feats = [body]
            for k in range(1, 5):
                feats.append(g.lrelu(c(g.concat(feats) if len(feats) > 1 else feats[0], f"{p}.conv{k}")))
            x5 = c(g.concat(feats), f"{p}.conv5")
            body = g.add(g.scale(x5, 0.2), body)
        body = g.add(g.scale(body, 0.2), block_in)
    feat = g.add(feat, c(body, "conv_body"))
    feat = g.lrelu(c(g.upsample_nearest(feat, 2), "conv_up1"))
    feat = g.lrelu(c(g.upsample_nearest(feat, 2), "conv_up2"))
    y = c(g.lrelu(c(feat, "conv_hr")), "conv_last")
    g.save(path, x, y)
    return 4


def export(sd: dict, path: str | Path, fp16: bool = False) -> int:
    arch = detect_arch(sd)
    return (export_compact if arch == "srvgg_compact" else export_rrdbnet)(sd, path, fp16)
