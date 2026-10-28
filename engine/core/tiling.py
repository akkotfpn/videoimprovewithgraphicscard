"""Tiling com sobreposição: processa imagens maiores que a VRAM em blocos e costura sem emenda visível."""
from __future__ import annotations

from typing import Callable

import numpy as np


def process_tiled(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray],
                  scale: int, tile: int, overlap: int = 16) -> np.ndarray:
    """
    img: HxWxC (float32 ou uint8). fn recebe um tile hxwxC e devolve (h*scale)x(w*scale)xC.
    Cada tile é processado com uma margem de `overlap` px; só o miolo é copiado para a saída,
    então as bordas (onde redes convolucionais erram) são descartadas.
    """
    h, w, c = img.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return fn(img)
    out = np.zeros((h * scale, w * scale, c), dtype=np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - overlap, 0), max(x0 - overlap, 0)
            py1, px1 = min(y1 + overlap, h), min(x1 + overlap, w)
            res = fn(img[py0:py1, px0:px1])
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                res[oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
    return out
