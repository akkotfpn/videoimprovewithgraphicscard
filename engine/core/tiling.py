"""Tiling com sobreposição: processa imagens maiores que a VRAM em blocos e costura sem emenda visível."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def balanced_tile(width: int, height: int, max_tile: int) -> tuple[int, int]:
    """
    Divide o frame em blocos do mesmo tamanho, o maior possível até `max_tile`.
    Ex.: 720x480 com máximo 384 -> 2x2 blocos de 360x240, em vez de 4 blocos de 384 com sobra.
    Menos blocos e menos preenchimento = menos computação desperdiçada.
    """
    if max_tile <= 0:
        return 0, 0
    tw = math.ceil(width / math.ceil(width / max_tile))
    th = math.ceil(height / math.ceil(height / max_tile))
    return th, tw


def process_tiled(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray],
                  scale: int, tile: int | tuple[int, int], overlap: int = 16) -> np.ndarray:
    """
    img: HxWxC. fn recebe um bloco hxwxC e devolve (h*scale)x(w*scale)xC.
    tile: lado do bloco (int) ou (altura, largura). 0 = frame inteiro.
    Cada bloco é processado com uma margem de `overlap` px; só o miolo é copiado para a saída,
    então as bordas (onde redes convolucionais erram) são descartadas.
    """
    th, tw = (tile, tile) if isinstance(tile, int) else tile
    h, w, c = img.shape
    if th <= 0 or tw <= 0 or (h <= th and w <= tw):
        return fn(img)
    out = np.zeros((h * scale, w * scale, c), dtype=np.float32)
    for y0 in range(0, h, th):
        for x0 in range(0, w, tw):
            y1, x1 = min(y0 + th, h), min(x0 + tw, w)
            py0, px0 = max(y0 - overlap, 0), max(x0 - overlap, 0)
            py1, px1 = min(y1 + overlap, h), min(x1 + overlap, w)
            res = fn(img[py0:py1, px0:px1])
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                res[oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
    return out
