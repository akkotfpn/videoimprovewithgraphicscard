"""
Interpolação de quadros (dobra o FPS).

BlendInterpolate é o "contrato" que o RIFE vai seguir: estágio temporal que precisa do
frame seguinte para gerar o intermediário, então retém sempre 1 frame entre lotes
e devolve o último no flush(). Aqui o intermediário é uma média simples; o RifeInterpolate
troca só a função `_mid` por uma chamada ao modelo.

Contagem de frames: N de entrada -> exatamente 2N de saída, para o áudio continuar
sincronizado com a taxa de saída = 2 x taxa de entrada.
"""
from __future__ import annotations

import numpy as np


class BlendInterpolate:
    name = "blend_x2"
    scale = 1
    fps_factor = 2

    def __init__(self):
        self._prev: np.ndarray | None = None

    def setup(self, width: int, height: int) -> None:
        self._prev = None

    def _mid(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((a.astype(np.uint16) + b) // 2).astype(np.uint8)

    def process(self, batch: np.ndarray) -> np.ndarray:
        out = []
        for f in batch:
            if self._prev is not None:
                out += [self._prev, self._mid(self._prev, f)]
            self._prev = f
        if not out:
            return batch[:0]
        return np.stack(out)

    def flush(self) -> np.ndarray | None:
        if self._prev is None:
            return None
        last, self._prev = self._prev, None
        return np.stack([last, last])  # último frame duplicado mantém 2N exatos
