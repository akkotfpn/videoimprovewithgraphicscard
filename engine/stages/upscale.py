"""
Estágios de super-resolução.

OnnxUpscale: roda qualquer modelo SR exportado para ONNX (Real-ESRGAN, realesr-general-x4v3,
SPAN, etc.) no melhor Execution Provider disponível, com tiling automático e fallback
de tamanho de tile quando a GPU fica sem memória.

ResizeUpscale: fallback sem IA (Lanczos via Pillow, ou nearest em NumPy). Serve para o
preset CPU e para testar o pipeline sem baixar modelos.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..core.hardware import Backend
from ..core.tiling import process_tiled

log = logging.getLogger("restaura.upscale")

_PROVIDERS = {
    Backend.TENSORRT: ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    Backend.CUDA: ["CUDAExecutionProvider"],
    Backend.ROCM: ["MIGraphXExecutionProvider", "ROCMExecutionProvider"],
    Backend.DIRECTML: ["DmlExecutionProvider"],
    Backend.OPENVINO: ["OpenVINOExecutionProvider"],
}


class ResizeUpscale:
    name = "resize"
    fps_factor = 1

    def __init__(self, scale: int = 2):
        self.scale = scale

    def setup(self, width: int, height: int) -> None:
        pass

    def process(self, batch: np.ndarray) -> np.ndarray:
        try:
            from PIL import Image
            n, h, w, _ = batch.shape
            size = (w * self.scale, h * self.scale)
            return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS)) for f in batch])
        except ImportError:
            return batch.repeat(self.scale, axis=1).repeat(self.scale, axis=2)

    def flush(self):
        return None


class OnnxUpscale:
    name = "onnx_sr"
    fps_factor = 1

    def __init__(self, model_path: str | Path, scale: int, backend: Backend,
                 tile: int = 512, overlap: int = 16, device_id: int = 0, cache_dir: str | Path = "cache"):
        import onnxruntime as ort

        self.scale, self.tile, self.overlap = scale, tile, overlap
        wanted = _PROVIDERS.get(backend, [])
        available = set(ort.get_available_providers())
        providers: list = []
        for p in wanted:
            if p not in available:
                continue
            if p == "TensorrtExecutionProvider":
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                providers.append((p, {
                    "device_id": device_id,
                    "trt_fp16_enable": True,
                    # o engine TRT é compilado na 1ª execução (minutos) e reaproveitado depois
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_timing_cache_enable": True,
                }))
            elif p in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
                providers.append((p, {"device_id": device_id, "cudnn_conv_algo_search": "HEURISTIC"}))
            else:
                providers.append(p)
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        log.info("ONNX %s em %s (%s)", Path(model_path).name,
                 self.session.get_providers()[0], self.dtype.__name__)

    def setup(self, width: int, height: int) -> None:
        pass

    def _infer(self, tile: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None]).astype(self.dtype)
        y = self.session.run(None, {self.input_name: x})[0][0]
        return y.transpose(1, 2, 0).astype(np.float32)

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        img = frame.astype(np.float32) / 255.0
        while True:
            try:
                out = process_tiled(img, self._infer, self.scale, self.tile, self.overlap)
                break
            except Exception as e:  # ORT não expõe uma exceção específica de OOM
                if "memory" not in str(e).lower() or self.tile <= 64:
                    raise
                # Fallback dinâmico: estourou a VRAM -> tile pela metade e tenta de novo
                self.tile = max(64, (self.tile or max(frame.shape[:2])) // 2)
                log.warning("sem VRAM; reduzindo tile para %d", self.tile)
        return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    def process(self, batch: np.ndarray) -> np.ndarray:
        # Modelos SR exportados com eixo de batch dinâmico poderiam receber o lote inteiro;
        # com tiling, cada frame vira N tiles e o ganho de batch vem dos próprios tiles.
        return np.stack([self._frame(f) for f in batch])

    def flush(self):
        return None
