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
from ..core.tiling import balanced_tile, process_tiled

log = logging.getLogger("restaura.upscale")

_PROVIDERS = {
    Backend.TENSORRT: ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    Backend.CUDA: ["CUDAExecutionProvider"],
    Backend.ROCM: ["MIGraphXExecutionProvider", "ROCMExecutionProvider"],
    Backend.DIRECTML: ["DmlExecutionProvider"],
    Backend.OPENVINO: ["OpenVINOExecutionProvider"],
}


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))
    except ImportError:  # sem Pillow: nearest (só para testes)
        ys = (np.arange(h) * img.shape[0] / h).astype(int)
        xs = (np.arange(w) * img.shape[1] / w).astype(int)
        return img[ys][:, xs]


class ResizeUpscale:
    name = "resize"
    fps_factor = 1

    def __init__(self, scale: int = 2):
        self.scale = scale

    def setup(self, width: int, height: int) -> None:
        pass

    def process(self, batch: np.ndarray) -> np.ndarray:
        h, w = batch.shape[1:3]
        return np.stack([_resize(f, w * self.scale, h * self.scale) for f in batch])

    def flush(self):
        return None


class OnnxUpscale:
    name = "onnx_sr"
    fps_factor = 1

    def __init__(self, model_path: str | Path, scale: int, backend: Backend, model_scale: int | None = None,
                 tile: int = 512, overlap: int = 16, device_id: int = 0, cache_dir: str | Path | None = None):
        """
        scale: fator de saída pedido (2 é o recomendado para VHS).
        model_scale: fator nativo do modelo (o compact x4v3 é 4). Se maior que `scale`,
                     o resultado é reduzido com Lanczos (mesma técnica do Real-ESRGAN --outscale).
        """
        self.scale, self.tile, self.overlap = scale, tile, overlap
        self.tile_hw: tuple[int, int] = (tile, tile)
        self._frame_wh: tuple[int, int] | None = None
        if cache_dir is None:
            from ..core.paths import cache_dir as _cache
            cache_dir = _cache() / "tensorrt"
        self.backend, self.device_id, self.cache_dir = backend, device_id, cache_dir
        self.model_path = Path(model_path)
        self._load(self.model_path)
        self.model_scale = model_scale or self._probe_scale()
        if self.scale > self.model_scale:
            raise ValueError(f"modelo é {self.model_scale}x; não dá para gerar {self.scale}x com ele")

    def _probe_scale(self) -> int:
        """
        Descobre o fator nativo do modelo rodando uma entrada 8x8 numa sessão de CPU separada.
        Na GPU isso faria o TensorRT compilar um engine inteiro só para o formato 8x8.
        """
        import onnxruntime as ort
        probe = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        x = np.zeros((1, 3, 8, 8), dtype=self.dtype)
        y = probe.run(None, {probe.get_inputs()[0].name: x})[0]
        return int(y.shape[-1] // 8)

    @staticmethod
    def fp32_sibling(path: Path) -> Path | None:
        """modelo.fp16.onnx -> modelo.onnx (gerados juntos pelo tools/export_onnx.py)."""
        if ".fp16" in path.name:
            alt = path.with_name(path.name.replace(".fp16", ""))
            return alt if alt.exists() else None
        return None

    def _load(self, model_path: Path) -> None:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls") and self.backend in (Backend.CUDA, Backend.TENSORRT):
            # onnxruntime-gpu[cuda,cudnn] instala CUDA/cuDNN como pacotes pip; isto os carrega
            # sem precisar do CUDA Toolkit instalado no Windows
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass
        backend, device_id, cache_dir = self.backend, self.device_id, self.cache_dir
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
            elif p == "DmlExecutionProvider":
                providers.append((p, {"device_id": device_id}))
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
        self._frame_wh = (width, height)
        self.tile_hw = balanced_tile(width, height, self.tile)

    def _infer(self, tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[:2]
        th, tw = self.tile_hw
        if th and tw:
            # tamanho fixo de entrada: DirectML e TensorRT recompilam/otimizam por formato,
            # então blocos de borda menores são completados (borda replicada) e recortados depois
            ph, pw = max(0, th + 2 * self.overlap - h), max(0, tw + 2 * self.overlap - w)
            if ph or pw:
                tile = np.pad(tile, ((0, ph), (0, pw), (0, 0)), mode="edge")
        x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None]).astype(self.dtype)
        y = self.session.run(None, {self.input_name: x})[0][0]
        s = self.model_scale
        return y.transpose(1, 2, 0)[: h * s, : w * s].astype(np.float32)

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        if self._frame_wh != (frame.shape[1], frame.shape[0]):
            self.setup(frame.shape[1], frame.shape[0])
        img = frame.astype(np.float32) / 255.0
        out = self._run_tiled(img, frame.shape)
        if self.dtype == np.float16 and not np.isfinite(out).all():
            # FP16 estourou (comum em GPUs sem Tensor Cores, ex. GTX 16): troca para FP32 de vez
            alt = self.fp32_sibling(self.model_path)
            if alt is None:
                raise RuntimeError("saída FP16 inválida (NaN) e não há versão FP32 do modelo")
            log.warning("FP16 gerou NaN nesta GPU; recarregando %s em FP32", alt.name)
            self.model_path = alt
            self._load(alt)
            out = self._run_tiled(img, frame.shape)
        return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    def _run_tiled(self, img: np.ndarray, shape) -> np.ndarray:
        while True:
            try:
                out = process_tiled(img, self._infer, self.model_scale, self.tile_hw, self.overlap)
                break
            except Exception as e:  # ORT não expõe uma exceção específica de OOM
                if "memory" not in str(e).lower() or self.tile <= 64:
                    raise
                # Fallback dinâmico: estourou a VRAM -> tile pela metade e tenta de novo
                self.tile = max(64, (self.tile or max(shape[:2])) // 2)
                self.tile_hw = balanced_tile(shape[1], shape[0], self.tile)
                log.warning("sem VRAM; reduzindo blocos para %dx%d", self.tile_hw[1], self.tile_hw[0])
        return out

    def process(self, batch: np.ndarray) -> np.ndarray:
        # Modelos SR exportados com eixo de batch dinâmico poderiam receber o lote inteiro;
        # com tiling, cada frame vira N tiles e o ganho de batch vem dos próprios tiles.
        frames = [self._frame(f) for f in batch]
        if self.model_scale != self.scale:
            h, w = batch.shape[1:3]
            frames = [_resize(f, w * self.scale, h * self.scale) for f in frames]
        return np.stack(frames)

    def flush(self):
        return None
