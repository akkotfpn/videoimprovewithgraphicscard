"""
Estágios de super-resolução.

OnnxUpscale roda os modelos ONNX gerados pelo app no melhor Execution Provider disponível.

Formato do app (modelos gerados a partir da v0.3): entrada `frames` uint8 [lote, A, L, 3] e saída `out`
uint8. A conversão para ponto flutuante, a normalização, a redução 4x -> 2x (média 2x2) e o
arredondamento acontecem dentro do grafo, na GPU. A CPU só fatia e copia bytes. Antes disso, cada
quadro 1080p ampliado gerava ~400 MB de float32 em 8K que a CPU convertia e reduzia com Lanczos: ~1 s
por quadro com a GPU parada (o "dente de serra" no Gerenciador de Tarefas).

Lote: quadros inteiros (ou blocos, quando não cabem) são agrupados numa única chamada, no limite de um
orçamento de memória calculado a partir da VRAM e do custo de memória de cada modelo. Se a GPU ficar
sem memória, o orçamento cai pela metade e a chamada é refeita.

Modelos antigos (entrada float NCHW) continuam funcionando pelo caminho legado.

ResizeUpscale: fallback sem IA (Lanczos), para o preset CPU e para testes.
"""
from __future__ import annotations

import logging
import math
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
_OOM_MARKERS = ("memory", "alloc", "8007000e", "887a0005", "out of", "cudnn_status_alloc")
PAD_MULTIPLE = 4  # x2plus exige dimensões pares; 4 também mantém a saída 2x/4x inteira


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))
    except ImportError:  # sem Pillow: nearest (só para testes)
        ys = (np.arange(h) * img.shape[0] / h).astype(int)
        xs = (np.arange(w) * img.shape[1] / w).astype(int)
        return img[ys][:, xs]


def _round_up(v: int, m: int) -> int:
    return -(-v // m) * m


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
                 tile: int | None = None, overlap: int = 32, device_id: int = 0,
                 cache_dir: str | Path | None = None, mem_budget_mb: int | None = None,
                 mem_limit_mb: int | None = None, max_batch: int = 8):
        """
        scale: fator de saída pedido. model_scale: fator nativo (detectado sozinho).
        tile: None = automático pelo orçamento de memória; 0 = quadro inteiro; N = blocos de N px.
        mem_budget_mb: memória que a inferência pode usar (o plano passa ~60% da VRAM).
        mem_limit_mb: teto da arena do CUDA (evita o ONNX Runtime ocupar toda a VRAM).
        """
        self.scale, self.overlap, self.max_batch = scale, overlap, max(1, max_batch)
        self.user_tile = tile
        if cache_dir is None:
            from ..core.paths import cache_dir as _cache
            cache_dir = _cache() / "tensorrt"
        self.backend, self.device_id, self.cache_dir = backend, device_id, cache_dir
        self.mem_limit_mb = mem_limit_mb
        self.model_path = Path(model_path)
        from ..models.catalog import spec_for
        spec = spec_for(self.model_path)
        self.mem_per_px = spec.mem_per_px if spec else 3000
        self.budget_mb = mem_budget_mb or 1024
        self._load(self.model_path)
        self.model_scale = model_scale or self._probe_scale()
        if self.scale > self.model_scale:
            raise ValueError(f"modelo é {self.model_scale}x; não dá para gerar {self.scale}x com ele")
        self._layout_key: tuple | None = None
        # compatibilidade com o caminho legado e com a interface/bench
        self.tile = tile if tile is not None else 0
        self.tile_hw: tuple[int, int] = (self.tile, self.tile)
        self._frame_wh: tuple[int, int] | None = None

    # ------------------------------------------------------------------ sessão
    def _provider_list(self, extra_cuda: bool):
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        providers: list = []
        for p in _PROVIDERS.get(self.backend, []):
            if p not in available:
                continue
            if p == "TensorrtExecutionProvider":
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                providers.append((p, {
                    "device_id": self.device_id, "trt_fp16_enable": True,
                    # o engine TRT é compilado na 1ª execução (minutos) e reaproveitado depois
                    "trt_engine_cache_enable": True, "trt_engine_cache_path": str(self.cache_dir),
                    "trt_timing_cache_enable": True,
                }))
            elif p == "CUDAExecutionProvider":
                o = {
                    "device_id": self.device_id,
                    # equivalente ao torch.backends.cudnn.benchmark=True: testa os algoritmos de
                    # convolução e guarda o mais rápido por formato (por isso os formatos são fixos)
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "arena_extend_strategy": "kSameAsRequested",
                    "do_copy_in_default_stream": "1",
                }
                if self.mem_limit_mb:
                    o["gpu_mem_limit"] = str(int(self.mem_limit_mb) * 2**20)
                if extra_cuda:
                    o["prefer_nhwc"] = "1"   # kernels channels-last: melhor uso dos Tensor Cores
                    o["use_tf32"] = "1"      # acelera os modelos FP32 em RTX 30+
                providers.append((p, o))
            elif p == "ROCMExecutionProvider":
                providers.append((p, {"device_id": self.device_id}))
            elif p == "DmlExecutionProvider":
                providers.append((p, {"device_id": self.device_id}))
            else:
                providers.append(p)
        providers.append("CPUExecutionProvider")
        return providers

    def _load(self, model_path: Path) -> None:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls") and self.backend in (Backend.CUDA, Backend.TENSORRT):
            # onnxruntime-gpu[cuda,cudnn] instala CUDA/cuDNN como pacotes pip; isto os carrega
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.backend == Backend.DIRECTML:
            # exigido pelo DirectML: sem padrão de memória e execução sequencial
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self.session = ort.InferenceSession(str(model_path), opts, providers=self._provider_list(True))
        except Exception as e:  # noqa: BLE001 - versões antigas recusam opções que não conhecem
            log.info("recriando a sessão sem opções extras do CUDA (%s)", str(e)[:120])
            self.session = ort.InferenceSession(str(model_path), opts, providers=self._provider_list(False))
        names = [i.name for i in self.session.get_inputs()]
        self.uint8_io = "frames" in names
        self.out_names = [o.name for o in self.session.get_outputs()]
        if self.uint8_io:
            self.input_name = "frames"
            self.dtype = np.float16 if ".fp16" in model_path.name else np.float32
        else:
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.dtype = np.float16 if "float16" in inp.type else np.float32
        self.provider = self.session.get_providers()[0]
        log.info("ONNX %s em %s (%s%s)", model_path.name, self.provider, self.dtype.__name__,
                 ", entrada uint8" if self.uint8_io else "")

    def _probe_scale(self) -> int:
        """Fator nativo: lê do produtor gravado no modelo ou roda uma entrada 8x8 na CPU."""
        import onnxruntime as ort
        producer = self.session.get_modelmeta().producer_name or ""
        if producer.startswith("videoimprove-u8-x"):
            return int(producer.rsplit("x", 1)[1])
        probe = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        if self.uint8_io:
            y = probe.run(["out"], {"frames": np.zeros((1, 8, 8, 3), np.uint8),
                                    "out_scale": np.array([1, 1, 1, 1], np.float32)})[0]
            return int(y.shape[2] // 8)
        y = probe.run(None, {probe.get_inputs()[0].name: np.zeros((1, 3, 8, 8), dtype=self.dtype)})[0]
        return int(y.shape[-1] // 8)

    @staticmethod
    def fp32_sibling(path: Path) -> Path | None:
        """modelo.fp16.onnx -> modelo.onnx (gerados juntos pelo `models install`)."""
        if ".fp16" in path.name:
            alt = path.with_name(path.name.replace(".fp16", ""))
            return alt if alt.exists() else None
        return None

    def _switch_to_fp32(self) -> None:
        alt = self.fp32_sibling(self.model_path)
        if alt is None:
            raise RuntimeError("saída FP16 inválida (NaN) e não há versão FP32 do modelo")
        log.warning("FP16 gerou NaN nesta GPU; recarregando %s em FP32", alt.name)
        self.model_path = alt
        self.mem_per_px *= 2
        self._load(alt)
        self._layout_key = None

    # ------------------------------------------------------------------ plano de lote
    def setup(self, width: int, height: int) -> None:
        self._frame_wh = (width, height)
        if not self.uint8_io:
            self.tile_hw = balanced_tile(width, height, self.tile)
            return
        self._layout(width, height)

    def _layout(self, width: int, height: int) -> None:
        """Decide quadro inteiro vs. blocos e quantos cabem por chamada, pelo orçamento de memória."""
        key = (width, height, self.budget_mb, self.user_tile)
        if key == self._layout_key:
            return
        self._layout_key = key
        budget_px = max(64 * 64, int(self.budget_mb * 2**20 / self.mem_per_px))
        pw, ph = _round_up(width, PAD_MULTIPLE), _round_up(height, PAD_MULTIPLE)
        whole = self.user_tile == 0 or (self.user_tile is None and pw * ph <= budget_px)
        if whole:
            self.tiles = None
            self.in_shape = (ph, pw)
            self.per_run = max(1, min(self.max_batch, budget_px // (pw * ph)))
            log.info("lote: %d quadro(s) %dx%d inteiros por chamada", self.per_run, width, height)
        else:
            side = self.user_tile or max(64, int(math.sqrt(budget_px)) - 2 * self.overlap)
            th, tw = balanced_tile(width, height, side)
            th, tw = _round_up(th, 2), _round_up(tw, 2)   # origens pares: o x2plus agrupa pixels 2x2
            ih = _round_up(th + 2 * self.overlap, PAD_MULTIPLE)
            iw = _round_up(tw + 2 * self.overlap, PAD_MULTIPLE)
            self.tiles = [(y0, x0, min(y0 + th, height), min(x0 + tw, width))
                          for y0 in range(0, height, th) for x0 in range(0, width, tw)]
            self.in_shape = (ih, iw)
            self.per_run = max(1, min(64, budget_px // (ih * iw)))
            log.info("lote: blocos %dx%d (%d por quadro), %d por chamada", tw, th, len(self.tiles), self.per_run)
        self.tile_hw = (0, 0) if whole else (th, tw)

    # ------------------------------------------------------------------ execução
    def _run(self, x: np.ndarray) -> np.ndarray:
        """Roda um lote uint8 NHWC, com troca automática para FP32 se o FP16 estourar."""
        f = self.scale / self.model_scale
        feeds = {"frames": x, "out_scale": np.array([1, 1, f, f], np.float32)}
        res = self.session.run(self.out_names, feeds)
        if len(res) > 1 and float(res[1]) > 0:
            self._switch_to_fp32()
            return self._run(x)
        return res[0]

    def _run_oom_safe(self, fn):
        while True:
            try:
                return fn()
            except Exception as e:  # ORT não expõe uma exceção específica de falta de memória
                msg = str(e).lower()
                if not any(m in msg for m in _OOM_MARKERS) or self.budget_mb <= 64:
                    raise
                self.budget_mb //= 2
                log.warning("sem memória na GPU; reduzindo o lote (orçamento %d MB)", self.budget_mb)
                self._layout_key = None
                return None

    def _process_whole(self, batch: np.ndarray) -> np.ndarray:
        n, h, w, _ = batch.shape
        ph, pw = self.in_shape
        s = self.scale
        out = np.empty((n, h * s, w * s, 3), np.uint8)
        i = 0
        while i < n:
            k = min(self.per_run, n - i)
            x = batch[i:i + k]
            if (ph, pw) != (h, w):
                x = np.pad(x, ((0, 0), (0, ph - h), (0, pw - w), (0, 0)), mode="edge")
            y = self._run_oom_safe(lambda: self._run(np.ascontiguousarray(x)))
            if y is None:            # orçamento caiu: refaz o plano e repete este trecho
                self._layout(w, h)
                if self.tiles is not None:
                    out[i:] = self._process_tiles(batch[i:])
                    return out
                continue
            out[i:i + k] = y[:, : h * s, : w * s]
            i += k
        return out

    def _process_tiles(self, batch: np.ndarray) -> np.ndarray:
        n, h, w, _ = batch.shape
        ih, iw = self.in_shape
        s, ov = self.scale, self.overlap
        out = np.empty((n, h * s, w * s, 3), np.uint8)
        jobs = [(fi, t) for fi in range(n) for t in self.tiles]
        i = 0
        while i < len(jobs):
            chunk = jobs[i:i + self.per_run]
            x = np.empty((len(chunk), ih, iw, 3), np.uint8)
            offs = []
            for j, (fi, (y0, x0, y1, x1)) in enumerate(chunk):
                py0, px0 = max(y0 - ov, 0), max(x0 - ov, 0)
                py1, px1 = min(y1 + ov, h), min(x1 + ov, w)
                piece = batch[fi, py0:py1, px0:px1]
                x[j] = np.pad(piece, ((0, ih - piece.shape[0]), (0, iw - piece.shape[1]), (0, 0)), mode="edge")
                offs.append((fi, y0, x0, y1, x1, y0 - py0, x0 - px0))
            y = self._run_oom_safe(lambda: self._run(x))
            if y is None:
                # orçamento caiu: refaz o plano e recomeça do primeiro quadro ainda incompleto
                first = chunk[0][0]
                out[first:] = self._process(batch[first:])
                return out
            for j, (fi, y0, x0, y1, x1, oy, ox) in enumerate(offs):
                out[fi, y0 * s:y1 * s, x0 * s:x1 * s] = \
                    y[j, oy * s:(oy + y1 - y0) * s, ox * s:(ox + x1 - x0) * s]
            i += len(chunk)
        return out

    def _process(self, batch: np.ndarray) -> np.ndarray:
        h, w = batch.shape[1:3]
        self._layout(w, h)
        return self._process_whole(batch) if self.tiles is None else self._process_tiles(batch)

    def process(self, batch: np.ndarray) -> np.ndarray:
        if self.uint8_io:
            return self._process(batch)
        return self._process_legacy(batch)

    def flush(self):
        return None

    # ------------------------------------------------------------------ legado (modelos float NCHW)
    def _infer(self, tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[:2]
        th, tw = self.tile_hw
        if th and tw:
            ph, pw = max(0, th + 2 * self.overlap - h), max(0, tw + 2 * self.overlap - w)
            if ph or pw:
                tile = np.pad(tile, ((0, ph), (0, pw), (0, 0)), mode="edge")
        x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None]).astype(self.dtype)
        y = self.session.run(None, {self.input_name: x})[0][0]
        s = self.model_scale
        return y.transpose(1, 2, 0)[: h * s, : w * s].astype(np.float32)

    def _frame_legacy(self, frame: np.ndarray) -> np.ndarray:
        if self._frame_wh != (frame.shape[1], frame.shape[0]):
            self.setup(frame.shape[1], frame.shape[0])
        img = frame.astype(np.float32) / 255.0
        out = process_tiled(img, self._infer, self.model_scale, self.tile_hw, self.overlap)
        if self.dtype == np.float16 and not np.isfinite(out).all():
            self._switch_to_fp32()
            out = process_tiled(img, self._infer, self.model_scale, self.tile_hw, self.overlap)
        return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    def _process_legacy(self, batch: np.ndarray) -> np.ndarray:
        frames = [self._frame_legacy(f) for f in batch]
        if self.model_scale != self.scale:
            h, w = batch.shape[1:3]
            frames = [_resize(f, w * self.scale, h * self.scale) for f in frames]
        return np.stack(frames)
