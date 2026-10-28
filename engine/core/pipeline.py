"""
Esqueleto do pipeline: FFmpeg (decode) -> lotes de frames -> estágios de IA -> FFmpeg (encode + áudio original).

Por que pipe em memória e não PNGs no disco:
  - um VHS de 2 h tem ~180 mil frames; extrair para PNG ocupa centenas de GB e é lento.
  - com pipe rawvideo o decode, a inferência e o encode rodam em paralelo (3 threads),
    e a GPU nunca espera o disco.

Sincronia de áudio: o vídeo processado entra no encoder com taxa exata (fração racional
do ffprobe, ex. 30000/1001) e o áudio é copiado da entrada original com -map 1:a.
Como nenhum frame é descartado e a taxa é a mesma (ou multiplicada exatamente pela
interpolação), a duração bate com o áudio.
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger("restaura.pipeline")
_SENTINEL = object()


# --------------------------------------------------------------------------- #
# metadados
# --------------------------------------------------------------------------- #
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frames: int
    interlaced: bool
    has_audio: bool
    duration: float
    dar: Fraction | None = None   # proporção de exibição (720x480 de VHS = 4:3, pixel não quadrado)


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    fps = Fraction(v.get("avg_frame_rate") or v["r_frame_rate"])
    if fps == 0:
        fps = Fraction(v["r_frame_rate"])
    duration = float(v.get("duration") or data["format"].get("duration") or 0)
    frames = int(v.get("nb_frames") or 0) or round(duration * fps)
    field_order = v.get("field_order", "progressive")
    return VideoInfo(
        width=int(v["width"]), height=int(v["height"]), fps=fps, frames=frames,
        interlaced=field_order not in ("progressive", "unknown"),
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
        duration=duration,
        dar=_parse_ratio(v.get("display_aspect_ratio")))


def _parse_ratio(text: str | None) -> Fraction | None:
    if not text or ":" not in text:
        return None
    a, b = text.split(":")
    return Fraction(int(a), int(b)) if int(a) and int(b) else None


# --------------------------------------------------------------------------- #
# estágios
# --------------------------------------------------------------------------- #
class Stage(Protocol):
    """Todo estágio recebe e devolve lotes NHWC uint8 RGB."""
    name: str
    scale: int        # fator espacial (1 = não muda tamanho)
    fps_factor: int   # fator temporal (1 = não muda número de frames)

    def setup(self, width: int, height: int) -> None: ...
    def process(self, batch: np.ndarray) -> np.ndarray: ...
    def flush(self) -> np.ndarray | None: ...  # frames retidos (estágios temporais)


@dataclass
class PipelineConfig:
    input: Path
    output: Path
    stages: list[Stage]
    batch_size: int = 4
    deinterlace: bool | None = None       # None = automático pelo ffprobe
    pre_filters: list[str] = field(default_factory=list)  # filtros clássicos FFmpeg antes da IA
    crf: int = 18
    encoder: str = "libx264"              # h264_nvenc / hevc_nvenc / hevc_amf / hevc_qsv
    queue_depth: int = 4                  # lotes em voo entre threads (limita RAM)
    progress: Callable[[int, int, float], None] | None = None


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.info = probe(cfg.input)
        self._errors: list[BaseException] = []
        self._stop = threading.Event()

    # ---- montagem dos comandos ffmpeg ----
    def _decoder_cmd(self) -> list[str]:
        filters = []
        deint = self.cfg.deinterlace if self.cfg.deinterlace is not None else self.info.interlaced
        if deint:
            # bwdif mode=0: 1 frame por frame (mantém fps e sincronia com o áudio)
            filters.append("bwdif=mode=0")
        filters += self.cfg.pre_filters
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(self.cfg.input)]
        if filters:
            cmd += ["-vf", ",".join(filters)]
        return cmd + ["-map", "0:v:0", "-fps_mode", "passthrough",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    def _encoder_cmd(self, w: int, h: int, fps: Fraction) -> list[str]:
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0",
               "-i", str(self.cfg.input),
               "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
               "-c:v", self.cfg.encoder]
        if h >= 720:
            # material SD é BT.601; saída HD deve ser BT.709, senão as cores ficam deslocadas
            cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        else:
            cmd += ["-pix_fmt", "yuv420p"]
        if self.info.dar:
            # preserva a proporção de tela original (4:3) mesmo com pixels não quadrados na entrada
            cmd += ["-aspect", f"{self.info.dar.numerator}:{self.info.dar.denominator}"]
        if self.cfg.encoder.endswith("nvenc"):
            cmd += ["-preset", "p6", "-rc", "vbr", "-cq", str(self.cfg.crf)]
        else:
            cmd += ["-crf", str(self.cfg.crf), "-preset", "medium"]
        # áudio: copia sem perda se o contêiner aceitar; senão AAC alta taxa
        if self.cfg.output.suffix.lower() in (".mkv", ".mov"):
            cmd += ["-c:a", "copy", "-c:s", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "256k", "-sn"]
        return cmd + ["-shortest", str(self.cfg.output)]

    # ---- threads ----
    def _reader(self, proc: subprocess.Popen, q: queue.Queue, w: int, h: int) -> None:
        frame_bytes = w * h * 3
        try:
            buf: list[np.ndarray] = []
            while not self._stop.is_set():
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                buf.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
                if len(buf) == self.cfg.batch_size:
                    q.put(np.stack(buf)); buf = []
            if buf:
                q.put(np.stack(buf))
        except BaseException as e:  # noqa: BLE001
            self._errors.append(e)
        finally:
            q.put(_SENTINEL)

    def _writer(self, proc: subprocess.Popen, q: queue.Queue) -> None:
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                proc.stdin.write(np.ascontiguousarray(item).tobytes())
        except BaseException as e:  # noqa: BLE001
            self._errors.append(e)
            self._stop.set()
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    # ---- execução ----
    def run(self) -> Path:
        info = self.info
        w, h = info.width, info.height
        out_w, out_h, fps_mult = w, h, 1
        for st in self.cfg.stages:
            st.setup(out_w, out_h)
            out_w, out_h = out_w * st.scale, out_h * st.scale
            fps_mult *= st.fps_factor
        out_w -= out_w % 2; out_h -= out_h % 2  # yuv420p exige par
        out_fps = info.fps * fps_mult
        log.info("entrada %dx%d @ %s fps (%d frames, entrelaçado=%s) -> saída %dx%d @ %s",
                 w, h, float(info.fps), info.frames, info.interlaced, out_w, out_h, float(out_fps))

        dec = subprocess.Popen(self._decoder_cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        enc = subprocess.Popen(self._encoder_cmd(out_w, out_h, out_fps),
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        q_in: queue.Queue = queue.Queue(self.cfg.queue_depth)
        q_out: queue.Queue = queue.Queue(self.cfg.queue_depth)
        t_r = threading.Thread(target=self._reader, args=(dec, q_in, w, h), daemon=True)
        t_w = threading.Thread(target=self._writer, args=(enc, q_out), daemon=True)
        t_r.start(); t_w.start()

        done, t0 = 0, time.perf_counter()
        try:
            while True:
                batch = q_in.get()
                if batch is _SENTINEL or self._stop.is_set():
                    break
                n_in = len(batch)
                for st in self.cfg.stages:
                    batch = st.process(batch)
                q_out.put(batch[:, :out_h, :out_w])
                done += n_in
                if self.cfg.progress:
                    self.cfg.progress(done, info.frames, done / (time.perf_counter() - t0))
            # estágios temporais (ex.: RIFE) podem reter o último frame
            # o que sai do flush do estágio i ainda passa pelos estágios i+1..n
            tail = None
            for st in self.cfg.stages:
                if tail is not None:
                    tail = st.process(tail)
                extra = st.flush()
                if extra is not None and len(extra):
                    tail = extra if tail is None else np.concatenate([tail, extra])
            if tail is not None and len(tail):
                q_out.put(tail[:, :out_h, :out_w])
        finally:
            q_out.put(_SENTINEL)
            t_w.join(); t_r.join(timeout=5)
            dec.wait(); enc.wait()

        if self._errors:
            raise RuntimeError(f"falha no pipeline: {self._errors[0]!r}")
        if enc.returncode != 0:
            raise RuntimeError("encoder: " + enc.stderr.read().decode(errors="replace")[-2000:])
        if dec.returncode != 0:
            raise RuntimeError("decoder: " + dec.stderr.read().decode(errors="replace")[-2000:])
        log.info("concluído: %d frames em %.1fs", done, time.perf_counter() - t0)
        return self.cfg.output


def require_ffmpeg() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if not shutil.which(exe):
            raise FileNotFoundError(f"{exe} não encontrado no PATH (ou na pasta bin/ do app)")
