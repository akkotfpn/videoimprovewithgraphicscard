"""
CLI do engine. É o mesmo ponto de entrada que a interface desktop chama como processo filho.

  python -m engine detect                 # mostra hardware e o plano escolhido
  python -m engine run entrada.mp4 saida.mkv --scale 2 --interpolate
  python -m engine run fita.avi saida.mkv --preset economico --model models/realesr-general-x4v3.onnx

Com --json, o progresso sai como uma linha JSON por evento no stdout (protocolo da UI).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from .core import paths
from .core.hardware import Backend, Preset, choose_plan, detect_hardware
from .core.pipeline import Pipeline, PipelineConfig, probe, require_ffmpeg
from .stages.interpolate import BlendInterpolate
from .models import catalog
from .stages.upscale import OnnxUpscale, ResizeUpscale

# Filtros clássicos do FFmpeg para ruído de fita (rodam antes da IA, baratos)
DENOISE_FILTERS = {
    "ffmpeg_hqdn3d": "hqdn3d=3:2:6:4",                 # rápido, bom para chroma noise de VHS
    "ffmpeg_nlmeans": "nlmeans=s=2.5:p=7:r=15",        # melhor qualidade, mais lento na CPU
    "scunet": "hqdn3d=2:1.5:4:3",                      # enquanto o SCUNet não entra, leve pré-filtro
}
# Escolha do usuário (interface): auto = o do preset
DENOISE_LEVELS = {"off": None, "light": "hqdn3d=2:1.5:4:3", "medium": "hqdn3d=3:2:6:4",
                  "strong": "nlmeans=s=3:p=7:r=15"}


def denoise_filter(choice: str, plan) -> list[str]:
    if choice == "auto":
        f = DENOISE_FILTERS.get(plan.denoise)
    else:
        f = DENOISE_LEVELS[choice]
    return [f] if f else []


def deinterlace_choice(arg: str) -> bool | None:
    return {"auto": None, "on": True, "off": False}[arg]


def make_plan(args, hw=None):
    hw = hw or detect_hardware()
    plan = choose_plan(hw, Preset(args.preset) if getattr(args, "preset", None) else None,
                       getattr(args, "gpu", None))
    if getattr(args, "tile", None) is not None:
        plan.tile_size = args.tile
    if getattr(args, "device", None) is not None:
        plan.device_id = args.device
    return hw, plan


def _encoder_available(name: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                         creationflags=paths.no_window_flags()).stdout
    return f" {name} " in out


def _encoder_works(name: str) -> bool:
    """Codifica 1 quadro de teste: o encoder existir no FFmpeg não garante que a placa/driver suporte."""
    if not _encoder_available(name):
        return False
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                        "-frames:v", "1", "-c:v", name, "-f", "null", "-"], capture_output=True,
                       creationflags=paths.no_window_flags(), timeout=60)
    return r.returncode == 0


def pick_encoder(hw, plan) -> str:
    """Encoder de vídeo da própria placa (NVENC/AMF/QSV) quando funciona; senão libx264 na CPU."""
    gpu = next((g for g in hw.gpus if g.name == plan.gpu_name), None)
    order: list[str] = []
    if gpu is not None:
        order = {"nvidia": ["hevc_nvenc", "h264_nvenc"], "amd": ["hevc_amf", "h264_amf"],
                 "intel": ["hevc_qsv", "h264_qsv"]}.get(gpu.vendor.value, [])
    for name in order:
        if _encoder_works(name):
            return name
    return "libx264"


def _prefer_fp16(plan) -> bool:
    # FP16 só em GPU (na CPU é mais lento e pode nem ter kernel)
    return plan.backend != Backend.CPU and plan.precision != "fp32"


def model_file(key_or_path: str, plan) -> Path:
    """Arquivo ONNX de um modelo do catálogo (FP16 na GPU, FP32 na CPU) ou um caminho .onnx."""
    prefer = _prefer_fp16(plan)
    p = catalog.find(key_or_path, prefer)
    if p and not prefer and ".fp16" in p.name:
        p = OnnxUpscale.fp32_sibling(p) or p
    if p is None:
        raise FileNotFoundError(f"modelo não encontrado: {key_or_path} "
                                f"(instale com: python -m engine models install {key_or_path})")
    return p


def installed_models(plan) -> set[str]:
    return {k for k in catalog.CATALOG if catalog.find(k, _prefer_fp16(plan))}


def plan_job(args, plan, info, path: str | None):
    """Resolução de trabalho + modelo para este vídeo (veja engine/core/adapt.py)."""
    from .core.adapt import choose_job
    requested = getattr(args, "model", None)
    custom = requested if requested and requested.endswith(".onnx") else None
    key = None if custom else requested
    job = choose_job(plan, info.width, info.height, args.scale, path=path, duration=info.duration,
                     work=getattr(args, "work", "auto"), model=key, available=installed_models(plan))
    return job, custom


def make_sr_stage(job, plan, custom: str | None = None, tile: int | None = None):
    log = logging.getLogger("restaura")
    if job.sr_scale <= 1:
        return None
    if job.model is None and not custom:
        if plan.sr_model == "classico":
            log.warning("sem GPU acelerada: usando filtros clássicos. IA na CPU é muito lenta, mas dá "
                        "para forçar em trechos curtos com --model realesr_general_x4v3")
        else:
            log.warning("nenhum modelo de IA instalado: usando redimensionamento clássico. "
                        "Instale com: python -m engine models install realesr_general_x4v3")
        return ResizeUpscale(job.sr_scale)
    path = Path(custom) if custom else model_file(job.model, plan)
    return OnnxUpscale(path, job.sr_scale, plan.backend, tile=tile, device_id=plan.device_id,
                       mem_budget_mb=plan.mem_budget_mb, mem_limit_mb=plan.mem_limit_mb,
                       max_batch=plan.batch_size)


def build_sr_stage(requested: str | None, scale: int, plan, width: int = 720, height: int = 480):
    """Atalho (bench sintético, selftest): mesmo critério do run, sem medir detalhe."""
    from .core.adapt import choose_job
    custom = requested if requested and requested.endswith(".onnx") else None
    job = choose_job(plan, width, height, scale, work="original", model=None if custom else requested,
                     available=installed_models(plan))
    return make_sr_stage(job, plan, custom) or ResizeUpscale(scale)


def cmd_models(args) -> int:
    if args.action == "list":
        for key, spec in catalog.CATALOG.items():
            inst = [p.name for p in (catalog.onnx_path(key), catalog.onnx_path(key, True)) if p.exists()]
            mark = "instalado" if inst else "não instalado"
            print(f"{key:24s} [{mark}] {spec.license}\n    {spec.description}")
        print(f"\npasta: {catalog.default_dir()}")
        return 0
    keys = list(catalog.CATALOG) if args.key == "all" else [args.key]
    for key in keys:
        if key not in catalog.CATALOG:
            print(f"modelo desconhecido: {key}. Opções: {', '.join(catalog.CATALOG)}", file=sys.stderr)
            return 2
        for p in catalog.install(key, denoise=args.denoise):
            print(p)
    return 0


def cmd_detect(args) -> int:
    hw, plan = make_plan(args)
    if args.json:
        print(json.dumps({"hardware": json.loads(hw.to_json()), "plan": json.loads(plan.to_json())},
                         ensure_ascii=False))
    else:
        print(hw.to_json()); print(plan.to_json())
        if len(hw.gpus) > 1:
            print("\nGPUs (use --gpu N para escolher outra):", file=sys.stderr)
            for i, g in enumerate(hw.gpus):
                kind = "integrada" if g.integrated else "dedicada"
                print(f"  {i}: {g.name} ({kind})", file=sys.stderr)
    return 0


def cmd_run(args) -> int:
    require_ffmpeg()
    hw, plan = make_plan(args)
    log = logging.getLogger("restaura")
    log.info("GPU=%s preset=%s backend=%s device=%s precisão=%s tile=%s (%s)", plan.gpu_name,
             plan.preset.value, plan.backend.value, plan.device_id, plan.precision, plan.tile_size,
             "; ".join(plan.reasons))

    info = probe(args.input)
    deint = deinterlace_choice(args.deinterlace)
    deinterlacing = deint if deint is not None else info.interlaced
    job, custom = plan_job(args, plan, info, args.input)
    for n in job.notes:
        log.info("%s", n)
    stages = []
    sr = make_sr_stage(job, plan, custom, args.tile)
    if sr:
        stages.append(sr)
    field_rate = False
    if args.smooth and deinterlacing:
        field_rate = True        # VHS entrelaçado: 1 quadro por campo = 60 fps com movimento real
    elif args.smooth or args.interpolate:
        stages.append(BlendInterpolate())

    encoder = args.encoder
    if encoder == "auto":
        encoder = pick_encoder(hw, plan)

    if args.json:
        print(json.dumps({"event": "start", "gpu": plan.gpu_name, "preset": plan.preset.value,
                          "backend": plan.backend.value, "stages": [st.name for st in stages],
                          "field_rate": field_rate, "encoder": encoder, "model": job.model,
                          "work": f"{job.work_w}x{job.work_h}", "output": f"{job.out_w}x{job.out_h}",
                          "provider": getattr(sr, "provider", None), "notes": job.notes},
                         ensure_ascii=False), flush=True)

    def progress(done, total, fps):
        eta = (total - done) / fps if fps > 0 and total else None
        if args.json:
            print(json.dumps({"event": "progress", "done": done, "total": total, "fps": round(fps, 2),
                              "eta": round(eta) if eta is not None else None}), flush=True)
        else:
            pct = 100 * done / total if total else 0
            print(f"\r{done}/{total} frames ({pct:5.1f}%) {fps:6.2f} fps", end="", file=sys.stderr)

    denoise = "off" if args.no_denoise else args.denoise
    cfg = PipelineConfig(
        input=Path(args.input), output=Path(args.output), stages=stages,
        batch_size=plan.batch_size, deinterlace=deint, field_rate=field_rate,
        pre_filters=denoise_filter(denoise, plan),
        work_size=(job.work_w, job.work_h) if job.work_div > 1 else None,
        hwaccel="cuda" if plan.backend in (Backend.CUDA, Backend.TENSORRT) else None,
        encoder=encoder, crf=args.crf, progress=progress)
    out = Pipeline(cfg).run()
    if args.json:
        print(json.dumps({"event": "done", "output": str(out)}, ensure_ascii=False), flush=True)
    else:
        print(f"\nOK -> {out}", file=sys.stderr)
    return 0


def _grab_frame(path: str, t: float, filters: list[str], w: int, h: int):
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, creationflags=paths.no_window_flags()).stdout
    if len(raw) < w * h * 3:
        raise RuntimeError(f"não consegui ler um quadro em {t:.1f}s")
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3)


def cmd_preview(args) -> int:
    """Processa um único quadro e grava antes/depois em PNG (mesmo tamanho, proporção de tela correta)."""
    import time

    import numpy as np
    from PIL import Image

    require_ffmpeg()
    hw, plan = make_plan(args)
    info = probe(args.input)
    t = args.time if args.time is not None else max(0.0, info.duration * 0.4)
    deint = deinterlace_choice(args.deinterlace)
    filters = (["bwdif=mode=0"] if (deint if deint is not None else info.interlaced) else [])
    filters += denoise_filter(args.denoise, plan)
    job, custom = plan_job(args, plan, info, args.input)
    original = _grab_frame(args.input, t, [], info.width, info.height)
    if job.work_div > 1:
        filters.append(f"scale={job.work_w}:{job.work_h}:flags=lanczos")
    frame = _grab_frame(args.input, t, filters, job.work_w, job.work_h)

    st = make_sr_stage(job, plan, custom, args.tile)
    seconds = 0.0
    if st:
        st.setup(job.work_w, job.work_h)
        batch = np.repeat(frame[None], max(1, plan.batch_size), axis=0)
        st.process(batch)                              # aquecimento (escolha de kernels do cuDNN)
        t0 = time.perf_counter()
        after = st.process(batch)[0]
        seconds = (time.perf_counter() - t0) / len(batch)
    else:
        from .stages.upscale import _resize
        after = _resize(frame, job.out_w, job.out_h)
    oh = after.shape[0]
    ow = round(oh * float(info.dar)) if info.dar else after.shape[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    before_p, after_p = out_dir / "antes.png", out_dir / "depois.png"
    Image.fromarray(original).resize((ow, oh), Image.LANCZOS).save(before_p)
    Image.fromarray(after).resize((ow, oh), Image.LANCZOS).save(after_p)

    frames = info.frames * (2 if args.smooth and (deint if deint is not None else info.interlaced) else 1)
    result = {"event": "preview", "before": str(before_p), "after": str(after_p), "time": round(t, 2),
              "seconds_per_frame": round(seconds, 3),
              "estimated_seconds": round(seconds * frames) if seconds else None,
              "model": getattr(st, "model_path", None) and st.model_path.name,
              "provider": st.session.get_providers()[0] if isinstance(st, OnnxUpscale) else None,
              "gpu": plan.gpu_name, "preset": plan.preset.value,
              "work": f"{job.work_w}x{job.work_h}", "output": f"{job.out_w}x{job.out_h}",
              "notes": job.notes}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def cmd_selftest(args) -> int:
    """
    Verifica a instalação: FFmpeg embutido, modelos, ONNX Runtime e o pipeline completo num clipe
    sintético curto (entrelaçado, com áudio). Usado no CI e pelo botão "Verificar instalação".
    """
    import tempfile

    checks: dict[str, object] = {}
    ok = True
    checks["app_dir"] = str(paths.app_dir())
    checks["ffmpeg"] = shutil.which("ffmpeg")
    checks["ffprobe"] = shutil.which("ffprobe")
    try:
        import onnxruntime as ort
        checks["onnxruntime"] = ort.__version__
        checks["providers"] = ort.get_available_providers()
    except Exception as e:  # noqa: BLE001
        checks["onnxruntime"] = f"ERRO: {e}"
        ok = False
    checks["models"] = sorted(p.name for p in catalog.default_dir().glob("*.onnx"))
    hw, plan = make_plan(args)
    checks["gpus"] = [g.name for g in hw.gpus]
    checks["plan"] = {"preset": plan.preset.value, "backend": plan.backend.value, "device": plan.device_id}
    if not (checks["ffmpeg"] and checks["ffprobe"]):
        ok = False
    if ok:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "teste.mkv", Path(tmp) / "saida.mkv"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30000/1001:duration=0.4",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4:sample_rate=48000",
                "-vf", "tinterlace=mode=interleave_top,fps=30000/1001,setfield=tff",
                "-c:v", "libx264", "-flags", "+ildct+ilme", "-c:a", "pcm_s16le", "-shortest", str(src)],
                check=True, creationflags=paths.no_window_flags())
            model = "realesr_general_x4v3" if "realesr_general_x4v3.onnx" in checks["models"] else None
            stages = [build_sr_stage(model, 2, plan)]
            checks["sr_stage"] = getattr(stages[0], "model_path", None) and stages[0].model_path.name
            if isinstance(stages[0], OnnxUpscale):
                checks["sr_provider"] = stages[0].session.get_providers()[0]
            Pipeline(PipelineConfig(src, dst, stages, batch_size=2, field_rate=True,
                                    pre_filters=["hqdn3d=2:1.5:4:3"])).run()
            out = probe(dst)
            checks["output"] = {"size": f"{out.width}x{out.height}", "fps": round(float(out.fps), 3),
                                "audio": out.has_audio}
            ok = out.width == 640 and out.height == 480 and out.has_audio and abs(float(out.fps) - 59.94) < 0.1
    checks["ok"] = ok
    print(json.dumps(checks, ensure_ascii=False, indent=None if args.json else 2), flush=True)
    return 0 if ok else 1


def _fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    return f"{int(h)} h {int((h % 1) * 60):02d} min"


def cmd_bench(args) -> int:
    """
    Mede a velocidade real do estágio de IA nesta máquina com frames do tamanho de um VHS
    (720x480) e estima quanto tempo leva 1 hora de fita. Exclui decode/encode, que rodam
    em paralelo e quase nunca são o gargalo.
    """
    import time

    import numpy as np

    hw, plan = make_plan(args)
    if args.input:
        require_ffmpeg()
        info = probe(args.input)
        job, custom = plan_job(args, plan, info, args.input)
        filt = [f"scale={job.work_w}:{job.work_h}:flags=lanczos"] if job.work_div > 1 else []
        src = _grab_frame(args.input, max(0.0, info.duration * 0.4), filt, job.work_w, job.work_h)
        frames_total = info.frames
    else:
        from .core.adapt import choose_job
        w, h = (int(v) for v in args.size.lower().split("x"))
        custom = args.model if args.model and args.model.endswith(".onnx") else None
        job = choose_job(plan, w, h, args.scale, work=args.work if args.work != "auto" else "original",
                         model=None if custom else args.model, available=installed_models(plan))
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:job.work_h, 0:job.work_w]
        src = np.stack([(xx * 255 // job.work_w), (yy * 255 // job.work_h), ((xx + yy) % 256)], -1)
        src = np.clip(src + rng.integers(-20, 20, src.shape), 0, 255).astype(np.uint8)
        frames_total = round(3600 * 29.97)
    st = make_sr_stage(job, plan, custom, args.tile) or ResizeUpscale(max(1, job.sr_scale))
    if isinstance(st, OnnxUpscale):
        label = f"{st.model_path.name} via {st.provider} ({st.dtype.__name__})"
    else:
        label = "redimensionamento clássico (sem modelo)"

    batch = np.repeat(src[None], max(1, plan.batch_size), axis=0)
    st.setup(job.work_w, job.work_h)
    st.process(batch)  # aquecimento (cuDNN/TensorRT/DirectML escolhem e compilam kernels aqui)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        st.process(batch); n += len(batch)
    fps = n / (time.perf_counter() - t0)
    hours = frames_total / fps / 3600

    layout = "inteiro" if getattr(st, "tiles", None) is None else f"{len(st.tiles)} blocos"
    result = {
        "gpu": plan.gpu_name or "CPU", "preset": plan.preset.value, "backend": plan.backend.value,
        "estagio": label, "escala": args.scale, "entrada_ia": f"{job.work_w}x{job.work_h}",
        "saida": f"{job.out_w}x{job.out_h}", "lote": getattr(st, "per_run", 1), "quadro": layout,
        "fps": round(fps, 2), "horas": round(hours, 2), "notas": job.notes,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        alvo = Path(args.input).name if args.input else "1 h de vídeo a 29,97 fps"
        print(f"GPU:        {result['gpu']}  ({plan.preset.value}, {plan.backend.value})")
        print(f"Modelo:     {label}")
        print(f"IA:         {job.work_w}x{job.work_h} -> {job.out_w}x{job.out_h}  "
              f"(quadro {layout}, {result['lote']} por chamada)")
        for nota in job.notes:
            print(f"            {nota}")
        print(f"Medido:     {fps:.2f} quadros/s ({n} quadros)")
        print(f"Estimativa: {alvo} leva ~{_fmt_hours(hours)} só na IA")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="engine", description="Restauração de vídeo adaptativa ao hardware")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detecta hardware e mostra o plano")
    d.add_argument("--preset", choices=[x.value for x in Preset])
    d.add_argument("--json", action="store_true")
    d.add_argument("--gpu", type=int, help="índice da GPU (veja 'detect') para montar o plano")
    d.set_defaults(fn=cmd_detect)

    r = sub.add_parser("run", help="processa um vídeo")
    r.add_argument("input"); r.add_argument("output")
    r.add_argument("--preset", choices=[x.value for x in Preset])
    r.add_argument("--model", help="nome do catálogo ou caminho .onnx (padrão: o do preset)")
    r.add_argument("--scale", type=int, default=2)
    r.add_argument("--tile", type=int)
    r.add_argument("--interpolate", action="store_true", help="dobra o FPS por mistura (vídeo progressivo)")
    r.add_argument("--smooth", action="store_true",
                   help="60 fps: em vídeo entrelaçado usa 1 quadro por campo (movimento real); senão interpola")
    r.add_argument("--deinterlace", choices=["auto", "on", "off"], default="auto")
    r.add_argument("--denoise", choices=["auto", *DENOISE_LEVELS], default="auto")
    r.add_argument("--work", choices=["auto", "original", "half"], default="auto",
                   help="resolução em que a IA roda: auto mede o detalhe real do vídeo")
    r.add_argument("--no-denoise", action="store_true", help="o mesmo que --denoise off")
    r.add_argument("--encoder", default="auto")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--json", action="store_true", help="eventos JSON no stdout (para a UI)")
    r.add_argument("--gpu", type=int, help="índice da GPU (veja 'detect') para montar o plano")
    r.add_argument("--device", type=int,
                   help="índice do adaptador no runtime (CUDA/DirectML). No DirectML segue a ordem do Windows")
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("models", help="lista ou instala modelos de IA (converte para ONNX, sem PyTorch)")
    m.add_argument("action", choices=["list", "install"])
    m.add_argument("key", nargs="?", default="realesr_general_x4v3", help="nome do modelo ou 'all'")
    m.add_argument("--denoise", type=float, help="compact: 0 = mantém textura, 1 = limpa mais (padrão 0.5)")
    m.set_defaults(fn=cmd_models)

    b = sub.add_parser("bench", help="mede a velocidade e estima o tempo por hora de fita")
    b.add_argument("--model", help="nome do catálogo ou caminho .onnx (padrão: o do preset)")
    b.add_argument("--scale", type=int, default=2)
    b.add_argument("--preset", choices=[x.value for x in Preset])
    b.add_argument("--tile", type=int)
    b.add_argument("--seconds", type=float, default=15)
    b.add_argument("--input", help="usa um vídeo real (resolução, detalhe e duração dele)")
    b.add_argument("--size", default="720x480", help="sem --input: resolução sintética (padrão 720x480)")
    b.add_argument("--work", choices=["auto", "original", "half"], default="auto")
    b.add_argument("--json", action="store_true")
    b.add_argument("--gpu", type=int, help="índice da GPU (veja 'detect') para montar o plano")
    b.add_argument("--device", type=int,
                   help="índice do adaptador no runtime (CUDA/DirectML). No DirectML segue a ordem do Windows")
    b.set_defaults(fn=cmd_bench)

    pv = sub.add_parser("preview", help="processa um quadro e grava antes.png/depois.png")
    pv.add_argument("input")
    pv.add_argument("--out", required=True, help="pasta de saída")
    pv.add_argument("--time", type=float, help="segundo do vídeo (padrão: 40%% da duração)")
    pv.add_argument("--preset", choices=[x.value for x in Preset])
    pv.add_argument("--model")
    pv.add_argument("--scale", type=int, default=2)
    pv.add_argument("--tile", type=int)
    pv.add_argument("--deinterlace", choices=["auto", "on", "off"], default="auto")
    pv.add_argument("--denoise", choices=["auto", *DENOISE_LEVELS], default="auto")
    pv.add_argument("--work", choices=["auto", "original", "half"], default="auto")
    pv.add_argument("--smooth", action="store_true", help="só afeta a estimativa de tempo")
    pv.add_argument("--gpu", type=int)
    pv.add_argument("--device", type=int)
    pv.add_argument("--json", action="store_true")
    pv.set_defaults(fn=cmd_preview)

    st = sub.add_parser("selftest", help="verifica FFmpeg, modelos, ONNX Runtime e o pipeline")
    st.add_argument("--preset", choices=[x.value for x in Preset])
    st.add_argument("--gpu", type=int)
    st.add_argument("--device", type=int)
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=cmd_selftest)

    args = p.parse_args(argv)
    # Windows: stdout redirecionado usa a página de código local (cp1252) e quebra com acentos.
    # A interface lê UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    paths.setup_environment()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        if args.verbose:
            raise
        if getattr(args, "json", False):
            print(json.dumps({"event": "error", "message": str(e)}, ensure_ascii=False), flush=True)
        print(f"erro: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
