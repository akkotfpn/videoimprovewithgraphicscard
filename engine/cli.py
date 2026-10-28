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

from .core.hardware import Backend, Preset, choose_plan, detect_hardware
from .core.pipeline import Pipeline, PipelineConfig, require_ffmpeg
from .stages.interpolate import BlendInterpolate
from .models import catalog
from .stages.upscale import OnnxUpscale, ResizeUpscale

# Filtros clássicos do FFmpeg para ruído de fita (rodam antes da IA, baratos)
DENOISE_FILTERS = {
    "ffmpeg_hqdn3d": "hqdn3d=3:2:6:4",                 # rápido, bom para chroma noise de VHS
    "ffmpeg_nlmeans": "nlmeans=s=2.5:p=7:r=15",        # melhor qualidade, mais lento na CPU
    "scunet": "hqdn3d=2:1.5:4:3",                      # enquanto o SCUNet não entra, leve pré-filtro
}


def _encoder_available(name: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {name} " in out


def resolve_model(requested: str | None, plan) -> Path | None:
    """
    Escolhe o arquivo ONNX: o pedido pelo usuário (caminho ou nome do catálogo) ou, sem pedido,
    o modelo do preset; se ele não estiver instalado, cai para o compact.
    FP16 só em GPU (na CPU é mais lento e pode nem ter kernel).
    """
    prefer_fp16 = plan.backend != Backend.CPU and plan.precision != "fp32"
    if requested:
        p = catalog.find(requested, prefer_fp16)
        if p and not prefer_fp16 and ".fp16" in p.name:
            p = OnnxUpscale.fp32_sibling(p) or p
        if p is None:
            raise FileNotFoundError(f"modelo não encontrado: {requested} "
                                    f"(instale com: python -m engine models install {requested})")
        return p
    if plan.sr_model == "classico":
        return None
    for key in dict.fromkeys([plan.sr_model, "realesr_general_x4v3"]):
        if key in catalog.CATALOG and (p := catalog.find(key, prefer_fp16)):
            return p
    return None


def build_sr_stage(requested: str | None, scale: int, plan):
    log = logging.getLogger("restaura")
    path = resolve_model(requested, plan)
    if path is None:
        if plan.sr_model == "classico":
            log.warning("sem GPU acelerada: usando filtros clássicos. IA na CPU é muito lenta, mas dá "
                        "para forçar em trechos curtos com --model realesr_general_x4v3")
        else:
            log.warning("nenhum modelo de IA instalado: usando redimensionamento clássico. "
                        "Instale com: python -m engine models install realesr_general_x4v3")
        return ResizeUpscale(scale)
    return OnnxUpscale(path, scale, plan.backend, tile=plan.tile_size)


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
    hw = detect_hardware()
    plan = choose_plan(hw, Preset(args.preset) if args.preset else None)
    if args.json:
        print(json.dumps({"hardware": json.loads(hw.to_json()), "plan": json.loads(plan.to_json())},
                         ensure_ascii=False))
    else:
        print(hw.to_json()); print(plan.to_json())
    return 0


def cmd_run(args) -> int:
    require_ffmpeg()
    hw = detect_hardware()
    plan = choose_plan(hw, Preset(args.preset) if args.preset else None)
    if args.tile is not None:
        plan.tile_size = args.tile
    log = logging.getLogger("restaura")
    log.info("preset=%s backend=%s precisão=%s tile=%s (%s)", plan.preset.value, plan.backend.value,
             plan.precision, plan.tile_size, "; ".join(plan.reasons))

    stages = []
    sr = build_sr_stage(args.model, args.scale, plan) if args.scale > 1 else None
    if sr:
        stages.append(sr)
    if args.interpolate:
        stages.append(BlendInterpolate())

    encoder = args.encoder
    if encoder == "auto":
        encoder = "hevc_nvenc" if plan.nvenc and _encoder_available("hevc_nvenc") else "libx264"

    def progress(done, total, fps):
        if args.json:
            print(json.dumps({"event": "progress", "done": done, "total": total, "fps": round(fps, 2)}),
                  flush=True)
        else:
            pct = 100 * done / total if total else 0
            print(f"\r{done}/{total} frames ({pct:5.1f}%) {fps:6.2f} fps", end="", file=sys.stderr)

    cfg = PipelineConfig(
        input=Path(args.input), output=Path(args.output), stages=stages,
        batch_size=plan.batch_size,
        deinterlace={"auto": None, "on": True, "off": False}[args.deinterlace],
        pre_filters=[DENOISE_FILTERS[plan.denoise]] if not args.no_denoise and plan.denoise in DENOISE_FILTERS else [],
        encoder=encoder, crf=args.crf, progress=progress)
    out = Pipeline(cfg).run()
    if args.json:
        print(json.dumps({"event": "done", "output": str(out)}), flush=True)
    else:
        print(f"\nOK -> {out}", file=sys.stderr)
    return 0


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

    hw = detect_hardware()
    plan = choose_plan(hw, Preset(args.preset) if args.preset else None)
    if args.tile is not None:
        plan.tile_size = args.tile
    st = build_sr_stage(args.model, args.scale, plan)
    if isinstance(st, OnnxUpscale):
        label = f"{st.model_path.name} via {st.session.get_providers()[0]} ({st.dtype.__name__})"
    else:
        label = "redimensionamento clássico (sem modelo)"

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:480, 0:720]
    base = np.stack([(xx * 255 // 720), (yy * 255 // 480), ((xx + yy) % 256)], -1).astype(np.uint8)
    frames = [np.clip(base.astype(np.int16) + rng.integers(-20, 20, base.shape), 0, 255).astype(np.uint8)
              for _ in range(4)]

    st.process(frames[0][None])  # aquecimento (TensorRT/DirectML compilam aqui)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        st.process(frames[n % 4][None]); n += 1
    fps = n / (time.perf_counter() - t0)
    hours = 3600 * 29.97 / fps / 3600

    g = hw.best_gpu
    result = {
        "gpu": g.name if g else "CPU", "preset": plan.preset.value, "backend": plan.backend.value,
        "estagio": label, "tile": plan.tile_size, "escala": args.scale,
        "fps": round(fps, 2), "horas_por_hora_de_fita": round(hours, 2),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"GPU:      {result['gpu']}")
        print(f"Preset:   {plan.preset.value} ({plan.backend.value}, tile {plan.tile_size})")
        print(f"Estágio:  {label}, {args.scale}x")
        print(f"Medido:   {fps:.2f} frames/s em 720x480 ({n} frames)")
        print(f"Estimativa: 1 h de fita (29,97 fps) leva ~{_fmt_hours(hours)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="engine", description="Restauração de vídeo adaptativa ao hardware")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detecta hardware e mostra o plano")
    d.add_argument("--preset", choices=[x.value for x in Preset])
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_detect)

    r = sub.add_parser("run", help="processa um vídeo")
    r.add_argument("input"); r.add_argument("output")
    r.add_argument("--preset", choices=[x.value for x in Preset])
    r.add_argument("--model", help="nome do catálogo ou caminho .onnx (padrão: o do preset)")
    r.add_argument("--scale", type=int, default=2)
    r.add_argument("--tile", type=int)
    r.add_argument("--interpolate", action="store_true", help="dobra o FPS")
    r.add_argument("--deinterlace", choices=["auto", "on", "off"], default="auto")
    r.add_argument("--no-denoise", action="store_true")
    r.add_argument("--encoder", default="auto")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--json", action="store_true", help="eventos JSON no stdout (para a UI)")
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
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
