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
    if args.model and Path(args.model).exists():
        stages.append(OnnxUpscale(args.model, args.scale, plan.backend, tile=plan.tile_size))
    elif args.scale > 1:
        log.warning("sem modelo ONNX: usando redimensionamento clássico (sem IA)")
        stages.append(ResizeUpscale(args.scale))
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
    r.add_argument("--model", help="modelo SR .onnx")
    r.add_argument("--scale", type=int, default=2)
    r.add_argument("--tile", type=int)
    r.add_argument("--interpolate", action="store_true", help="dobra o FPS")
    r.add_argument("--deinterlace", choices=["auto", "on", "off"], default="auto")
    r.add_argument("--no-denoise", action="store_true")
    r.add_argument("--encoder", default="auto")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--json", action="store_true", help="eventos JSON no stdout (para a UI)")
    r.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
