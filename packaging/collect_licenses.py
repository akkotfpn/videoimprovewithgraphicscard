"""
Junta as licenças de tudo que vai dentro do app portátil na pasta licenses/.

    python packaging/collect_licenses.py dist/VideoImprove/licenses build/ffmpeg
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("onnxruntime", "PySide6", "shiboken6", "numpy", "pillow", "flatbuffers", "protobuf", "nvidia")
REAL_ESRGAN = "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/LICENSE"


def main(dest: str, ffmpeg_dir: str) -> int:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "LICENSE", out / "VideoImprove-Apache-2.0.txt")

    for f in Path(ffmpeg_dir).glob("FFMPEG-*.txt"):
        shutil.copy(f, out / f.name)

    try:
        with urllib.request.urlopen(REAL_ESRGAN, timeout=60) as r:
            (out / "Real-ESRGAN-BSD-3-Clause.txt").write_bytes(r.read())
    except OSError as e:
        print(f"aviso: não baixei a licença do Real-ESRGAN: {e}")
        (out / "Real-ESRGAN-BSD-3-Clause.txt").write_text(
            "Real-ESRGAN: BSD 3-Clause License. https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE\n")

    count = 0
    for dist in metadata.distributions():
        name = dist.metadata["Name"] or ""
        if not any(name.lower().startswith(p.lower()) for p in PACKAGES):
            continue
        for f in dist.files or []:
            fname = f.name.upper()
            if any(k in fname for k in ("LICENSE", "LICENCE", "COPYING", "NOTICE", "EULA")) and \
                    not str(f).endswith(".py"):
                src = Path(dist.locate_file(f))
                if src.is_file():
                    shutil.copy(src, out / f"{name}-{f.name}")
                    count += 1
    print(f"{count} arquivos de licença de pacotes copiados para {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
