"""
Baixa o FFmpeg para Windows (builds do BtbN, variante GPL "shared", versão estável mais nova)
e copia ffmpeg.exe, ffprobe.exe, as DLLs e a licença para a pasta de destino.

    python packaging/fetch_ffmpeg.py build/ffmpeg
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/latest"
PATTERN = re.compile(r"^ffmpeg-n(\d+)\.(\d+)(?:\.\d+)?-latest-win64-gpl-shared-[\d.]+\.zip$")


def _get(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "videoimprove-build"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main(dest: str) -> int:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    release = json.loads(_get(API))
    candidates = []
    for a in release["assets"]:
        m = PATTERN.match(a["name"])
        if m:
            candidates.append(((int(m.group(1)), int(m.group(2))), a))
    if not candidates:
        names = "\n  ".join(a["name"] for a in release["assets"] if "win64" in a["name"])
        raise SystemExit(f"nenhum zip win64-gpl-shared estável encontrado. Disponíveis:\n  {names}")
    version, asset = max(candidates, key=lambda c: c[0])
    print(f"FFmpeg {version[0]}.{version[1]}: {asset['name']} ({asset['size'] / 2**20:.0f} MB)")
    data = _get(asset["browser_download_url"], "application/octet-stream")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if "/bin/" in name and base.lower().endswith((".exe", ".dll")) and base.lower() != "ffplay.exe":
                (out / base).write_bytes(z.read(name))
            elif base.upper() in ("LICENSE.TXT", "LICENSE"):
                (out / "FFMPEG-LICENSE.txt").write_bytes(z.read(name))
    (out / "FFMPEG-VERSION.txt").write_text(f"{asset['name']}\n{asset['browser_download_url']}\n"
                                            "Código-fonte: https://ffmpeg.org/download.html "
                                            "e https://github.com/BtbN/FFmpeg-Builds\n", encoding="utf-8")
    exes = sorted(p.name for p in out.glob("*.exe"))
    print("copiados:", exes, f"+ {len(list(out.glob('*.dll')))} DLLs")
    if "ffmpeg.exe" not in exes or "ffprobe.exe" not in exes:
        raise SystemExit("ffmpeg.exe/ffprobe.exe ausentes no zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "build/ffmpeg"))
