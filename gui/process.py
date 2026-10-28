"""
Roda a engine como processo filho e entrega as linhas de saída como sinais Qt.

Não usa QProcess porque, no Windows, um app de janela chamando um executável de console abre uma
janela preta; com subprocess dá para passar CREATE_NO_WINDOW. As leituras ficam em threads Python que
só colocam as linhas numa fila; um QTimer na thread da interface esvazia a fila e emite os sinais.
Assim nenhum código de interface roda fora da thread principal.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading

from PySide6.QtCore import QObject, QTimer, Signal

from engine.core import paths


class EngineProcess(QObject):
    event = Signal(dict)      # linhas JSON do stdout
    log = Signal(str)         # stderr e linhas não-JSON
    finished = Signal(int)    # código de saída

    def __init__(self, args: list[str], parent=None):
        super().__init__(parent)
        self.args = args
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self._q: queue.Queue = queue.Queue()
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._drain)
        self._finished = False

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self._q.get_nowait()
            except queue.Empty:
                return
            if kind == "event":
                self.event.emit(payload)
            elif kind == "log":
                self.log.emit(payload)
            elif kind == "finished":
                self._timer.stop()
                self._finished = True
                self.finished.emit(payload)
                return

    def start(self) -> None:
        cmd = paths.cli_command() + self.args
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.log.emit("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                env=env, cwd=str(paths.app_dir()), creationflags=paths.no_window_flags(),
                encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            self.log.emit(f"não foi possível iniciar a engine: {e}")
            self._finished = True
            QTimer.singleShot(0, lambda: self.finished.emit(-1))
            return
        self._timer.start()
        t_out = threading.Thread(target=self._read_stdout, daemon=True)
        t_err = threading.Thread(target=self._read_stderr, daemon=True)
        t_out.start()
        t_err.start()
        threading.Thread(target=self._wait, args=(t_out, t_err), daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        buf: list[str] = []
        for line in self.proc.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("{"):
                try:
                    self._q.put(("event", json.loads(line)))
                    continue
                except json.JSONDecodeError:
                    pass
            buf.append(line)
            self._q.put(("log", line))
        if buf:
            # saída com várias linhas (ex.: selftest sem --json) também vira evento
            try:
                self._q.put(("event", json.loads("\n".join(buf))))
            except json.JSONDecodeError:
                pass

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            line = line.rstrip("\r\n")
            if line:
                self._q.put(("log", line))

    def _wait(self, *threads: threading.Thread) -> None:
        assert self.proc
        code = self.proc.wait()
        for t in threads:
            t.join(timeout=5)
        self._q.put(("finished", code))

    def kill(self) -> None:
        """Encerra a engine e os ffmpeg filhos."""
        self.cancelled = True
        if not self.proc or self.proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=paths.no_window_flags())
        else:
            self.proc.kill()

    @property
    def running(self) -> bool:
        """Verdadeiro até o sinal `finished` ser entregue (evita corrida com a fila)."""
        return self.proc is not None and not self._finished
