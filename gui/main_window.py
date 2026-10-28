"""Janela principal do VideoImprove."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSettings, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QRadioButton, QSplitter, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from engine import __version__
from engine.core import paths
from engine.core.hardware import Backend, HardwareProfile, Preset, choose_plan

from .compare import PreviewDialog
from .process import EngineProcess

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".mpg", ".mpeg", ".m2ts", ".mts", ".ts", ".vob", ".dv",
              ".wmv", ".m4v", ".3gp", ".flv", ".webm", ".m2v", ".mod", ".tod"}

PRESETS = [("auto", "Automática"), ("ultra", "Ultra"), ("equilibrado", "Equilibrada"),
           ("economico", "Econômica (PCs modestos)"), ("cpu", "Sem IA (só filtros, rápido)")]
MODELS = [("auto", "Automático"), ("realesr_general_x4v3", "Leve e rápido"),
          ("realesrgan_x4plus", "Detalhado (mais lento)")]
SCALES = [(2, "2× (recomendado para VHS)"), (4, "4×"), (1, "1× (só restaurar, sem ampliar)")]
DENOISE = [("auto", "Automática"), ("off", "Desligada"), ("light", "Leve"), ("medium", "Média"),
           ("strong", "Forte")]
BACKEND_NAMES = {"tensorrt": "TensorRT", "cuda": "CUDA", "directml": "DirectML", "openvino": "OpenVINO",
                 "rocm": "ROCm", "ncnn_vulkan": "Vulkan", "cpu": "CPU"}
PRESET_NAMES = {"ultra": "Ultra", "equilibrado": "Equilibrada", "economico": "Econômica", "cpu": "Sem IA"}
MODEL_NAMES = {"classico": "nenhum (filtros clássicos)", "realesr_general_x4v3": "Leve"}


def num(x: float, digits: int = 1) -> str:
    """Número com vírgula decimal (pt-BR)."""
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m = s // 60
    if m < 60:
        return f"{m} min"
    return f"{m // 60} h {m % 60:02d} min"


# --------------------------------------------------------------------------- #
# fila
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    path: Path
    item: QTreeWidgetItem
    status: str = "queued"      # queued | running | done | error | cancelled
    duration: float | None = None
    size: str = ""
    output: Path | None = None
    error: str = ""
    extra: dict = field(default_factory=dict)


class _ProbeSignals(QObject):
    done = Signal(str, dict)


class ProbeTask(QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = _ProbeSignals()

    def run(self):
        from engine.core.pipeline import probe
        try:
            info = probe(self.path)
            res = {"duration": info.duration, "size": f"{info.width}×{info.height}",
                   "fps": float(info.fps), "interlaced": info.interlaced}
        except Exception as e:  # noqa: BLE001
            res = {"error": str(e)[:300]}
        self.signals.done.emit(self.path, res)


class QueueTree(QTreeWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setColumnCount(4)
        self.setHeaderLabels(["Vídeo", "Duração", "Resolução", "Situação"])
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        h = self.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Interactive)
        self.setColumnWidth(3, 190)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if files:
            self.files_dropped.emit(files)
            e.acceptProposedAction()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.topLevelItemCount() == 0:
            p = QPainter(self.viewport())
            p.setPen(self.palette().placeholderText().color())
            f = p.font()
            f.setPointSizeF(f.pointSizeF() * 1.15)
            p.setFont(f)
            p.drawText(self.viewport().rect(), Qt.AlignCenter,
                       "Arraste os vídeos para cá\nou clique em “Adicionar vídeos”")


# --------------------------------------------------------------------------- #
# janela
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoImprove")
        self.resize(1120, 700)
        self.setAcceptDrops(True)
        self.settings = QSettings(str(paths.data_dir() / "settings.ini"), QSettings.IniFormat)
        self.hw: HardwareProfile | None = None
        self.jobs: list[Job] = []
        self.current: Job | None = None
        self.proc: EngineProcess | None = None
        self.side_proc: EngineProcess | None = None   # prévia / verificação
        self.pool = QThreadPool.globalInstance()
        self._probe_tasks: list[ProbeTask] = []
        self._shot = ScreenshotDriver.from_env(self)

        self._build_ui()
        self._load_settings()
        self._update_buttons()
        QTimer.singleShot(0, self._detect)

    # ---------------- construção ----------------
    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        # cabeçalho
        header = QHBoxLayout()
        title = QLabel("VideoImprove")
        tf = title.font()
        tf.setPointSizeF(tf.pointSizeF() * 1.6)
        tf.setBold(True)
        title.setFont(tf)
        subtitle = QLabel("Restauração de vídeos antigos com IA")
        subtitle.setStyleSheet("color: palette(placeholder-text);")
        self.hw_chip = QLabel("Detectando placa de vídeo…")
        self.hw_chip.setObjectName("chip")
        self.hw_chip.setTextInteractionFlags(Qt.TextSelectableByMouse)
        menu_btn = QToolButton()
        menu_btn.setText("Mais")
        menu_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(menu_btn)
        menu.addAction("Detalhes do hardware…", self._show_hw_details)
        menu.addAction("Verificar instalação…", self._selftest)
        menu.addSeparator()
        menu.addAction("Abrir pasta de configurações", lambda: self._open_path(paths.data_dir()))
        menu.addAction("Sobre o VideoImprove…", self._about)
        menu_btn.setMenu(menu)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        header.addWidget(self.hw_chip)
        header.addWidget(menu_btn)
        outer.addLayout(header)

        # centro: fila | ajustes
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.tree = QueueTree()
        self.tree.files_dropped.connect(self.add_files)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemDoubleClicked.connect(self._open_item_result)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        lv.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.btn_add = QPushButton("Adicionar vídeos…")
        self.btn_add.clicked.connect(self._pick_files)
        self.btn_remove = QPushButton("Remover")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear = QPushButton("Limpar concluídos")
        self.btn_clear.clicked.connect(self._clear_finished)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_remove)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        lv.addLayout(row)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        box = QGroupBox("Ajustes")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.cb_gpu = QComboBox()
        self.cb_preset = self._combo(PRESETS)
        self.cb_model = self._combo(MODELS)
        self.cb_scale = self._combo(SCALES)
        self.cb_denoise = self._combo(DENOISE)
        self.chk_smooth = QCheckBox("Movimento fluido (60 fps)")
        self.chk_smooth.setToolTip(
            "Em fitas VHS (entrelaçadas), usa cada campo da imagem como um quadro: dobra a fluidez com o "
            "movimento real gravado, sem inventar quadros. Em vídeos progressivos, interpola.")
        form.addRow("Placa de vídeo", self.cb_gpu)
        form.addRow("Qualidade", self.cb_preset)
        form.addRow("Modelo de IA", self.cb_model)
        form.addRow("Ampliação", self.cb_scale)
        form.addRow("Redução de ruído", self.cb_denoise)
        form.addRow("", self.chk_smooth)
        for w in (self.cb_gpu, self.cb_preset, self.cb_model, self.cb_scale, self.cb_denoise):
            w.currentIndexChanged.connect(self._settings_changed)
        self.chk_smooth.toggled.connect(self._settings_changed)
        rv.addWidget(box)

        out_box = QGroupBox("Onde salvar")
        ov = QVBoxLayout(out_box)
        self.rb_same = QRadioButton("Na mesma pasta de cada vídeo")
        self.rb_custom = QRadioButton("Nesta pasta:")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_same)
        grp.addButton(self.rb_custom)
        self.rb_same.setChecked(True)
        orow = QHBoxLayout()
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Escolha uma pasta")
        btn_browse = QPushButton("Procurar…")
        btn_browse.clicked.connect(self._pick_out_dir)
        orow.addWidget(self.ed_out, 1)
        orow.addWidget(btn_browse)
        ov.addWidget(self.rb_same)
        ov.addWidget(self.rb_custom)
        ov.addLayout(orow)
        note = QLabel("O arquivo sai como <i>nome</i>_restaurado.mkv, com o áudio original.")
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(placeholder-text);")
        ov.addWidget(note)
        self.rb_custom.toggled.connect(self._settings_changed)
        rv.addWidget(out_box)

        self.plan_label = QLabel("")
        self.plan_label.setWordWrap(True)
        self.plan_label.setTextFormat(Qt.RichText)
        self.plan_label.setObjectName("plan")
        rv.addWidget(self.plan_label)
        self.btn_preview = QPushButton("Ver prévia de um quadro")
        self.btn_preview.setToolTip("Processa só um quadro do vídeo selecionado e mostra antes/depois, "
                                    "com a estimativa de tempo para o vídeo inteiro.")
        self.btn_preview.clicked.connect(self._preview)
        rv.addWidget(self.btn_preview)
        rv.addStretch(1)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([680, 420])
        outer.addWidget(split, 1)

        # rodapé
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        outer.addWidget(line)
        foot = QHBoxLayout()
        status_box = QVBoxLayout()
        self.status = QLabel("Adicione vídeos para começar.")
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("")
        status_box.addWidget(self.status)
        status_box.addWidget(self.progress)
        foot.addLayout(status_box, 1)
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_start = QPushButton("Restaurar")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumWidth(150)
        self.btn_start.setMinimumHeight(38)
        self.btn_start.clicked.connect(self._start)
        foot.addWidget(self.btn_cancel)
        foot.addWidget(self.btn_start)
        outer.addLayout(foot)

        log_row = QHBoxLayout()
        self.btn_log = QToolButton()
        self.btn_log.setText("Mostrar registro")
        self.btn_log.setCheckable(True)
        self.btn_log.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_log.setArrowType(Qt.RightArrow)
        self.btn_log.toggled.connect(self._toggle_log)
        self.btn_open_out = QPushButton("Abrir pasta do resultado")
        self.btn_open_out.clicked.connect(self._open_last_output)
        self.btn_open_out.setVisible(False)
        log_row.addWidget(self.btn_log)
        log_row.addStretch(1)
        log_row.addWidget(self.btn_open_out)
        outer.addLayout(log_row)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setVisible(False)
        self.log_view.setMinimumHeight(140)
        f = self.log_view.font()
        f.setFamily("Consolas")
        self.log_view.setFont(f)
        outer.addWidget(self.log_view)

        self.setCentralWidget(root)
        self.setStyleSheet("""
            QLabel#chip { padding: 6px 12px; border-radius: 13px;
                          background: palette(alternate-base); border: 1px solid palette(mid); }
            QPushButton#primary { font-weight: 600; padding: 6px 18px; }
            QLabel#plan { padding: 8px 10px; border-radius: 6px; background: palette(alternate-base); }
        """)

    @staticmethod
    def _combo(options) -> QComboBox:
        cb = QComboBox()
        for value, label in options:
            cb.addItem(label, value)
        return cb

    # ---------------- configurações ----------------
    def _load_settings(self):
        s = self.settings
        for cb, key in ((self.cb_preset, "preset"), (self.cb_model, "model"), (self.cb_scale, "scale"),
                        (self.cb_denoise, "denoise")):
            v = s.value(key)
            if v is None:
                continue
            for i in range(cb.count()):
                if str(cb.itemData(i)) == str(v):
                    cb.blockSignals(True)
                    cb.setCurrentIndex(i)
                    cb.blockSignals(False)
                    break
        self.chk_smooth.blockSignals(True)
        self.chk_smooth.setChecked(str(s.value("smooth", "false")) == "true")
        self.chk_smooth.blockSignals(False)
        self.ed_out.setText(s.value("out_dir", "") or "")
        if s.value("out_mode", "same") == "custom":
            self.rb_custom.blockSignals(True)
            self.rb_custom.setChecked(True)
            self.rb_custom.blockSignals(False)

    def _save_settings(self):
        s = self.settings
        s.setValue("preset", self.cb_preset.currentData())
        s.setValue("model", self.cb_model.currentData())
        s.setValue("scale", self.cb_scale.currentData())
        s.setValue("denoise", self.cb_denoise.currentData())
        s.setValue("smooth", "true" if self.chk_smooth.isChecked() else "false")
        s.setValue("out_dir", self.ed_out.text())
        s.setValue("out_mode", "custom" if self.rb_custom.isChecked() else "same")
        s.setValue("gpu", self.cb_gpu.currentData() if self.cb_gpu.currentData() is not None else -1)

    def _settings_changed(self, *_):
        self._update_plan()
        self._save_settings()

    # ---------------- hardware ----------------
    def _detect(self):
        p = EngineProcess(["detect", "--json"], self)
        p.log.connect(self._log)
        result: dict = {}
        p.event.connect(result.update)
        p.finished.connect(lambda code: self._detected(result, code))
        self._detect_proc = p
        p.start()

    def _detected(self, result: dict, code: int):
        if code != 0 or "hardware" not in result:
            self.hw_chip.setText("Não foi possível detectar a placa (usando CPU)")
            self.hw = HardwareProfile("?", "?", os.cpu_count() or 1, 0)
        else:
            self.hw = HardwareProfile.from_dict(result["hardware"])
        self.cb_gpu.blockSignals(True)
        self.cb_gpu.clear()
        best = self.hw.best_gpu
        self.cb_gpu.addItem(f"Automática ({best.name})" if best else "Automática (sem GPU)", None)
        for i, g in enumerate(self.hw.gpus):
            kind = "integrada" if g.integrated else f"{g.vram_mb / 1024:.0f} GB"
            self.cb_gpu.addItem(f"{g.name} ({kind})", i)
        saved = self.settings.value("gpu", -1)
        try:
            saved = int(saved)
        except (TypeError, ValueError):
            saved = -1
        idx = self.cb_gpu.findData(saved) if saved >= 0 else 0
        self.cb_gpu.setCurrentIndex(max(0, idx))
        self.cb_gpu.blockSignals(False)
        self._update_plan()
        if self._shot:
            self._shot.on_ready()

    def _current_plan(self):
        if not self.hw:
            return None
        preset = self.cb_preset.currentData()
        try:
            return choose_plan(self.hw, None if preset == "auto" else Preset(preset), self.cb_gpu.currentData())
        except ValueError:
            return choose_plan(self.hw, None if preset == "auto" else Preset(preset))

    def _update_plan(self):
        plan = self._current_plan()
        if not plan:
            return
        backend = BACKEND_NAMES.get(plan.backend.value, plan.backend.value)
        gpu = plan.gpu_name or "CPU"
        self.hw_chip.setText(f"{gpu}  ·  {backend}" if plan.gpu_name else "Sem placa de vídeo compatível · CPU")
        self.hw_chip.setToolTip("\n".join(plan.reasons))
        model = self.cb_model.currentData()
        scale = self.cb_scale.currentData()
        if scale == 1:
            model_txt = "nenhum (sem ampliação)"
        elif model == "auto":
            model_txt = MODEL_NAMES.get(plan.sr_model, "Detalhado")
        else:
            model_txt = dict(MODELS)[model]
        warn = ""
        if plan.backend == Backend.CPU and (model != "auto" and scale > 1):
            warn = ("<br><b>Atenção:</b> sem aceleração de GPU, IA leva muitas horas por hora de vídeo. "
                    "Use a prévia para ver a estimativa.")
        elif plan.backend == Backend.NCNN_VULKAN:
            warn = "<br><b>Atenção:</b> esta placa ainda não tem suporte de IA nesta versão; a IA vai rodar na CPU."
        quality = PRESET_NAMES.get(plan.preset.value, plan.preset.value)
        if plan.preset == Preset.CPU and model != "auto" and scale > 1:
            quality = "processamento na CPU"
        where = f"{gpu} via {backend}" if plan.gpu_name else "o processador (CPU)"
        self.plan_label.setText(
            f"<b>Vai usar:</b> {where}<br>Qualidade {quality} · modelo {model_txt}{warn}")
        self.plan_label.setToolTip("\n".join(plan.reasons))

    def _show_hw_details(self):
        if not self.hw:
            return
        plan = self._current_plan()
        lines = [f"<b>Sistema:</b> {self.hw.os} · {self.hw.cpu_threads} threads · "
                 f"{self.hw.ram_mb / 1024:.0f} GB de RAM", "<b>Placas de vídeo:</b>"]
        for i, g in enumerate(self.hw.gpus):
            kind = "integrada" if g.integrated else "dedicada"
            lines.append(f"&nbsp;&nbsp;{i}: {g.name} ({kind}, {g.vram_mb} MB, origem: {g.source})")
        if not self.hw.gpus:
            lines.append("&nbsp;&nbsp;nenhuma")
        lines.append(f"<b>Runtimes de IA:</b> {', '.join(self.hw.ort_providers) or 'nenhum'}")
        if plan:
            lines.append(f"<b>Plano:</b> {plan.preset.value} · {plan.backend.value} · {plan.precision} · "
                         f"blocos {plan.tile_size or 'inteiro'} · device {plan.device_id}")
            lines.append("<b>Motivos:</b><br>&nbsp;&nbsp;" + "<br>&nbsp;&nbsp;".join(plan.reasons))
        QMessageBox.information(self, "Hardware detectado", "<br>".join(lines))

    # ---------------- fila ----------------
    def add_files(self, files: list[str]):
        known = {str(j.path) for j in self.jobs}
        added = 0
        for f in files:
            p = Path(f)
            if p.is_dir():
                self.add_files([str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in VIDEO_EXTS])
                continue
            if p.suffix.lower() not in VIDEO_EXTS or str(p) in known or not p.exists():
                continue
            item = QTreeWidgetItem([p.name, "…", "…", "Na fila"])
            item.setToolTip(0, str(p))
            self.tree.addTopLevelItem(item)
            job = Job(p, item)
            self.jobs.append(job)
            known.add(str(p))
            task = ProbeTask(str(p))
            task.signals.done.connect(self._probed)
            self._probe_tasks.append(task)
            self.pool.start(task)
            added += 1
        if added and not self._running():
            self.status.setText(f"{len(self._queued())} vídeo(s) na fila.")
        self._update_buttons()

    def _probed(self, path: str, res: dict):
        self._probe_tasks = [t for t in self._probe_tasks if t.path != path]
        for j in self.jobs:
            if str(j.path) == path:
                if "error" in res:
                    j.status, j.error = "error", "não consegui ler este arquivo"
                    j.item.setText(3, "Arquivo inválido")
                    j.item.setToolTip(3, res["error"])
                else:
                    j.duration, j.size = res["duration"], res["size"]
                    j.extra = res
                    j.item.setText(1, fmt_duration(res["duration"]))
                    tag = " (entrelaçado)" if res.get("interlaced") else ""
                    j.item.setText(2, res["size"] + tag)
        self._update_buttons()
        if self._shot and not self._probe_tasks:
            self._shot.on_probed()

    def _queued(self) -> list[Job]:
        return [j for j in self.jobs if j.status == "queued"]

    def _selected_jobs(self) -> list[Job]:
        sel = set(id(i) for i in self.tree.selectedItems())
        return [j for j in self.jobs if id(j.item) in sel]

    def _pick_files(self):
        start = self.settings.value("last_dir", str(Path.home()))
        pattern = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "Escolha os vídeos", start,
                                                f"Vídeos ({pattern});;Todos os arquivos (*)")
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_files(files)

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta de saída", self.ed_out.text() or str(Path.home()))
        if d:
            self.ed_out.setText(d)
            self.rb_custom.setChecked(True)
            self._save_settings()

    def _remove_selected(self):
        for j in self._selected_jobs():
            if j is self.current:
                continue
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(j.item))
            self.jobs.remove(j)
        self._update_buttons()

    def _clear_finished(self):
        for j in [j for j in self.jobs if j.status in ("done", "cancelled", "error")]:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(j.item))
            self.jobs.remove(j)
        self._update_buttons()

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        job = next((j for j in self.jobs if j.item is item), None)
        if not job:
            return
        m = QMenu(self)
        if job.output and job.output.exists():
            m.addAction("Abrir vídeo restaurado", lambda: self._open_path(job.output))
            m.addAction("Mostrar na pasta", lambda: self._open_path(job.output.parent))
        m.addAction("Ver prévia de um quadro", self._preview)
        if job.status in ("error", "cancelled", "done"):
            m.addAction("Processar de novo", lambda: self._requeue(job))
        if job is not self.current:
            m.addAction("Remover da fila", self._remove_selected)
        if job.error:
            m.addAction("Ver erro", lambda: QMessageBox.warning(self, "Erro", job.error))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _requeue(self, job: Job):
        job.status, job.error = "queued", ""
        job.item.setText(3, "Na fila")
        self._update_buttons()

    def _open_item_result(self, item, _col):
        job = next((j for j in self.jobs if j.item is item), None)
        if job and job.output and job.output.exists():
            self._open_path(job.output)

    # ---------------- processamento ----------------
    def _running(self) -> bool:
        return self.proc is not None and self.proc.running

    def _output_for(self, src: Path) -> Path:
        folder = Path(self.ed_out.text()) if self.rb_custom.isChecked() and self.ed_out.text() else src.parent
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{src.stem}_restaurado.mkv"
        n = 2
        while out.exists():
            out = folder / f"{src.stem}_restaurado ({n}).mkv"
            n += 1
        return out

    def _common_args(self) -> list[str]:
        args = ["--scale", str(self.cb_scale.currentData()), "--denoise", self.cb_denoise.currentData()]
        if self.cb_preset.currentData() != "auto":
            args += ["--preset", self.cb_preset.currentData()]
        if self.cb_model.currentData() != "auto":
            args += ["--model", self.cb_model.currentData()]
        if self.cb_gpu.currentData() is not None:
            args += ["--gpu", str(self.cb_gpu.currentData())]
        if self.chk_smooth.isChecked():
            args += ["--smooth"]
        return args

    def _start(self):
        if self.rb_custom.isChecked() and not self.ed_out.text():
            QMessageBox.information(self, "Pasta de saída", "Escolha a pasta onde salvar os vídeos.")
            return
        if not self._queued():
            return
        self._total_batch = len(self._queued())
        self._done_batch = 0
        self.btn_open_out.setVisible(False)
        self._next()

    def _next(self):
        queued = self._queued()
        if not queued:
            self._finish_batch()
            return
        job = queued[0]
        self.current = job
        try:
            job.output = self._output_for(job.path)
        except OSError as e:
            job.status, job.error = "error", f"não foi possível criar a pasta de saída: {e}"
            job.item.setText(3, "Erro")
            QTimer.singleShot(0, self._next)
            return
        job.status = "running"
        job.item.setText(3, "Iniciando…")
        self.tree.scrollToItem(job.item)
        n = self._done_batch + 1
        self.status.setText(f"Processando {n} de {self._total_batch}: {job.path.name}")
        self.progress.setRange(0, 0)  # indeterminado até o primeiro quadro
        self.proc = EngineProcess(["run", str(job.path), str(job.output), "--json"] + self._common_args(), self)
        self.proc.event.connect(self._on_event)
        self.proc.log.connect(self._log)
        self.proc.finished.connect(self._on_finished)
        self._update_buttons()
        self.proc.start()

    def _on_event(self, ev: dict):
        job = self.current
        if not job:
            return
        kind = ev.get("event")
        if kind == "start":
            job.extra["start"] = ev
            self._log(f"usando {ev.get('gpu') or 'CPU'} · {ev.get('backend')} · etapas: {', '.join(ev.get('stages', []))}")
        elif kind == "progress":
            total, done = ev.get("total") or 0, ev.get("done") or 0
            if total:
                self.progress.setRange(0, 1000)
                self.progress.setValue(int(1000 * done / total))
                pct = 100 * done / total
                self.progress.setFormat(f"{num(pct)}%")
                job.item.setText(3, f"Processando {pct:.0f}%")
            n = self._done_batch + 1
            self.status.setText(
                f"Processando {n} de {self._total_batch}: {job.path.name}  ·  {num(ev.get('fps', 0))} quadros/s  ·  "
                f"faltam ~{fmt_duration(ev.get('eta'))}")
            if self._shot:
                self._shot.on_progress(done)
        elif kind == "done":
            job.extra["done"] = True
        elif kind == "error":
            job.error = ev.get("message", "erro desconhecido")

    def _on_finished(self, code: int):
        job, proc = self.current, self.proc
        self.current = None
        if job is None or proc is None:
            return
        if proc.cancelled:
            job.status = "cancelled"
            job.item.setText(3, "Cancelado")
            if job.output and job.output.exists():
                try:
                    job.output.unlink()
                except OSError:
                    pass
            self._finish_batch(cancelled=True)
            return
        if code == 0 and job.extra.get("done"):
            job.status = "done"
            job.item.setText(3, "Concluído ✓")
            job.item.setToolTip(3, str(job.output))
            self._last_output = job.output
        else:
            job.status = "error"
            job.error = job.error or self._last_log_lines()
            job.item.setText(3, "Erro (clique com o botão direito)")
            job.item.setToolTip(3, job.error)
        self._done_batch += 1
        self._next()

    def _finish_batch(self, cancelled: bool = False):
        self.proc = None
        self.current = None
        self.progress.setRange(0, 1000)
        done = [j for j in self.jobs if j.status == "done"]
        errors = [j for j in self.jobs if j.status == "error"]
        if cancelled:
            self.status.setText("Cancelado.")
            self.progress.setValue(0)
            self.progress.setFormat("")
        else:
            self.progress.setValue(1000)
            self.progress.setFormat("100%")
            msg = f"Pronto: {len(done)} vídeo(s) restaurado(s)."
            if errors:
                msg += f" {len(errors)} com erro (clique com o botão direito para ver)."
            self.status.setText(msg)
            QApplication.alert(self)
        self.btn_open_out.setVisible(bool(getattr(self, "_last_output", None)))
        self._update_buttons()
        if self._shot:
            self._shot.on_batch_done()

    def _cancel(self):
        if self._running():
            self.status.setText("Cancelando…")
            self.proc.kill()
        if self.side_proc and self.side_proc.running:
            self.side_proc.kill()

    def _update_buttons(self):
        running = self._running()
        side = self.side_proc is not None and self.side_proc.running
        self.btn_start.setEnabled(not running and not side and bool(self._queued()) and self.hw is not None)
        self.btn_cancel.setEnabled(running or side)
        self.btn_remove.setEnabled(bool(self._selected_jobs()))
        self.btn_clear.setEnabled(any(j.status in ("done", "cancelled", "error") for j in self.jobs))
        has_sel = bool(self._selected_jobs()) or len(self.jobs) == 1
        self.btn_preview.setEnabled(has_sel and not running and not side and self.hw is not None)
        for w in (self.cb_gpu, self.cb_preset, self.cb_model, self.cb_scale, self.cb_denoise, self.chk_smooth,
                  self.rb_same, self.rb_custom, self.ed_out):
            w.setEnabled(not running)

    # ---------------- prévia ----------------
    def _preview(self):
        sel = self._selected_jobs() or (self.jobs[:1] if len(self.jobs) == 1 else [])
        if not sel:
            QMessageBox.information(self, "Prévia", "Selecione um vídeo da lista.")
            return
        job = sel[0]
        out = Path(tempfile.mkdtemp(prefix="previa_", dir=paths.data_dir()))
        args = ["preview", str(job.path), "--out", str(out), "--json"] + self._common_args()
        self.side_proc = EngineProcess(args, self)
        result: dict = {}
        self.side_proc.event.connect(result.update)
        self.side_proc.log.connect(self._log)
        self.side_proc.finished.connect(lambda code: self._preview_done(job, result, code, out))
        self.status.setText(f"Gerando prévia de {job.path.name}… (a primeira vez pode levar mais tempo)")
        self.progress.setRange(0, 0)
        self.side_proc.start()
        self._update_buttons()

    def _preview_done(self, job: Job, r: dict, code: int, out: Path):
        cancelled = self.side_proc.cancelled if self.side_proc else False
        self.side_proc = None
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self._update_buttons()
        if cancelled:
            self.status.setText("Prévia cancelada.")
            return
        if code != 0 or "after" not in r:
            self.status.setText("A prévia falhou. Veja o registro.")
            QMessageBox.warning(self, "Prévia", r.get("message") or self._last_log_lines())
            return
        self.status.setText("Prévia pronta.")
        spf, est = r.get("seconds_per_frame"), r.get("estimated_seconds")
        engine = f"{r.get('gpu') or 'CPU'}"
        if r.get("provider"):
            engine += f" ({r['provider'].replace('ExecutionProvider', '')})"
        parts = [f"Quadro em {fmt_duration(r.get('time'))} · {engine}"]
        if spf:
            parts.append(f"{num(spf, 2)} s por quadro · vídeo inteiro: <b>~{fmt_duration(est)}</b> só na IA")
        dlg = PreviewDialog(r["before"], r["after"], "<br>".join(parts), self)
        dlg.finished.connect(lambda *_: shutil.rmtree(out, ignore_errors=True))
        if self._shot:
            self._shot.on_preview(dlg)
        dlg.show()

    # ---------------- utilidades ----------------
    def _selftest(self):
        self.side_proc = EngineProcess(["selftest", "--json"], self)
        result: dict = {}
        self.side_proc.event.connect(result.update)
        self.side_proc.log.connect(self._log)
        self.status.setText("Verificando a instalação (leva alguns segundos)…")
        self.progress.setRange(0, 0)

        def done(code):
            self.side_proc = None
            self.progress.setRange(0, 1000)
            self._update_buttons()
            if result.get("ok"):
                self.status.setText("Instalação OK.")
                QMessageBox.information(self, "Verificar instalação",
                                        "Tudo certo: FFmpeg, modelos e IA funcionando.\n\n"
                                        f"Runtime de IA: {result.get('sr_provider', '?')}\n"
                                        f"Modelos: {', '.join(result.get('models', []))}")
            else:
                self.status.setText("A verificação encontrou problemas.")
                QMessageBox.warning(self, "Verificar instalação",
                                    json.dumps(result, ensure_ascii=False, indent=2) or self._last_log_lines())
        self.side_proc.finished.connect(done)
        self.side_proc.start()
        self._update_buttons()

    def _about(self):
        QMessageBox.about(self, "Sobre o VideoImprove", f"""
            <h3>VideoImprove {__version__}</h3>
            <p>Restauração e ampliação de vídeos antigos (VHS, Hi8, filmes caseiros) com IA,
            adaptada à placa de vídeo de cada PC.</p>
            <p>Código aberto sob a licença Apache-2.0.</p>
            <p><b>Componentes de terceiros</b><br>
            Modelos Real-ESRGAN (BSD-3-Clause, Xintao Wang et al.)<br>
            FFmpeg (GPL; executável separado em <i>bin</i>, código-fonte em ffmpeg.org)<br>
            ONNX Runtime (MIT, Microsoft) · Qt / PySide6 (LGPL-3.0)<br>
            Versão NVIDIA: CUDA e cuDNN redistribuídos sob os termos da NVIDIA.</p>
            <p>As licenças completas estão na pasta <i>licenses</i>.</p>""")

    def _log(self, line: str):
        self.log_view.appendPlainText(line)

    def _last_log_lines(self, n: int = 12) -> str:
        lines = self.log_view.toPlainText().splitlines()
        relevant = [l for l in lines if not l.startswith("$ ")]
        return "\n".join(relevant[-n:]) or "erro desconhecido (veja o registro)"

    def _toggle_log(self, on: bool):
        self.log_view.setVisible(on)
        self.btn_log.setText("Ocultar registro" if on else "Mostrar registro")
        self.btn_log.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def _open_last_output(self):
        out = getattr(self, "_last_output", None)
        if out:
            self._open_path(out.parent)

    @staticmethod
    def _open_path(p: Path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    # ---------------- arrastar e soltar na janela ----------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def closeEvent(self, e):
        if self._running():
            box = QMessageBox(QMessageBox.Question, "Sair", "Há um vídeo sendo processado. Cancelar e sair?",
                              QMessageBox.Yes | QMessageBox.No, self)
            box.button(QMessageBox.Yes).setText("Cancelar e sair")
            box.button(QMessageBox.No).setText("Continuar processando")
            box.setDefaultButton(QMessageBox.No)
            if box.exec() != QMessageBox.Yes:
                e.ignore()
                return
            self.proc.kill()
        if self.side_proc and self.side_proc.running:
            self.side_proc.kill()
        self._save_settings()
        super().closeEvent(e)


# --------------------------------------------------------------------------- #
# modo de captura de tela (CI): percorre o app sozinho e salva prints
# --------------------------------------------------------------------------- #
class ScreenshotDriver:
    """
    Ativado por VIDEOIMPROVE_SCREENSHOT_DIR e VIDEOIMPROVE_SCREENSHOT_VIDEO. Abre o app, adiciona o vídeo,
    tira prints da fila, da prévia, do processamento e do fim, e fecha com código 0 se tudo deu certo.
    """

    def __init__(self, win: MainWindow, out: Path, video: str):
        self.win, self.out, self.video = win, out, video
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage = "start"
        self.events: list[str] = []
        QTimer.singleShot(15 * 60 * 1000, lambda: self._quit(3, "tempo esgotado"))

    @classmethod
    def from_env(cls, win: MainWindow):
        out, video = os.environ.get("VIDEOIMPROVE_SCREENSHOT_DIR"), os.environ.get("VIDEOIMPROVE_SCREENSHOT_VIDEO")
        return cls(win, Path(out), video) if out and video else None

    def _grab(self, widget, name: str):
        widget.grab().save(str(self.out / name))
        self.events.append(f"print {name}")

    def on_ready(self):
        # exercita o caminho de IA mesmo sem GPU (modelo leve, vídeo de teste pequeno) e o 60 fps
        w = self.win
        w.cb_model.setCurrentIndex(w.cb_model.findData("realesr_general_x4v3"))
        w.chk_smooth.setChecked(True)
        QTimer.singleShot(500, lambda: self._grab(self.win, "0-inicio.png"))
        QTimer.singleShot(800, lambda: self.win.add_files([self.video]))

    def on_probed(self):
        if self.stage != "start":
            return
        self.stage = "preview"
        self.win.tree.topLevelItem(0).setSelected(True)
        QTimer.singleShot(400, lambda: self._grab(self.win, "1-fila.png"))
        QTimer.singleShot(900, self.win._preview)

    def on_preview(self, dlg):
        def shoot():
            self._grab(dlg, "2-previa.png")
            dlg.accept()
            self.stage = "run"
            QTimer.singleShot(300, self.win._start)
        QTimer.singleShot(1200, shoot)

    def on_progress(self, done: int):
        if self.stage == "run" and done > 0:
            self.stage = "running"
            QTimer.singleShot(200, lambda: self._grab(self.win, "3-processando.png"))

    def on_batch_done(self):
        def finish():
            self._grab(self.win, "4-concluido.png")
            ok = all(j.status == "done" for j in self.win.jobs)
            self._quit(0 if ok else 1, "ok" if ok else "falha no processamento")
        QTimer.singleShot(600, finish)

    def _quit(self, code: int, reason: str):
        jobs = [{"file": j.path.name, "status": j.status, "error": j.error,
                 "output": str(j.output) if j.output else None} for j in self.win.jobs]
        (self.out / "resultado.json").write_text(json.dumps(
            {"code": code, "reason": reason, "events": self.events, "jobs": jobs,
             "log": self.win.log_view.toPlainText().splitlines()[-200:]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        QApplication.exit(code)


def run() -> int:
    paths.setup_environment()
    QApplication.setApplicationName("VideoImprove")
    QApplication.setOrganizationName("VideoImprove")
    app = QApplication(sys.argv)
    # textos padrão do Qt (botões OK/Cancelar, diálogos de arquivo) em português
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
    tr = QTranslator(app)
    if tr.load(QLocale("pt_BR"), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(tr)
    icon = paths.app_dir() / ("icon.ico" if os.name == "nt" else "icon.png")
    for cand in (icon, Path(__file__).resolve().parents[1] / "packaging" / "icon.png"):
        if cand.exists():
            app.setWindowIcon(QIcon(str(cand)))
            break
    win = MainWindow()
    win.show()
    files = [a for a in sys.argv[1:] if Path(a).exists()]
    if files:
        win.add_files(files)
    return app.exec()
