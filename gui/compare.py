"""Comparação antes/depois com divisória arrastável e diálogo de prévia."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)


class CompareWidget(QWidget):
    """Mostra `antes` à esquerda e `depois` à direita da divisória; arraste para comparar."""

    def __init__(self, before: QPixmap, after: QPixmap, parent=None):
        super().__init__(parent)
        self.before, self.after = before, after
        self.split = 0.5
        self.actual_size = False
        self.setMouseTracking(True)
        self.setCursor(Qt.SplitHCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 200)
        self._cache: tuple[QSize, QPixmap, QPixmap] | None = None

    def set_actual_size(self, on: bool) -> None:
        self.actual_size = on
        if on:
            self.setFixedSize(self.after.size())
        else:
            self.setMinimumSize(320, 200)
            self.setMaximumSize(16777215, 16777215)
        self._cache = None
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(960, 640)

    def _image_rect(self) -> QRectF:
        if self.actual_size:
            return QRectF(0, 0, self.after.width(), self.after.height())
        img = self.after.size().scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - img.width()) / 2
        y = (self.height() - img.height()) / 2
        return QRectF(x, y, img.width(), img.height())

    def _scaled(self, rect: QRectF) -> tuple[QPixmap, QPixmap]:
        size = rect.size().toSize()
        if self._cache is None or self._cache[0] != size:
            if self.actual_size:
                self._cache = (size, self.before, self.after)
            else:
                self._cache = (size,
                               self.before.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
                               self.after.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        return self._cache[1], self._cache[2]

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(20, 20, 22))
        rect = self._image_rect()
        before, after = self._scaled(rect)
        x_split = rect.left() + rect.width() * self.split
        p.drawPixmap(rect.topLeft(), before)
        p.save()
        p.setClipRect(QRectF(x_split, rect.top(), rect.right() - x_split, rect.height()))
        p.drawPixmap(rect.topLeft(), after)
        p.restore()

        # divisória + alça
        p.setPen(QPen(QColor(255, 255, 255, 230), 2))
        p.drawLine(QPointF(x_split, rect.top()), QPointF(x_split, rect.bottom()))
        cy = rect.center().y()
        p.setBrush(QColor(255, 255, 255, 235))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(x_split, cy), 14, 14)
        p.setPen(QPen(QColor(40, 40, 45), 2))
        for d in (-1, 1):
            path = QPainterPath()
            path.moveTo(x_split + d * 4, cy - 5)
            path.lineTo(x_split + d * 9, cy)
            path.lineTo(x_split + d * 4, cy + 5)
            p.drawPath(path)

        # rótulos
        font = QFont(self.font())
        font.setBold(True)
        p.setFont(font)
        for text, left in (("Antes", True), ("Depois", False)):
            w = p.fontMetrics().horizontalAdvance(text) + 20
            h = p.fontMetrics().height() + 8
            x = rect.left() + 12 if left else rect.right() - w - 12
            box = QRectF(x, rect.top() + 12, w, h)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(box, h / 2, h / 2)
            p.setPen(QColor(255, 255, 255))
            p.drawText(box, Qt.AlignCenter, text)

    def _move(self, x: float) -> None:
        rect = self._image_rect()
        if rect.width() > 0:
            self.split = min(1.0, max(0.0, (x - rect.left()) / rect.width()))
            self.update()

    def mousePressEvent(self, e):
        self._move(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._move(e.position().x())

    def resizeEvent(self, e):
        self._cache = None
        super().resizeEvent(e)


class PreviewDialog(QDialog):
    def __init__(self, before_path: str, after_path: str, info_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prévia: antes e depois")
        self.resize(1100, 760)
        self.compare = CompareWidget(QPixmap(before_path), QPixmap(after_path))
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.compare)
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setFrameShape(QScrollArea.NoFrame)

        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        actual = QCheckBox("Tamanho real (100%)")
        actual.toggled.connect(self._toggle_actual)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        bottom = QHBoxLayout()
        bottom.addWidget(info, 1)
        bottom.addWidget(actual)
        bottom.addWidget(buttons)
        layout = QVBoxLayout(self)
        hint = QLabel("Arraste a linha branca para comparar.")
        hint.setStyleSheet("color: palette(placeholder-text);")
        layout.addWidget(hint)
        layout.addWidget(self.scroll, 1)
        layout.addLayout(bottom)

    def _toggle_actual(self, on: bool):
        self.scroll.setWidgetResizable(not on)
        self.compare.set_actual_size(on)
        if on:
            bar = self.scroll.horizontalScrollBar()
            bar.setValue(int(bar.maximum() * self.compare.split))
            vbar = self.scroll.verticalScrollBar()
            vbar.setValue(vbar.maximum() // 2)
