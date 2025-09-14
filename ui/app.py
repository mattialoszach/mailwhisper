import sys
import os
from typing import Dict, Any
from PySide6.QtCore import Qt, QObject, QThread, Signal, QSize, QTimer, QEvent, QPropertyAnimation, QPoint
from PySide6.QtGui import QFont, QAction, QPalette, QColor, QIcon, QTextOption, QPainter, QPen, QBrush, QPainterPath, QShortcut, QKeySequence
from PySide6.QtWidgets import (
QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
QLineEdit, QPlainTextEdit, QTextEdit, QPushButton, QScrollArea, QFrame, QToolButton,
QDialog, QDialogButtonBox, QSizePolicy, QAbstractButton, QComboBox, QFormLayout, QSpacerItem
)
import subprocess
from PySide6.QtWidgets import QGraphicsOpacityEffect
import numpy as np
from agent.state import initial_state, DraftState
from agent.nodes import intent_node, apply_node
from agent import nodes as nodes_mod
from utils.stt_whisper_mem import transcribe_array, set_whisper_model
from ui.recorder import ButtonControlledRecorder

class ProcessAudioWorker(QObject):
    finished = Signal(dict) # emits patch for state update
    failed = Signal(str)

    def __init__(self, audio: np.ndarray, samplerate: int, state: Dict[str, Any]):
        super().__init__()
        self.audio = audio
        self.samplerate = samplerate
        self.state_snapshot = dict(state)  # shallow copy
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            text, detected_lang = transcribe_array(self.audio, samplerate=self.samplerate)
            text = (text or "").strip()
            if not text:
                self.failed.emit("No Transcription recognized.")
                return
            if self._cancelled:
                self.finished.emit({})
                return

            # 1) set transcript
            self.state_snapshot["transcript"] = text

            # 2) determine intent (LLM)
            if self._cancelled:
                patch_intent = {}
            else:
                patch_intent = intent_node(self.state_snapshot)
            self.state_snapshot.update(patch_intent)  # last_op, intent

            # 3) apply to draft
            if self._cancelled:
                patch_apply = {}
            else:
                patch_apply = apply_node(self.state_snapshot)

            # Combine: return intent+apply (UI primarily needs apply fields)
            patch = {}
            patch.update(patch_intent or {})
            patch.update(patch_apply or {})
            self.finished.emit(patch)

        except Exception as e:
            self.failed.emit(str(e))

class FieldCard(QWidget):
    def __init__(self, title: str, multiline: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(6)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("cardTitle")
        self.title_lbl.setContentsMargins(0, 2, 0, 0)
        # Uniform label column width for consistent editor width
        LABEL_COL_WIDTH = 82
        self.title_lbl.setMinimumWidth(LABEL_COL_WIDTH)

        self.copy_btn = QPushButton()
        self.copy_btn.setObjectName("copyBtn")
        # Use custom SoftToolTip; keep native tooltip empty
        self.copy_btn.setToolTip("")
        self.copy_btn.setProperty("softTip", "Copy")
        self._copy_icon = QIcon("ui/icons/copy.svg")
        self._check_icon = QIcon("ui/icons/check.svg")
        self.copy_btn.setIcon(self._copy_icon)
        self.copy_btn.setIconSize(QSize(18, 18))
        self.copy_btn.setText("")

        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(0)

        if multiline:
            # Use auto-resizing QTextEdit (wraps, no inner scrollbars)
            self.editor = AutoResizingTextEdit(min_height=60)
            self.editor.setObjectName("bodyText")
        else:
            self.editor = QLineEdit()
            self.editor.setReadOnly(True)
            self.editor.setMinimumHeight(32)

        grid.addWidget(self.title_lbl, 0, 0, alignment=Qt.AlignTop | Qt.AlignRight)
        grid.addWidget(self.editor,    0, 1)
        grid.addWidget(self.copy_btn,  0, 2, alignment=Qt.AlignTop | Qt.AlignRight)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid)

        # Connect copy interaction
        self.copy_btn.clicked.connect(self._on_copy_clicked)
        # Install custom tooltip behavior for the copy button
        self.copy_btn.installEventFilter(self)
        self._tip_timer = QTimer(self)
        self._tip_timer.setSingleShot(True)
        self._tip_timer.timeout.connect(self._show_copy_tip)
        self._soft_tip = SoftToolTip(self)

    def set_text(self, text: str):
        e = self.editor
        t = text or ""
        # Prefer setPlainText when available (QTextEdit/QPlainTextEdit)
        if hasattr(e, "setPlainText") and not isinstance(e, QLineEdit):
            e.setPlainText(t)
        else:
            e.setText(t)
        # Trigger height recompute for auto-resizing widgets
        if hasattr(e, "update_height"):
            QTimer.singleShot(0, e.update_height)

    def text(self) -> str:
        # Prefer plain text if available (covers QTextEdit/QPlainTextEdit)
        try:
            if hasattr(self.editor, "toPlainText") and not isinstance(self.editor, QLineEdit):
                return self.editor.toPlainText()
            if hasattr(self.editor, "text"):
                return self.editor.text()
        except Exception:
            pass
        return ""

    def _on_copy_clicked(self):
        # Copy current text to clipboard
        QApplication.clipboard().setText(self.text())
        # Feedback: swap icon to check for 1.5s
        self.copy_btn.setIcon(self._check_icon)
        self.copy_btn.setProperty("softTip", "Copied!")
        QTimer.singleShot(1500, self._restore_copy_icon)

    def _restore_copy_icon(self):
        self.copy_btn.setIcon(self._copy_icon)
        self.copy_btn.setProperty("softTip", "Copy")

    # ---- Soft tooltip helpers for copy button ----
    def _show_copy_tip(self):
        try:
            text = self.copy_btn.property("softTip") or "Copy"
            self._soft_tip.setText(text)
            self._soft_tip.showNear(self.copy_btn, above=True, y_offset=10)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if obj is self.copy_btn:
            if event.type() == QEvent.Enter:
                self._tip_timer.start(600)
            elif event.type() in (QEvent.Leave, QEvent.MouseButtonPress):
                self._tip_timer.stop()
                self._soft_tip.hide()
        return super().eventFilter(obj, event)


class AutoResizingPlainTextEdit(QPlainTextEdit):
    """A read-only QPlainTextEdit that grows with its content and never shows
    inner scrollbars. The outer QScrollArea handles scrolling.
    """
    def __init__(self, min_height: int = 160, parent: QWidget | None = None):
        super().__init__(parent)
        self._min_height = min_height
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Enable wrapping to widget width
        try:
            self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        except Exception:
            pass
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Recompute height when content changes
        self.document().contentsChanged.connect(self.update_height)
        try:
            self.document().documentLayout().documentSizeChanged.connect(lambda _=None: self.update_height())
        except Exception:
            pass
        # Ensure first layout pass updates height after widget is shown
        QTimer.singleShot(0, self.update_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Ensure layout width tracks viewport width for correct doc height
        self.document().setTextWidth(self.viewport().width())
        self.update_height()

    def update_height(self):
        # Calculate document height and set widget height accordingly
        # Determine effective content width
        vw = self.viewport().width()
        if vw <= 0:
            vw = max(0, self.width() - self.frameWidth() * 2 - 4)
        self.document().setTextWidth(vw)
        # Compute document height
        layout = self.document().documentLayout()
        doc_size = layout.documentSize()
        doc_h = int(doc_size.height())
        frame = self.frameWidth() * 2
        pad = 12  # small breathing space
        target = max(self._min_height, doc_h + frame + pad)
        # Enforce exact height so inner scrolling never appears
        self.setMinimumHeight(target)
        self.setMaximumHeight(target)

    def wheelEvent(self, event):
        """Avoid inner scrolling; let the outer scroll area handle the wheel."""
        event.ignore()


class AutoResizingTextEdit(QTextEdit):
    """Read-only QTextEdit that wraps to widget width and auto-resizes
    to its content height. No inner scrollbars; outer scroll handles scrolling.
    """
    def __init__(self, min_height: int = 60, parent: QWidget | None = None):
        super().__init__(parent)
        self._min_height = min_height
        self.setReadOnly(True)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Signals to react on content/size changes
        self.textChanged.connect(self.update_height)
        try:
            self.document().documentLayout().documentSizeChanged.connect(lambda _=None: self.update_height())
        except Exception:
            pass
        QTimer.singleShot(0, self.update_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Match text width to viewport width for correct wrapping height
        self.document().setTextWidth(self.viewport().width())
        self.update_height()

    def update_height(self):
        vw = self.viewport().width()
        if vw <= 0:
            vw = max(0, self.width() - self.frameWidth() * 2 - 4)
        self.document().setTextWidth(vw)
        layout = self.document().documentLayout()
        doc_size = layout.documentSize()
        h = max(self._min_height, int(doc_size.height()) + self.frameWidth() * 2 + 2)
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)
        self.updateGeometry()

    def wheelEvent(self, event):
        # Avoid inner scrolling; let outer scroll area handle it
        event.ignore()

class AutoWrapLabel(QLabel):
    """A selectable, word-wrapping label that grows with its content and never
    shows inner scrollbars. Lets the outer scroll area handle scrolling."""
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Use zero contents margins; padding is controlled via QSS for consistency
        self.setContentsMargins(0, 0, 0, 0)

    def setText(self, text: str) -> None:
        super().setText(text or "")
        QTimer.singleShot(0, self._adjust_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._adjust_height)

    def _adjust_height(self):
        # Compute height based on the current available width for proper wrapping
        w = self.width()
        if w <= 0 and self.parent() is not None:
            w = max(0, self.parent().width() - 16)
        if w <= 0:
            w = self.sizeHint().width()
        # QLabel provides heightForWidth when wordWrap is enabled
        h_text = self.heightForWidth(w) if self.wordWrap() else self.sizeHint().height()
        h = max(60, h_text)
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)


class SoftToolTip(QWidget):
    """A lightweight, custom tooltip to avoid native borders/shadows."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        # Enable transparent corners so rounded shape is visible
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._radius = 6
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._label)
        # Background is painted manually for crisp rounded corners; only text color via QSS
        self.setStyleSheet("color: #1f2937; background: #ffffff;")

    def setText(self, text: str):
        self._label.setText(text or "")
        self.adjustSize()

    def showNear(self, widget: QWidget, above: bool = True, y_offset: int = 8):
        if not widget:
            return
        r = widget.rect()
        if above:
            pos = widget.mapToGlobal(QPoint(r.center().x() - self.width() // 2, r.top() - self.height() - y_offset))
        else:
            pos = widget.mapToGlobal(QPoint(r.center().x() - self.width() // 2, r.bottom() + y_offset))
        self.move(pos)
        self.show()
        self.updateGeometry()

    def paintEvent(self, event):
        # Paint a rounded white background with transparent outside
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            rect = self.rect().adjusted(0, 0, -1, -1)
            path = QPainterPath()
            path.addRoundedRect(rect, self._radius, self._radius)
            # Fill
            p.fillPath(path, QBrush(QColor("#ffffff")))
            # Light gray border
            pen = QPen(QColor("#e6e6e6"))
            pen.setWidthF(1.5)
            p.setPen(pen)
            p.drawPath(path)
        finally:
            p.end()


class Spinner(QWidget):
    """Tiny, smooth progress spinner (indeterminate)."""
    def __init__(self, parent=None, diameter: int = 14, line_width: int = 2, color: QColor | str = "#ffffff"):
        super().__init__(parent)
        self._diam = int(diameter)
        self._lw = int(line_width)
        self._color = QColor(color)
        self._angle = 0
        self.setFixedSize(self._diam, self._diam)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(16)  # ~60 FPS for smoothness

    def start(self):
        if not self._timer.isActive():
            self._timer.start()
        self.show()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 6) % 360  # 60fps * 6deg = full rotation in 1s
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.translate(self.width() / 2, self.height() / 2)
            p.rotate(self._angle)
            pen = QPen(self._color)
            pen.setWidth(self._lw)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            r = (min(self.width(), self.height()) - self._lw) / 2.0
            # Draw 280-degree arc to create a gap
            start_angle = 0
            span_angle = 280
            # Convert to radians along circle via incremental lines for simplicity
            path = QPainterPath()
            from math import cos, sin, radians
            steps = 40
            for i in range(steps + 1):
                a = radians(start_angle + span_angle * (i / steps))
                x = r * cos(a)
                y = r * sin(a)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.drawPath(path)
        finally:
            p.end()


class Toast(QWidget):
    """Small, auto-dismissing toast shown in the top-right corner."""
    closed = Signal(object)

    def __init__(self, text: str, kind: str = "info", duration_ms: int = 3000, parent: QWidget | None = None):
        # Child widget inside our toast area; no top-level tooltip window
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._radius = 10
        self._duration = max(1200, int(duration_ms))
        # Colors
        self._bg = QColor("#ffffff")
        self._fg = QColor("#1f2937")
        self._border = QColor("#e6e6e6")
        # Accent + lightly tinted backgrounds
        if kind in ("info", "warning"):
            self._accent = QColor("#f59e0b")  # orange
            self._bg = QColor("#FFF7ED")     # light orange tint
        elif kind == "success":
            self._accent = QColor("#22c55e")  # green
            self._bg = QColor("#ECFDF5")      # light green tint
        elif kind == "error":
            self._accent = QColor("#ef4444")  # red
            self._bg = QColor("#FEF2F2")      # light red tint
        else:
            self._accent = QColor("#9ca3af")  # gray fallback

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(8)
        self._label = QLabel(text)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("color: #1f2937;")
        lay.addWidget(self._label)

        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._fx.setOpacity(0.0)
        self._fade = QPropertyAnimation(self._fx, b"opacity", self)
        self._fade.setDuration(180)

        # Auto dismiss timers
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.timeout.connect(self._fade_out)

    def show_with_fade(self):
        self.adjustSize()
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self.show()
        self._hold.start(self._duration)

    def _fade_out(self):
        self._fade.stop()
        self._fade.setStartValue(self._fx.opacity())
        self._fade.setEndValue(0.0)
        self._fade.setDuration(200)
        def _done():
            self.closed.emit(self)
            self.hide()
            self.deleteLater()
        self._fade.finished.connect(_done)
        self._fade.start()

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            rect = self.rect().adjusted(0, 0, -1, -1)
            path = QPainterPath()
            path.addRoundedRect(rect, self._radius, self._radius)
            p.fillPath(path, QBrush(self._bg))
            pen = QPen(self._border)
            pen.setWidthF(1.0)
            p.setPen(pen)
            p.drawPath(path)
        finally:
            p.end()

class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_ollama: str = "qwen3:8b", current_whisper: str = "medium"):
        super().__init__(parent)
        self.setObjectName("settingsDialog")
        self.setWindowTitle("Settings")
        self.setMinimumSize(520, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        # Section: Prerequisites
        header_font = QFont()
        header_font.setWeight(QFont.DemiBold)

        prereq_header = QLabel("Prerequisites")
        prereq_header.setFont(header_font)
        layout.addWidget(prereq_header)

        prereq_text = QLabel(
            "Ollama is required. Install from <a href=\"https://ollama.com\">ollama.com</a>.\n"
            "You also need to install models locally (e.g., qwen3 family).\n"
            "Whisper will be downloaded automatically on first use; the initial run may take a bit longer."
        )
        prereq_text.setOpenExternalLinks(True)
        prereq_text.setWordWrap(True)
        prereq_text.setStyleSheet("color: #6b7280;")
        layout.addWidget(prereq_text)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setStyleSheet("color: #e6e6e6;")
        layout.addWidget(line1)

        # Ollama model selection
        ollama_header = QLabel("Ollama Model")
        ollama_header.setFont(header_font)
        layout.addWidget(ollama_header)
        hint1 = QLabel("Recommended: qwen3 family. More parameters → better quality, but slower.")
        hint1.setStyleSheet("color: #6b7280;")
        hint1.setWordWrap(True)
        layout.addWidget(hint1)

        self.cmb_ollama = QComboBox()
        self.cmb_ollama.setEditable(False)
        self._populate_ollama_models(current_ollama)
        layout.addWidget(self.cmb_ollama)

        layout.addSpacing(8)

        # Whisper model selection
        whisper_header = QLabel("Whisper")
        whisper_header.setFont(header_font)
        layout.addWidget(whisper_header)
        hint2 = QLabel("Medium has many more parameters. Transcriptions are much better, but STT can take longer.")
        hint2.setStyleSheet("color: #6b7280;")
        hint2.setWordWrap(True)
        layout.addWidget(hint2)

        self.cmb_whisper = QComboBox()
        self.cmb_whisper.addItems(["base", "small", "medium"]) 
        # set default/current
        idx = max(0, self.cmb_whisper.findText(current_whisper))
        self.cmb_whisper.setCurrentIndex(idx if idx >= 0 else 2)
        layout.addWidget(self.cmb_whisper)

        layout.addStretch(1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Local, minimal styling to match app
        self.setStyleSheet(
            """
            #settingsDialog { background: #ffffff; }
            QLabel { color: #1f2937; }
            QComboBox { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px; padding: 6px 10px; color: #1f2937; }
            QComboBox QAbstractItemView { background: #fff; color: #1f2937; border: 1px solid #e6e6e6; selection-background-color: #ececec; }
            QDialogButtonBox QPushButton { padding: 8px 14px; border-radius: 10px; background: #f0f0f0; color: #1f2937; border: 1px solid #e6e6e6; }
            QDialogButtonBox QPushButton:hover { background: #e7e7e7; }
            QDialogButtonBox QPushButton:default { background: #1f2937; color: #ffffff; border: none; }
            QFrame { background: transparent; }
            """
        )

    def _populate_ollama_models(self, current: str):
        models = []
        # Try to call `ollama list` for real models
        try:
            proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
            out = proc.stdout or ""
            for line in out.splitlines():
                # typical format: name tag size modified
                parts = line.strip().split()
                if parts:
                    name = parts[0]
                    if name != "NAME":
                        models.append(name)
        except Exception:
            pass
        # Fallback curated list
        if not models:
            models = [
                "qwen3:8b", "qwen3:14b", "qwen2.5:7b", "qwen2.5:14b",
                "llama3.1:8b", "llama3.1:70b", "phi3:3.8b", "mistral:7b"
            ]
        # Fill combobox
        self.cmb_ollama.clear()
        self.cmb_ollama.addItems(models)
        # Select current/default
        idx = self.cmb_ollama.findText(current)
        if idx < 0:
            idx = max(0, self.cmb_ollama.findText("qwen3:8b"))
        self.cmb_ollama.setCurrentIndex(idx if idx >= 0 else 0)


class CircleButton(QToolButton):
    """Small circular button painted manually to guarantee round traffic lights,
    supports active/inactive state and group-hover darkening.
    """
    hoverChanged = Signal(bool)

    def __init__(self, color_hex: str, diameter: int = 14, parent: QWidget | None = None):
        super().__init__(parent)
        self._base = QColor(color_hex)
        self._inactive = QColor("#cfcfcf")
        self._border = QColor("#dadada")
        self._diameter = int(diameter)
        self._active = True
        self._group_hover = False
        self.setFixedSize(self._diameter, self._diameter)
        # Neutralize native styling
        self.setStyleSheet("QToolButton{background: transparent; border: none; padding:0; margin:0;}")
        self.setCursor(Qt.ArrowCursor)

    def set_active(self, active: bool):
        if self._active != active:
            self._active = active
            self.update()

    def set_group_hover(self, hov: bool):
        if self._group_hover != hov:
            self._group_hover = hov
            self.update()

    def enterEvent(self, event):
        self.hoverChanged.emit(True)
        return super().enterEvent(event)

    def leaveEvent(self, event):
        self.hoverChanged.emit(False)
        return super().leaveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        color = QColor(self._base if self._active else self._inactive)
        if self._group_hover:
            color = color.darker(110)  # ~10% darker
        p.setBrush(QBrush(color))
        pen = QPen(self._border)
        pen.setWidth(0.5)
        p.setPen(pen)
        p.drawEllipse(rect)
        p.end()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MailWhisper")
        self.resize(900, 700)
        # Minimum window size (adjust here)
        self.setMinimumSize(560, 520)
        
        # State
        self.state: DraftState = initial_state()
        self.recorder = ButtonControlledRecorder(samplerate=16000, channels=1)
        self._recording = False
        self._worker_thread: QThread | None = None
        # Intro dismissed flag must exist before any event filters run
        self._intro_dismissed = False

        # Top bar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(24, 10, 24, 10)
        top_bar.setSpacing(12)

        # macOS-style traffic lights (custom painted circles)
        self.btn_close = CircleButton("#ff5f57")
        self.btn_min   = CircleButton("#ffbd2e")
        self.btn_max   = CircleButton("#28c840")
        traffic_wrap = QWidget()
        traffic_layout = QHBoxLayout(traffic_wrap)
        traffic_layout.setContentsMargins(0, 0, 0, 0)
        traffic_layout.setSpacing(7)
        traffic_layout.addWidget(self.btn_close)
        traffic_layout.addWidget(self.btn_min)
        traffic_layout.addWidget(self.btn_max)
        traffic_layout.addSpacing(6)
        top_bar.addWidget(traffic_wrap, alignment=Qt.AlignLeft)
        # Group-hover: when any light is hovered, all darken slightly
        self._traffic = [self.btn_close, self.btn_min, self.btn_max]
        for b in self._traffic:
            b.hoverChanged.connect(lambda s, self=self: self._set_traffic_hover(s))

        self.menu_btn = QToolButton()
        self.menu_btn.setObjectName("menuBtn")
        self.menu_btn.setFixedSize(40, 40)
        self.menu_btn.setIcon(QIcon("ui/icons/menu.svg"))
        self.menu_btn.setIconSize(QSize(22, 22))
        self.menu_btn.setText("")
        top_bar.addWidget(self.menu_btn, alignment=Qt.AlignLeft)
        top_bar.addSpacing(16)

        self.logo_lbl = QLabel("🎙️✉️ MailWhisper")
        font = QFont()
        font.setPointSize(16)
        font.setWeight(QFont.DemiBold)
        self.logo_lbl.setFont(font)
        top_bar.addWidget(self.logo_lbl, alignment=Qt.AlignLeft)

        top_bar.addStretch(1)

        self.settings_btn = QToolButton()
        self.settings_btn.setObjectName("settingsBtn")
        self.settings_btn.setFixedSize(40, 40)
        self.settings_btn.setIcon(QIcon("ui/icons/settings.svg"))
        self.settings_btn.setIconSize(QSize(22, 22))
        self.settings_btn.setText("")
        self.settings_btn.clicked.connect(self.show_settings)
        top_bar.addWidget(self.settings_btn, alignment=Qt.AlignRight)

        top_bar_widget = QWidget()
        top_bar_widget.setObjectName("topBar")
        top_bar_widget.setLayout(top_bar)
        top_bar_widget.setMinimumHeight(60)

        # Center: scrollable area with cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.center_layout = QVBoxLayout(content)
        self.center_layout.setContentsMargins(32, 24, 32, 24)
        self.center_layout.setSpacing(10)

        # Cards for fields
        self.card_to = FieldCard("To")
        self.card_cc = FieldCard("Cc")
        self.card_subject = FieldCard("Subject")
        self.card_tone = FieldCard("Tone")
        self.card_body = FieldCard("Body", multiline=True)

        # copy actions handled inside each FieldCard (icon feedback + clipboard)

        # Assemble cards
        self.center_layout.addWidget(self.card_to)
        self.center_layout.addWidget(self.card_cc)
        self.center_layout.addWidget(self.card_subject)
        self.center_layout.addWidget(self.card_tone)
        self.center_layout.addWidget(self.card_body)
        self.center_layout.addStretch(1)
        self.scroll_area.setWidget(content)

        # Bottom bar: mic + save
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 12, 16, 12)
        bottom.setSpacing(12)

        # New draft button (bottom-left)
        self.new_btn = QToolButton()
        self.new_btn.setObjectName("newBtn")
        # Use custom tooltip instead of native to avoid dark border
        self.new_btn.setToolTip("")
        self.new_btn.setFixedSize(40, 40)
        self.new_btn.setIcon(QIcon("ui/icons/badge-plus.svg"))
        self.new_btn.setIconSize(QSize(20, 20))
        self.new_btn.clicked.connect(self.on_new_draft_clicked)
        self.new_btn.installEventFilter(self)

        self.mic_btn = QPushButton("🎤  Record")
        self.mic_btn.setObjectName("micBtn")
        self.mic_btn.clicked.connect(self.toggle_recording)
        self.mic_btn.setFixedWidth(140)  # keep width stable across text changes
        self._mic_btn_base_width = 140
        # Processing overlay container (created on demand)
        self._proc_container = None
        self._mic_spinner = None
        self.mic_btn.installEventFilter(self)

        self.save_btn = QPushButton("💾  Save")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self.on_save_clicked)
        self.save_btn.setFixedWidth(100)

        bottom.addWidget(self.new_btn)
        bottom.addStretch(1)
        bottom.addWidget(self.mic_btn)
        bottom.addWidget(self.save_btn)

        bottom_widget = QFrame()
        bottom_widget.setObjectName("bottomBar")
        bottom_widget.setLayout(bottom)

        # Main layout
        central = QWidget()
        central.setObjectName("rootContainer")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top_bar_widget)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(bottom_widget, 0)
        self.setCentralWidget(central)

        # Frameless window look and interactions
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.btn_close.clicked.connect(self.close)
        self.btn_min.clicked.connect(self.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max_restore)

        self.apply_styles()
        self.refresh_view()

        # Keyboard shortcut: Space toggles recording (start/stop)
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.setContext(Qt.ApplicationShortcut)
        self._space_shortcut.setAutoRepeat(False)
        self._space_shortcut.activated.connect(self._on_space_pressed)

        # Intro overlay (subtle rotating messages until first interaction)
        self._intro_messages = [
            "Welcome to MailWhisper",
            "Use SPACE to start recording",
            "Speak your email idea — we’ll draft it",
            "Try a tone: friendly, formal, brief",
            "You can refine later — start talking",
        ]
        self._setup_intro_overlay()
        # Global event filter: observe button clicks; do not dismiss on outside focus changes
        QApplication.instance().installEventFilter(self)
        # Also watch the scroll viewport for resize so overlay stays perfectly centered
        self.scroll_area.viewport().installEventFilter(self)
        # Custom tooltip instance
        self._soft_tip = SoftToolTip(self)

        # Toast area for transient messages (top-right)
        self._toast_area = QWidget(self)
        self._toast_area.setAttribute(Qt.WA_TranslucentBackground, True)
        self._toast_area.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._toast_layout = QVBoxLayout(self._toast_area)
        self._toast_layout.setContentsMargins(0, 0, 0, 0)
        self._toast_layout.setSpacing(8)
        self._toasts: list[Toast] = []
        self._position_toast_area()

    # ------- Styling -------
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: transparent; }
            QWidget#rootContainer { background: #f0f0f0; border-radius: 10px; }

            #topBar {
                background: #ffffff;
                border-bottom: 1px solid #e6e6e6;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            /* Traffic light buttons drawn via CircleButton painter; no extra styling here */
            QToolButton#menuBtn, QToolButton#settingsBtn { border: none; background: transparent; color: #6b7280; }
            QToolButton#menuBtn:hover, QToolButton#settingsBtn:hover { background: #ececec; border-radius: 4px; color: #1f2937; }
            QLabel { color: #1f2937; }
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > * { background: transparent; }
            #bottomBar {
                border-top: 1px solid #eee;
                background: #fff;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QLabel#cardTitle {
                color: #1f2937;
                font-weight: 600;
                padding-top: 10px;
            }
            QWidget#card { background: #ffffff; border: 1px solid #e6e6e6; border-radius: 10px; }
            QLineEdit, QPlainTextEdit, QTextEdit { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px; padding: 8px 10px; font-size: 14px; color: #1f2937; }
            /* Body text styled like the input fields */
            QLabel#bodyText {
                background: #fff;
                border: 1px solid #e6e6e6;
                border-radius: 8px;
                padding: 8px 10px; /* match QLineEdit padding (top/bottom 8, left/right 10) */
                font-size: 14px;
                color: #1f2937;
            }

            /* Minimal, modern scrollbars */
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #e0e0e0; /* match copy hover bg */
                min-height: 24px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical:hover {
                background: #d9d9d9;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; width: 0px; background: transparent; border: none;
            }
            QScrollBar:horizontal { height: 10px; background: transparent; }
            QScrollBar::handle:horizontal { background: #ececec; min-width: 24px; border-radius: 8px; }
            QScrollBar::handle:horizontal:hover { background: #e0e0e0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
            QPushButton#copyBtn { padding: 4px; min-width: 32px; min-height: 32px; border-radius: 4px; border: none; background: transparent; color: #6b7280; }
            QPushButton#copyBtn:hover { background: #d9d9d9; color: #1f2937; }
            QPushButton#copyBtn:pressed { background: #cfcfcf; }
            /* Global tooltip styling (clean, no harsh border) */
            QToolTip { background: #ffffff; color: #1f2937; border: none; border-radius: 10px; padding: 0px 0px; }
            QPushButton#micBtn {
                padding: 10px 16px;
                border-radius: 12px;
                background: #1f2937;
                color: white;
                border: none;
            }
            QPushButton#micBtn:hover { background: #2b3647; }
            QPushButton#micBtn[recording="true"] {
                background: #e53935;
            }
            QPushButton#micBtn[recording="true"]:hover {
                background: #e53935;
            }
            /* Processing state: orange with pulse animation handled in code */
            QPushButton#micBtn[processing="true"],
            QPushButton#micBtn:disabled[processing="true"] {
                background: #f59e0b; /* orange */
                color: white;
            }
            QPushButton#saveBtn {
                padding: 10px 16px;
                border-radius: 12px;
                background: #f0f0f0;
                color: #1f2937;
                border: 1px solid #e6e6e6;
            }
            QPushButton#saveBtn:hover { background: #e7e7e7; }
            QToolButton#newBtn { border: none; background: transparent; color: #6b7280; padding: 6px; border-radius: 8px; }
            QToolButton#newBtn:hover { background: #ececec; color: #1f2937; }
            QLabel#introOverlay {
                background: transparent;
                color: #c8c8c8; /* subtle vs #f0f0f0 background */
                font-size: 22px;
            }
        """)

    # ---- Frameless window helpers ----
    def _toggle_max_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _set_traffic_hover(self, hovered: bool):
        for b in getattr(self, '_traffic', []):
            b.set_group_hover(hovered)

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:
            active = self.isActiveWindow()
            for b in getattr(self, '_traffic', []):
                b.set_active(active)
        super().changeEvent(event)

    def mousePressEvent(self, event):
        try:
            posf = event.position()
            pos = posf.toPoint()
        except Exception:
            pos = event.pos()
        if event.button() == Qt.LeftButton and pos.y() <= 72:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft() if hasattr(event, 'globalPosition') else event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, '_drag_offset', None) is not None and (event.buttons() & Qt.LeftButton):
            global_p = event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else event.globalPos()
            self.move(global_p - self._drag_offset)
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    # ------- UI Actions -------
    def show_settings(self):
        # Current selections come from module settings/env
        try:
            curr_ollama = getattr(nodes_mod, 'GLOBAL_LLM_MODEL', 'qwen3:8b')
        except Exception:
            curr_ollama = 'qwen3:8b'
        curr_whisper = os.getenv('WHISPER_LOCAL_MODEL', 'medium')
        dlg = SettingsDialog(self, current_ollama=curr_ollama, current_whisper=curr_whisper)
        if dlg.exec() == QDialog.Accepted:
            selected_ollama = dlg.cmb_ollama.currentText().strip()
            selected_whisper = dlg.cmb_whisper.currentText().strip()
            # Apply runtime settings
            try:
                nodes_mod.set_llm_model(selected_ollama)
            except Exception:
                pass
            try:
                set_whisper_model(selected_whisper)
            except Exception:
                pass

    def _on_space_pressed(self):
        # Respect disabled state while processing
        if not self.mic_btn.isEnabled():
            return
        self.dismiss_intro()
        self.toggle_recording()

    # ----- Intro overlay helpers -----
    def _setup_intro_overlay(self):
        if self._intro_dismissed:
            return
        # Overlay label is parented to the scroll area's viewport to stay centered
        vp = self.scroll_area.viewport()
        self._intro_label = QLabel(vp)
        self._intro_label.setObjectName("introOverlay")
        self._intro_label.setWordWrap(True)
        self._intro_label.setAlignment(Qt.AlignCenter)
        self._intro_label.setText("")
        self._intro_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._sync_intro_geometry()
        # Typing animation state and timers
        self._intro_index = 0
        self._typed_pos = 0
        self._typing_timer = QTimer(self)
        self._typing_timer.timeout.connect(self._advance_typing)
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._next_message)
        self._intro_label.show()
        # Kick off first typing cycle
        QTimer.singleShot(0, self._start_typing_cycle)

    def _start_typing_cycle(self):
        if self._intro_dismissed:
            return
        self._current_text = self._intro_messages[self._intro_index]
        self._typed_pos = 0
        self._intro_label.setText("")
        # Compute per-char interval to finish around 0.5s, with a lower bound for readability
        total_ms = 500
        n = max(1, len(self._current_text))
        per_char = max(18, int(total_ms / n))
        self._typing_timer.start(per_char)

    def _advance_typing(self):
        if self._intro_dismissed:
            self._typing_timer.stop()
            return
        self._typed_pos += 1
        self._intro_label.setText(self._current_text[: self._typed_pos])
        if self._typed_pos >= len(self._current_text):
            # Done typing: hold for 3s, then next
            self._typing_timer.stop()
            self._hold_timer.start(3000)

    def _next_message(self):
        if self._intro_dismissed:
            return
        self._intro_index = (self._intro_index + 1) % len(self._intro_messages)
        self._start_typing_cycle()

    def dismiss_intro(self):
        if self._intro_dismissed:
            return
        self._intro_dismissed = True
        try:
            self._typing_timer.stop()
            self._hold_timer.stop()
        except Exception:
            pass
        if hasattr(self, "_intro_label") and self._intro_label:
            self._intro_label.hide()

    def resizeEvent(self, event):
        # Keep the overlay sized to the center area
        super().resizeEvent(event)
        self._sync_intro_geometry()
        self._position_toast_area()

    def _sync_intro_geometry(self):
        if hasattr(self, "_intro_label") and self._intro_label and not self._intro_dismissed:
            vp = self.scroll_area.viewport()
            self._intro_label.setGeometry(vp.rect())

    def _position_toast_area(self):
        try:
            right = 20
            top = 76  # below top bar
            width = 360
            x = max(0, self.width() - right - width)
            y = top
            self._toast_area.setGeometry(x, y, width, max(60, self.height() - y - 20))
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # Dismiss only when the Record button is clicked (not other buttons)
        if (not getattr(self, '_intro_dismissed', True)
            and event.type() == QEvent.MouseButtonPress
            and obj is getattr(self, 'mic_btn', None)):
            self.dismiss_intro()
        # Keep overlay centered when the scroll viewport resizes
        if obj is self.scroll_area.viewport() and event.type() == QEvent.Resize:
            self._sync_intro_geometry()
        # Custom tooltip handling for New Draft button
        if obj is getattr(self, 'new_btn', None):
            if event.type() == QEvent.Enter:
                # Show after a short delay to mimic native behavior
                if not hasattr(self, '_new_tip_timer'):
                    self._new_tip_timer = QTimer(self)
                    self._new_tip_timer.setSingleShot(True)
                    self._new_tip_timer.timeout.connect(lambda: self._show_soft_tip("New Draft"))
                self._new_tip_timer.start(600)
            elif event.type() in (QEvent.Leave, QEvent.MouseButtonPress):
                if hasattr(self, '_new_tip_timer'):
                    self._new_tip_timer.stop()
                self._soft_tip.hide()
        # Button resize: layout will keep overlay centered; ensure size hint is updated
        if obj is getattr(self, 'mic_btn', None) and event.type() == QEvent.Resize:
            self._center_proc_container()
        return super().eventFilter(obj, event)

    def _show_soft_tip(self, text: str):
        if not self.isActiveWindow():
            return
        try:
            self._soft_tip.setText(text)
            self._soft_tip.showNear(self.new_btn, above=True, y_offset=10)
        except Exception:
            pass

    # ---- Toast helpers ----
    def show_toast(self, text: str, kind: str = "info", duration_ms: int = 3000):
        try:
            # Only one toast at a time: clear any existing
            for t in list(self._toasts):
                try:
                    t.hide()
                    t.deleteLater()
                except Exception:
                    pass
            self._toasts.clear()
            # Remove any leftover layout items
            try:
                while self._toast_layout.count():
                    item = self._toast_layout.takeAt(0)
                    w = item.widget()
                    if w is not None:
                        w.deleteLater()
            except Exception:
                pass

            t = Toast(text, kind=kind, duration_ms=duration_ms, parent=self._toast_area)
            t.closed.connect(self._remove_toast)
            self._toast_layout.addWidget(t, 0, Qt.AlignRight | Qt.AlignTop)
            self._toasts.append(t)
            t.show_with_fade()
        except Exception:
            pass

    def _remove_toast(self, toast: Toast):
        try:
            if toast in self._toasts:
                self._toasts.remove(toast)
            # Also remove from layout to avoid stacking gaps
            for i in reversed(range(self._toast_layout.count())):
                item = self._toast_layout.itemAt(i)
                if item and item.widget() is toast:
                    self._toast_layout.takeAt(i)
                    break
        except Exception:
            pass

    def _ensure_proc_container(self):
        if self._proc_container is not None:
            return
        try:
            # Container with spinner + label
            cont = QWidget(self.mic_btn)
            lay = QHBoxLayout(cont)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            sp = Spinner(cont, diameter=16, line_width=2, color="#ffffff")
            lbl = QLabel("Processing...", cont)
            lbl.setStyleSheet("color: #ffffff;")
            lay.addWidget(sp)
            lay.addWidget(lbl)
            cont.adjustSize()

            # Put the container centered inside the button using a layout on the button itself
            if not hasattr(self, "_mic_btn_layout") or self._mic_btn_layout is None:
                self._mic_btn_layout = QHBoxLayout(self.mic_btn)
                self._mic_btn_layout.setContentsMargins(0, 0, 0, 0)
                self._mic_btn_layout.setSpacing(0)
                self._mic_btn_layout.setAlignment(Qt.AlignCenter)
            self._mic_btn_layout.addWidget(cont, 0, Qt.AlignCenter)

            self._proc_container = cont
            self._mic_spinner = sp
        except Exception:
            self._proc_container = None
            self._mic_spinner = None

    def _center_proc_container(self):
        # Layout on the button keeps it centered; ensure preferred size is current
        try:
            if self._proc_container:
                self._proc_container.adjustSize()
        except Exception:
            pass

    def toggle_recording(self):
        if not self._recording:
            # Start
            self._recording = True
            self.mic_btn.setText("⏹  Stop")
            self.mic_btn.setProperty("recording", True)
            self.mic_btn.style().unpolish(self.mic_btn)
            self.mic_btn.style().polish(self.mic_btn)
            # Avoid resetting while recording
            if hasattr(self, 'new_btn'):
                self.new_btn.setEnabled(False)
            try:
                self.recorder.start()
            except Exception as e:
                self._recording = False
                self.mic_btn.setText("🎤  Record")
                self.mic_btn.setProperty("recording", False)
                self.mic_btn.style().unpolish(self.mic_btn)
                self.mic_btn.style().polish(self.mic_btn)
                print(f"Audio error: {e}", file=sys.stderr)
                if hasattr(self, 'new_btn'):
                    self.new_btn.setEnabled(True)
                self.show_toast("Audio error — see console for details", kind="error")
        else:
            # Stop and process
            self._recording = False
            self.mic_btn.setText("🎤  Record")
            self.mic_btn.setProperty("recording", False)
            self.mic_btn.style().unpolish(self.mic_btn)
            self.mic_btn.style().polish(self.mic_btn)

            try:
                audio = self.recorder.stop()
            except Exception as e:
                print(f"Stop error: {e}", file=sys.stderr)
                self.show_toast("Audio stop error", kind="error")
                return

            # Worker thread for transcription + LLM
            self.run_processing_worker(audio)

    def run_processing_worker(self, audio: np.ndarray):
        # Button enters processing state and becomes disabled
        self._set_mic_processing(True)
        self._worker_thread = QThread(self)
        # Keep a strong reference so Python GC doesn't collect the worker
        self._worker = ProcessAudioWorker(audio, 16000, self.state)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_patch_ready)
        self._worker.failed.connect(self.on_processing_failed)
        # Cleanup
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._on_thread_finished)
        # Prevent starting a fresh draft during processing
        if hasattr(self, 'new_btn'):
            self.new_btn.setEnabled(False)
        self._worker_thread.start()

    def on_patch_ready(self, patch: Dict[str, Any]):
        # Update state
        self.state.update(patch or {})
        self.refresh_view()
        self._set_mic_processing(False)
        if hasattr(self, 'new_btn'):
            self.new_btn.setEnabled(True)
        # Notify if nothing came back (e.g., cancelled)
        if not patch:
            self.show_toast("Cancelled", kind="info", duration_ms=1800)

    def on_processing_failed(self, msg: str):
        print(f"Processing failed: {msg}", file=sys.stderr)
        self._set_mic_processing(False)
        if hasattr(self, 'new_btn'):
            self.new_btn.setEnabled(True)
        self.show_toast(f"Processing failed: {msg}", kind="error")

    def _on_thread_finished(self):
        # Drop references so GC can clean up
        try:
            self._worker_thread.deleteLater()
        except Exception:
            pass
        self._worker_thread = None
        self._worker = None

    def _set_mic_processing(self, on: bool):
        if on:
            self.mic_btn.setEnabled(False)
            # Text moved into overlay for perfect centering
            self.mic_btn.setText("")
            self.mic_btn.setProperty("processing", True)
            self.mic_btn.style().unpolish(self.mic_btn)
            self.mic_btn.style().polish(self.mic_btn)
            self._ensure_proc_container()
            self._center_proc_container()
            if self._mic_spinner:
                self._mic_spinner.start()
            if self._proc_container:
                self._proc_container.show()
                # Ensure button is wide enough for spinner + label
                try:
                    self._proc_container.adjustSize()
                    required = self._proc_container.width() + 20  # small side padding
                    if required > self.mic_btn.width():
                        self.mic_btn.setFixedWidth(required)
                        self._center_proc_container()
                except Exception:
                    pass
        else:
            if self._mic_spinner:
                self._mic_spinner.stop()
            if self._proc_container:
                self._proc_container.hide()
            self.mic_btn.setEnabled(True)
            self.mic_btn.setText("🎤  Record")
            self.mic_btn.setProperty("processing", False)
            self.mic_btn.style().unpolish(self.mic_btn)
            self.mic_btn.style().polish(self.mic_btn)
            # Restore baseline width to keep layout consistent
            try:
                if getattr(self, '_mic_btn_base_width', None):
                    self.mic_btn.setFixedWidth(self._mic_btn_base_width)
            except Exception:
                pass

    def on_save_clicked(self):
        # Placeholder: implement later (export, sending, etc.)
        self.state["done"] = True
        print("Save clicked — state marked as done.")
        self.show_toast("Saved", kind="success")

    def on_new_draft_clicked(self):
        """Reset the UI to a fresh DraftState and restore recording UI."""
        # If processing is running, cancel and perform reset once finished
        if getattr(self, "_worker_thread", None) is not None and self._worker_thread.isRunning():
            try:
                if getattr(self, "_worker", None) is not None:
                    # Ask worker to cancel further work
                    try:
                        self._worker.cancel()
                    except Exception:
                        pass
            except Exception:
                pass
            # Disable actions while waiting
            self.mic_btn.setEnabled(False)
            if hasattr(self, 'new_btn'):
                self.new_btn.setEnabled(False)
            # Once the worker thread finishes, complete reset
            def _after():
                self._reset_to_new_draft()
            self._worker_thread.finished.connect(_after)
            return

        # If currently recording, stop stream and discard frames
        if self._recording:
            try:
                self.recorder.stop()
            except Exception:
                pass
            self._recording = False

        self._reset_to_new_draft()
        # Show intro again for a fresh draft
        self.restart_intro()
        self.show_toast("New draft", kind="info", duration_ms=1500)

    def _reset_to_new_draft(self):
        # Ensure we are not in processing/recording visuals
        self._set_mic_processing(False)
        self.mic_btn.setProperty("recording", False)
        self.mic_btn.style().unpolish(self.mic_btn)
        self.mic_btn.style().polish(self.mic_btn)
        if hasattr(self, 'new_btn'):
            self.new_btn.setEnabled(True)
        # Reset application state and UI
        self.state = initial_state()
        self.refresh_view()

    # ---- Intro control ----
    def restart_intro(self):
        """Re-enable the intro overlay and restart the typing cycle."""
        try:
            self._intro_dismissed = False
            if hasattr(self, '_intro_label') and self._intro_label is not None:
                self._intro_label.show()
                # Reset typing cycle
                if hasattr(self, '_typing_timer') and self._typing_timer is not None:
                    self._typing_timer.stop()
                if hasattr(self, '_hold_timer') and self._hold_timer is not None:
                    self._hold_timer.stop()
                self._intro_index = 0
                self._typed_pos = 0
                self._start_typing_cycle()
            else:
                # Not initialized yet; set up fresh
                self._setup_intro_overlay()
        except Exception:
            pass

    # ---- Graceful shutdown on app close ----
    def closeEvent(self, event):
        try:
            # Cancel running worker if any
            if getattr(self, "_worker_thread", None) is not None and self._worker_thread.isRunning():
                try:
                    if getattr(self, "_worker", None) is not None:
                        try:
                            self._worker.cancel()
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    self._worker_thread.quit()
                except Exception:
                    pass
                # Wait briefly for clean exit
                try:
                    self._worker_thread.wait(3000)
                except Exception:
                    pass
                # Fallback: forcefully terminate if still alive (last resort)
                try:
                    if self._worker_thread.isRunning():
                        self._worker_thread.terminate()
                        self._worker_thread.wait(1000)
                except Exception:
                    pass
        finally:
            # Ensure audio resources are released
            try:
                self.recorder.shutdown()
            except Exception:
                pass
            # Stop app-level timers to avoid late events on shutdown
            for name in ("_typing_timer", "_hold_timer", "_new_tip_timer"):
                try:
                    t = getattr(self, name, None)
                    if t:
                        t.stop()
                except Exception:
                    pass
        super().closeEvent(event)

    # ------- View Binding -------
    def refresh_view(self):
        # to
        tos = self.state.get("to", []) or []
        to_text = ", ".join(tos)
        self.card_to.set_text(to_text)
        self.card_to.setVisible(bool(to_text))

        # cc
        ccs = self.state.get("cc", []) or []
        cc_text = ", ".join(ccs)
        self.card_cc.set_text(cc_text)
        self.card_cc.setVisible(bool(cc_text))

        # subject
        subject = self.state.get("subject", "") or ""
        self.card_subject.set_text(subject)
        self.card_subject.setVisible(bool(subject.strip()))

        # tone
        tone = (self.state.get("tone") or "").strip()
        # Minimalist: hide tone when neutral
        if tone and tone != "neutral":
            self.card_tone.set_text(tone)
            self.card_tone.setVisible(True)
        else:
            self.card_tone.set_text(tone or "neutral")
            self.card_tone.setVisible(False)

        # body
        body = self.state.get("body", "") or ""
        self.card_body.set_text(body)
        self.card_body.setVisible(bool(body.strip()))


def main():
    app = QApplication(sys.argv)
    # Force a light theme regardless of OS dark mode
    QApplication.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f0f0f0"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f0f0f0"))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.Text, QColor("#1f2937"))
    pal.setColor(QPalette.WindowText, QColor("#1f2937"))
    pal.setColor(QPalette.ButtonText, QColor("#1f2937"))
    # Ensure tooltips use the same clean palette (avoid dark outlines)
    try:
        pal.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
        pal.setColor(QPalette.ToolTipText, QColor("#1f2937"))
    except Exception:
        pass
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
