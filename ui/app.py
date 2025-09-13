import sys
from typing import Dict, Any
from PySide6.QtCore import Qt, QObject, QThread, Signal, QSize, QTimer, QEvent, QPropertyAnimation
from PySide6.QtGui import QFont, QAction, QPalette, QColor, QIcon, QTextOption, QPainter, QPen, QBrush, QShortcut, QKeySequence
from PySide6.QtWidgets import (
QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QFrame, QToolButton,
QDialog, QDialogButtonBox, QSizePolicy, QAbstractButton
)
from PySide6.QtWidgets import QGraphicsOpacityEffect
import numpy as np
from agent.state import initial_state, DraftState
from agent.nodes import intent_node, apply_node
from utils.stt_whisper_mem import transcribe_array
from ui.recorder import ButtonControlledRecorder

class ProcessAudioWorker(QObject):
    finished = Signal(dict) # emits patch for state update
    failed = Signal(str)

    def __init__(self, audio: np.ndarray, samplerate: int, state: Dict[str, Any]):
        super().__init__()
        self.audio = audio
        self.samplerate = samplerate
        self.state_snapshot = dict(state)  # shallow copy

    def run(self):
        try:
            text, detected_lang = transcribe_array(self.audio, samplerate=self.samplerate)
            text = (text or "").strip()
            if not text:
                self.failed.emit("No Transcription recognized.")
                return

            # 1) set transcript
            self.state_snapshot["transcript"] = text

            # 2) determine intent (LLM)
            patch_intent = intent_node(self.state_snapshot)
            self.state_snapshot.update(patch_intent)  # last_op, intent

            # 3) apply to draft
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
        self.copy_btn.setToolTip("Copy")
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
            self.editor = AutoWrapLabel()
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

    def set_text(self, text: str):
        if isinstance(self.editor, QPlainTextEdit):
            self.editor.setPlainText(text or "")
            if isinstance(self.editor, AutoResizingPlainTextEdit):
                QTimer.singleShot(0, self.editor.update_height)
        elif isinstance(self.editor, AutoWrapLabel):
            self.editor.setText(text or "")
        else:
            self.editor.setText(text or "")

    def text(self) -> str:
        if isinstance(self.editor, QPlainTextEdit):
            return self.editor.toPlainText()
        return self.editor.text()

    def _on_copy_clicked(self):
        # Copy current text to clipboard
        QApplication.clipboard().setText(self.text())
        # Feedback: swap icon to check for 1.5s
        self.copy_btn.setIcon(self._check_icon)
        self.copy_btn.setToolTip("Copied!")
        QTimer.singleShot(1500, self._restore_copy_icon)

    def _restore_copy_icon(self):
        self.copy_btn.setIcon(self._copy_icon)
        self.copy_btn.setToolTip("Copy")


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
        sh = self.sizeHint()
        h = max(60, sh.height())
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Settings (coming soon)"))
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)


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

        self.mic_btn = QPushButton("🎤  Record")
        self.mic_btn.setObjectName("micBtn")
        self.mic_btn.clicked.connect(self.toggle_recording)
        self.mic_btn.setFixedWidth(140)  # keep width stable across text changes

        self.save_btn = QPushButton("💾  Save")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self.on_save_clicked)
        self.save_btn.setFixedWidth(100)

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
        self._intro_dismissed = False
        self._setup_intro_overlay()
        # Global event filter: observe button clicks; do not dismiss on outside focus changes
        QApplication.instance().installEventFilter(self)
        # Also watch the scroll viewport for resize so overlay stays perfectly centered
        self.scroll_area.viewport().installEventFilter(self)

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
            QLineEdit, QPlainTextEdit { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px; padding: 8px 10px; font-size: 14px; color: #1f2937; }
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
            QToolTip { background: #ffffff; color: #1f2937; border: 1px solid #e6e6e6; border-radius: 8px; padding: 6px 8px; }
            QPushButton#micBtn {
                padding: 10px 16px;
                border-radius: 12px;
                background: #1f2937;
                color: white;
                border: none;
            }
            QPushButton#micBtn[recording="true"] {
                background: #e53935;
            }
            QPushButton#saveBtn {
                padding: 10px 16px;
                border-radius: 12px;
                background: #f0f0f0;
                color: #1f2937;
                border: 1px solid #e6e6e6;
            }
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
        dlg = SettingsDialog(self)
        dlg.exec()

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

    def _sync_intro_geometry(self):
        if hasattr(self, "_intro_label") and self._intro_label and not self._intro_dismissed:
            vp = self.scroll_area.viewport()
            self._intro_label.setGeometry(vp.rect())

    def eventFilter(self, obj, event):
        # Dismiss only when a button within the app is clicked, not on focus changes
        if (not self._intro_dismissed and event.type() == QEvent.MouseButtonPress
            and isinstance(obj, QAbstractButton)):
            self.dismiss_intro()
        # Keep overlay centered when the scroll viewport resizes
        if obj is self.scroll_area.viewport() and event.type() == QEvent.Resize:
            self._sync_intro_geometry()
        return super().eventFilter(obj, event)

    def toggle_recording(self):
        if not self._recording:
            # Start
            self._recording = True
            self.mic_btn.setText("⏹  Stop")
            self.mic_btn.setProperty("recording", True)
            self.mic_btn.style().unpolish(self.mic_btn)
            self.mic_btn.style().polish(self.mic_btn)
            try:
                self.recorder.start()
            except Exception as e:
                self._recording = False
                self.mic_btn.setText("🎤  Record")
                self.mic_btn.setProperty("recording", False)
                self.mic_btn.style().unpolish(self.mic_btn)
                self.mic_btn.style().polish(self.mic_btn)
                print(f"Audio error: {e}", file=sys.stderr)
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
                return

            # Worker thread for transcription + LLM
            self.run_processing_worker(audio)

    def run_processing_worker(self, audio: np.ndarray):
        # Button disabled during processing
        self.mic_btn.setEnabled(False)
        self.mic_btn.setText("⏳ Processing...")
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
        self._worker_thread.start()

    def on_patch_ready(self, patch: Dict[str, Any]):
        # Update state
        self.state.update(patch or {})
        self.refresh_view()
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("🎤  Record")

    def on_processing_failed(self, msg: str):
        print(f"Processing failed: {msg}", file=sys.stderr)
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("🎤  Record")

    def _on_thread_finished(self):
        # Drop references so GC can clean up
        try:
            self._worker_thread.deleteLater()
        except Exception:
            pass
        self._worker_thread = None
        self._worker = None

    def on_save_clicked(self):
        # Placeholder: implement later (export, sending, etc.)
        self.state["done"] = True
        print("Save clicked — state marked as done.")

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
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
