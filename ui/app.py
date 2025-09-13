import sys
from typing import Dict, Any
from PySide6.QtCore import Qt, QObject, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QFont, QAction, QPalette, QColor, QIcon, QTextOption
from PySide6.QtWidgets import (
QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QFrame, QToolButton,
QDialog, QDialogButtonBox, QSizePolicy
)
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
                self.failed.emit("Keine Transkription erkannt.")
                return

            # 1) transcript setzen
            self.state_snapshot["transcript"] = text

            # 2) intent ermitteln (LLM)
            patch_intent = intent_node(self.state_snapshot)
            self.state_snapshot.update(patch_intent)  # last_op, intent

            # 3) apply auf Draft
            patch_apply = apply_node(self.state_snapshot)

            # combine: wir geben intent+apply zurück (UI will v.a. apply Felder)
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
        # Einheitliche Label-Spaltenbreite für konsistente Editor-Breite
        LABEL_COL_WIDTH = 82
        self.title_lbl.setMinimumWidth(LABEL_COL_WIDTH)

        self.copy_btn = QPushButton()
        self.copy_btn.setObjectName("copyBtn")
        self.copy_btn.setToolTip("Copy")
        self.copy_btn.setIcon(QIcon("ui/icons/copy.svg"))
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
        self.setContentsMargins(8, 6, 8, 6)

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MailWhisper")
        self.resize(900, 700)
        
        # State
        self.state: DraftState = initial_state()
        self.recorder = ButtonControlledRecorder(samplerate=16000, channels=1)
        self._recording = False
        self._worker_thread: QThread | None = None

        # Top bar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(24, 10, 24, 10)
        top_bar.setSpacing(12)

        # macOS-style traffic lights (custom)
        self.btn_close = QToolButton()
        self.btn_close.setObjectName("trafficClose")
        self.btn_close.setFixedSize(14, 14)
        self.btn_min = QToolButton()
        self.btn_min.setObjectName("trafficMin")
        self.btn_min.setFixedSize(14, 14)
        self.btn_max = QToolButton()
        self.btn_max.setObjectName("trafficMax")
        self.btn_max.setFixedSize(14, 14)
        traffic_wrap = QWidget()
        traffic_layout = QHBoxLayout(traffic_wrap)
        traffic_layout.setContentsMargins(0, 0, 0, 0)
        traffic_layout.setSpacing(8)
        traffic_layout.addWidget(self.btn_close)
        traffic_layout.addWidget(self.btn_min)
        traffic_layout.addWidget(self.btn_max)
        traffic_layout.addSpacing(6)
        top_bar.addWidget(traffic_wrap, alignment=Qt.AlignLeft)

        self.menu_btn = QToolButton()
        self.menu_btn.setObjectName("menuBtn")
        self.menu_btn.setFixedSize(40, 40)
        self.menu_btn.setIcon(QIcon("ui/icons/menu.svg"))
        self.menu_btn.setIconSize(QSize(22, 22))
        self.menu_btn.setText("")
        top_bar.addWidget(self.menu_btn, alignment=Qt.AlignLeft)
        top_bar.addSpacing(16)

        self.logo_lbl = QLabel("⛓️✉️ MailWhisper")
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

        # copy actions
        for card in (self.card_to, self.card_cc, self.card_subject, self.card_tone, self.card_body):
            card.copy_btn.clicked.connect(lambda _, c=card: QApplication.clipboard().setText(c.text()))

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
            /* Traffic light buttons */
            QToolButton#trafficClose, QToolButton#trafficMin, QToolButton#trafficMax {
                border: 1px solid #dadada;
                border-radius: 7px;
                background: #e5e5e5;
            }
            QToolButton#trafficClose { background: #ff5f57; border-color: #e0443e; }
            QToolButton#trafficMin   { background: #ffbd2e; border-color: #e0a722; }
            QToolButton#trafficMax   { background: #28c840; border-color: #1f9e31; }
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
                padding: 12px 12px;
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
        """)

    # ---- Frameless window helpers ----
    def _toggle_max_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

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

            # Worker thread für Transkription + LLM
            self.run_processing_worker(audio)

    def run_processing_worker(self, audio: np.ndarray):
        # Button disabled während Verarbeitung
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
        # State aktualisieren
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
        # Platzhalter: später Implementierung (Export, Versand, etc.)
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
        # minimalist: tone ausblenden, wenn neutral
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
