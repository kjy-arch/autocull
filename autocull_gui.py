import sys
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QTabWidget,
    QScrollArea, QGridLayout, QFrame, QGroupBox, QSplitter, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QMimeData
from PyQt6.QtGui import QPixmap, QFont, QImage, QColor, QDrag

THUMB_SIZE = 200
COLS = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Stdout → Qt signal bridge
# ---------------------------------------------------------------------------

class _LogStream:
    def __init__(self, signal):
        self._signal = signal
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            cleaned = line.replace("\r", "").strip()
            if cleaned:
                self._signal.emit(cleaned)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Analysis worker
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        import sys as _sys
        import traceback
        old_out, old_err = _sys.stdout, _sys.stderr
        stream = _LogStream(self.log)
        _sys.stdout = stream
        _sys.stderr = stream
        try:
            from autocull import run
            run(**self.params)
            self.finished.emit(True)
        except Exception:
            self.log.emit(traceback.format_exc())
            self.finished.emit(False)
        finally:
            _sys.stdout = old_out
            _sys.stderr = old_err


# ---------------------------------------------------------------------------
# Thumbnail loader (background thread)
# ---------------------------------------------------------------------------

class ThumbnailLoader(QThread):
    ready = pyqtSignal(str, QImage)  # kind, image
    done = pyqtSignal()

    def __init__(self, kept: list[Path], rejected: list[Path]):
        super().__init__()
        self._kept = kept
        self._rejected = rejected

    def run(self):
        for path in self._kept:
            self.ready.emit("kept", self._load(path))
        for path in self._rejected:
            self.ready.emit("rej", self._load(path))
        self.done.emit()

    def _load(self, path: Path) -> QImage:
        try:
            from PIL import Image
            img = Image.open(path)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            img = img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, 3 * img.width,
                          QImage.Format.Format_RGB888)
            return qimg.copy()
        except Exception:
            placeholder = QImage(THUMB_SIZE, THUMB_SIZE, QImage.Format.Format_RGB888)
            placeholder.fill(QColor(200, 200, 200))
            return placeholder


# ---------------------------------------------------------------------------
# Thumbnail card — draggable
# ---------------------------------------------------------------------------

class ThumbnailCard(QFrame):
    def __init__(self, path: Path, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._path = path
        self._pixmap = pixmap
        self._drag_start: QPoint | None = None

        self.setFixedWidth(THUMB_SIZE + 12)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 3)
        layout.setSpacing(3)

        img_label = QLabel()
        img_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setPixmap(pixmap)
        layout.addWidget(img_label)

        name_label = QLabel()
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setFixedWidth(THUMB_SIZE)
        elided = name_label.fontMetrics().elidedText(
            path.name, Qt.TextElideMode.ElideMiddle, THUMB_SIZE
        )
        name_label.setText(elided)
        name_label.setToolTip(str(path))
        name_label.setStyleSheet("font-size: 9px; color: #555;")
        layout.addWidget(name_label)

    # ── Drag initiation ────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self._drag_start is None:
            return
        if (event.pos() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(str(self._path))
        drag.setMimeData(mime)
        drag.setPixmap(
            self._pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
        )
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        drag.exec(Qt.DropAction.MoveAction)
        self.setCursor(Qt.CursorShape.OpenHandCursor)


# ---------------------------------------------------------------------------
# Thumbnail grid — drop target
# ---------------------------------------------------------------------------

class ThumbnailGrid(QScrollArea):
    drop_received = pyqtSignal(str)  # path_str dropped onto this grid

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind  # "kept" or "rej"
        self._cards: dict[str, ThumbnailCard] = {}   # path_str → card
        self._order: list[str] = []                   # ordered path_strs

        self.setWidgetResizable(True)
        self.setAcceptDrops(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(6)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setWidget(self._container)

    @property
    def _count(self) -> int:
        return len(self._order)

    def add_card(self, path: Path, pixmap: QPixmap):
        path_str = str(path)
        card = ThumbnailCard(path, pixmap)
        self._cards[path_str] = card
        self._order.append(path_str)
        i = len(self._order) - 1
        self._grid.addWidget(card, i // COLS, i % COLS)

    def remove_card(self, path_str: str) -> QPixmap | None:
        if path_str not in self._cards:
            return None
        card = self._cards.pop(path_str)
        self._order.remove(path_str)
        pixmap = card._pixmap
        card.hide()
        self._grid.removeWidget(card)
        card.setParent(None)
        card.deleteLater()
        self._rebuild_layout()
        return pixmap

    def _rebuild_layout(self):
        for p in self._order:
            self._cards[p].hide()
        while self._grid.count():
            self._grid.takeAt(0)
        for i, p in enumerate(self._order):
            card = self._cards[p]
            self._grid.addWidget(card, i // COLS, i % COLS)
            card.show()

    def clear_all(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()
        self._order.clear()

    # ── Drop events ────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            color = "#4CAF50" if self.kind == "kept" else "#EF5350"
            self.setStyleSheet(f"QScrollArea {{ border: 2px solid {color}; }}")
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event):
        self.setStyleSheet("")
        if event.mimeData().hasText():
            self.drop_received.emit(event.mimeData().text())
            event.acceptProposedAction()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoCull")
        self.setMinimumSize(1100, 800)
        self._worker: AnalysisWorker | None = None
        self._loader: ThumbnailLoader | None = None
        self._output_dir: Path | None = None
        self._dry_run = False
        self._mode = "copy"
        self._kept_paths: list[Path] = []
        self._rej_paths: list[Path] = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Settings ──────────────────────────────────────────────────────
        settings = QGroupBox("설정")
        sl = QVBoxLayout(settings)

        row = QHBoxLayout()
        lbl = QLabel("입력 폴더:")
        lbl.setFixedWidth(70)
        row.addWidget(lbl)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("분석할 사진 폴더")
        row.addWidget(self.input_edit)
        btn = QPushButton("선택")
        btn.setFixedWidth(60)
        btn.clicked.connect(self._pick_input_folder)
        row.addWidget(btn)
        sl.addLayout(row)

        row = QHBoxLayout()
        lbl = QLabel("출력 폴더:")
        lbl.setFixedWidth(70)
        row.addWidget(lbl)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("결과 저장 폴더 (비워두면 입력 폴더에 best/rejected 생성)")
        row.addWidget(self.output_edit)
        btn = QPushButton("선택")
        btn.setFixedWidth(60)
        btn.clicked.connect(lambda: self._pick_folder(self.output_edit))
        row.addWidget(btn)
        sl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("세션 간격(초):"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 3600)
        self.gap_spin.setValue(0)
        self.gap_spin.setSpecialValueText("자동")
        self.gap_spin.setFixedWidth(80)
        row.addWidget(self.gap_spin)
        row.addSpacing(20)
        row.addWidget(QLabel("선명도 기준:"))
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0, 10000)
        self.blur_spin.setValue(100.0)
        self.blur_spin.setFixedWidth(80)
        row.addWidget(self.blur_spin)
        row.addSpacing(20)
        row.addWidget(QLabel("파일 처리:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["copy", "move", "remove"])
        row.addWidget(self.mode_combo)
        row.addStretch()
        sl.addLayout(row)

        row = QHBoxLayout()
        self.dry_run_cb = QCheckBox("Dry-run (미리보기만)")
        self.log_cb = QCheckBox("로그 저장 (CSV)")
        self.recursive_cb = QCheckBox("하위 폴더 포함")
        row.addWidget(self.dry_run_cb)
        row.addSpacing(16)
        row.addWidget(self.log_cb)
        row.addSpacing(16)
        row.addWidget(self.recursive_cb)
        row.addStretch()
        sl.addLayout(row)

        root.addWidget(settings)

        # ── Run button + progress ──────────────────────────────────────────
        row = QHBoxLayout()
        self.run_btn = QPushButton("분석 시작")
        self.run_btn.setFixedHeight(42)
        f = self.run_btn.font()
        f.setPointSize(11)
        f.setBold(True)
        self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(42)
        self.progress_bar.setVisible(False)
        row.addWidget(self.progress_bar, 2)
        root.addLayout(row)

        # ── Tabs ───────────────────────────────────────────────────────────
        self.tabs = QTabWidget()

        # Log tab
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.tabs.addTab(self.log_edit, "로그")

        # Results tab
        results_widget = QWidget()
        rl = QVBoxLayout(results_widget)
        rl.setContentsMargins(0, 4, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        kept_w = QWidget()
        kl = QVBoxLayout(kept_w)
        kl.setContentsMargins(0, 0, 4, 0)
        self.kept_label = QLabel("보관 (0)  💡 드래그해서 ↔ 이동")
        self.kept_label.setStyleSheet(
            "font-weight: bold; color: #2e7d32; font-size: 13px; padding: 2px 4px;"
        )
        kl.addWidget(self.kept_label)
        self.kept_grid = ThumbnailGrid("kept")
        self.kept_grid.drop_received.connect(self._on_kept_grid_drop)
        kl.addWidget(self.kept_grid)
        splitter.addWidget(kept_w)
        splitter.setStretchFactor(0, 1)

        rej_w = QWidget()
        jl = QVBoxLayout(rej_w)
        jl.setContentsMargins(4, 0, 0, 0)
        self.rej_label = QLabel("제외 (0)")
        self.rej_label.setStyleSheet(
            "font-weight: bold; color: #c62828; font-size: 13px; padding: 2px 4px;"
        )
        jl.addWidget(self.rej_label)
        self.rej_grid = ThumbnailGrid("rej")
        self.rej_grid.drop_received.connect(self._on_rej_grid_drop)
        jl.addWidget(self.rej_grid)
        splitter.addWidget(rej_w)
        splitter.setStretchFactor(1, 1)

        rl.addWidget(splitter, 1)
        self.tabs.addTab(results_widget, "결과")

        root.addWidget(self.tabs, 1)

    # ── Folder picker ──────────────────────────────────────────────────────

    def _pick_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if path:
            self.input_edit.setText(path)
            if not self.output_edit.text().strip():
                self.output_edit.setText(path)

    def _pick_folder(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if path:
            edit.setText(path)

    # ── Run ────────────────────────────────────────────────────────────────

    def _on_run(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not input_dir:
            self.log_edit.append("[오류] 입력 폴더를 선택하세요.")
            self.tabs.setCurrentIndex(0)
            return
        if not output_dir:
            output_dir = input_dir
            self.output_edit.setText(output_dir)

        mode = self.mode_combo.currentText()

        if mode == "remove":
            reply = QMessageBox.warning(
                self,
                "주의",
                "실행 시 제외된 파일은 영구 삭제됩니다.\n그래도 하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._output_dir = Path(output_dir)
        self._dry_run = self.dry_run_cb.isChecked()
        self._mode = mode

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_edit.clear()
        self.kept_grid.clear_all()
        self.rej_grid.clear_all()
        self.kept_label.setText("보관 (0)")
        self.rej_label.setText("제외 (0)")
        self.tabs.setCurrentIndex(0)

        params = {
            "input_dir": Path(input_dir),
            "output_dir": self._output_dir,
            "gap": self.gap_spin.value() or None,
            "blur_threshold": self.blur_spin.value(),
            "mode": mode,
            "dry_run": self._dry_run,
            "log": self.log_cb.isChecked(),
            "recursive": self.recursive_cb.isChecked(),
        }

        self._worker = AnalysisWorker(params)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.start()

    def _on_log(self, text: str):
        self.log_edit.append(text)
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_analysis_done(self, success: bool):
        self.progress_bar.setVisible(False)
        if not success or self._dry_run or self._output_dir is None or self._mode == "remove":
            self.run_btn.setEnabled(True)
            if self._mode == "remove" and success:
                self.log_edit.append(
                    "\n[remove 모드] 제외 파일이 삭제되었습니다. "
                    "결과 탭은 copy/move 모드에서만 사용 가능합니다."
                )
            return

        best_dir = self._output_dir / "best"
        rej_dir = self._output_dir / "rejected"

        self._kept_paths = sorted(
            p for p in (best_dir.iterdir() if best_dir.exists() else [])
            if p.suffix.lower() in IMAGE_EXTS
        )
        self._rej_paths = sorted(
            p for p in (rej_dir.iterdir() if rej_dir.exists() else [])
            if p.suffix.lower() in IMAGE_EXTS
        )

        self.kept_label.setText(f"보관 ({len(self._kept_paths)}) — 로딩 중...  💡 드래그해서 ↔ 이동")
        self.rej_label.setText(f"제외 ({len(self._rej_paths)}) — 로딩 중...")
        self.tabs.setCurrentIndex(1)

        self._loader = ThumbnailLoader(self._kept_paths, self._rej_paths)
        self._loader.ready.connect(self._on_thumb_ready)
        self._loader.done.connect(self._on_load_done)
        self._loader.start()

    def _on_thumb_ready(self, kind: str, qimg: QImage):
        pixmap = QPixmap.fromImage(qimg)
        if kind == "kept":
            idx = self.kept_grid._count
            if idx < len(self._kept_paths):
                self.kept_grid.add_card(self._kept_paths[idx], pixmap)
        else:
            idx = self.rej_grid._count
            if idx < len(self._rej_paths):
                self.rej_grid.add_card(self._rej_paths[idx], pixmap)

    def _on_load_done(self):
        self.kept_label.setText(f"보관 ({len(self._kept_paths)})  💡 드래그해서 ↔ 이동")
        self.rej_label.setText(f"제외 ({len(self._rej_paths)})")
        self.run_btn.setEnabled(True)

    # ── Drag-and-drop between grids ────────────────────────────────────────

    def _on_kept_grid_drop(self, path_str: str):
        if path_str in self.rej_grid._cards:
            self._move_card(path_str, from_grid=self.rej_grid, to_grid=self.kept_grid)

    def _on_rej_grid_drop(self, path_str: str):
        if path_str in self.kept_grid._cards:
            self._move_card(path_str, from_grid=self.kept_grid, to_grid=self.rej_grid)

    def _move_card(self, path_str: str, from_grid: ThumbnailGrid, to_grid: ThumbnailGrid):
        if self._output_dir is None:
            return

        pixmap = from_grid.remove_card(path_str)
        if pixmap is None:
            return

        src = Path(path_str)
        dest_dir = self._output_dir / ("best" if to_grid.kind == "kept" else "rejected")
        dest = dest_dir / src.name

        # Avoid overwriting existing file
        if dest.exists() and dest != src:
            stem, suffix = src.stem, src.suffix
            i = 2
            while (dest_dir / f"{stem}_{i}{suffix}").exists():
                i += 1
            dest = dest_dir / f"{stem}_{i}{suffix}"

        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            self.log_edit.append(f"[오류] 파일 이동 실패: {e}")
            from_grid.add_card(src, pixmap)  # revert
            return

        to_grid.add_card(dest, pixmap)
        self.kept_label.setText(f"보관 ({len(self.kept_grid._cards)})")
        self.rej_label.setText(f"제외 ({len(self.rej_grid._cards)})")


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
