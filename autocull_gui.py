import sys
import json
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QTabWidget,
    QScrollArea, QGridLayout, QFrame, QGroupBox, QSplitter, QMessageBox,
    QDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QMimeData, QSettings, QTimer
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
# Workers
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


class ThumbnailLoader(QThread):
    ready = pyqtSignal(str, str, QImage)  # kind, path_str, image
    done = pyqtSignal()

    def __init__(self, kept: list[Path], rejected: list[Path]):
        super().__init__()
        self._kept = kept
        self._rejected = rejected

    def run(self):
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        items = [("kept", p) for p in self._kept] + [("rej", p) for p in self._rejected]
        workers = min(8, os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to = {executor.submit(self._load, p): (kind, p) for kind, p in items}
            for future in as_completed(future_to):
                kind, path = future_to[future]
                self.ready.emit(kind, str(path), future.result())

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


class OrganizeWorker(QThread):
    done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, best_dir: Path):
        super().__init__()
        self._best_dir = best_dir

    def run(self):
        try:
            from autocull import organize_by_location
            counts = organize_by_location(self._best_dir)
            self.done.emit(counts)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Image preview dialog  (기능 1)
# ---------------------------------------------------------------------------

class ImagePreviewDialog(QDialog):
    def __init__(self, paths: list[Path], idx: int, meta: dict | None = None, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._idx = idx
        self._meta = meta or {}

        self.setWindowTitle("미리보기")
        self.setModal(True)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, int(screen.width() * 0.9)),
                    min(960,  int(screen.height() * 0.9)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet("background: #111;")
        layout.addWidget(self._img_label, 1)

        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #ccc; font-size: 11px; padding: 2px;")
        layout.addWidget(self._info_label)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀  이전")
        self._prev_btn.setFixedWidth(100)
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn = QPushButton("다음  ▶")
        self._next_btn.setFixedWidth(100)
        self._next_btn.clicked.connect(self._next)
        nav.addWidget(self._prev_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        self._show_current()

    def _show_current(self):
        path = self._paths[self._idx]
        avail_w = self.width() - 24
        avail_h = self.height() - 100
        try:
            from PIL import Image
            img = Image.open(path)
            img.thumbnail((avail_w, avail_h), Image.LANCZOS)
            img = img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, 3 * img.width,
                          QImage.Format.Format_RGB888)
            self._img_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            self._img_label.setText("이미지를 불러올 수 없습니다.")

        info = f"{path.name}  ({self._idx + 1} / {len(self._paths)})"
        m = self._meta.get(path.name, {})
        if m:
            blur = m.get("blur_score", "")
            faces = m.get("face_count", 0)
            eyes_ok = not m.get("eyes_closed", False)
            reason = m.get("reason", "")
            parts = []
            if blur != "":
                parts.append(f"선명도 {blur:.0f}")
            if faces:
                parts.append(f"얼굴 {faces}명")
                parts.append("눈 정상" if eyes_ok else "눈 감음")
            if reason:
                parts.append(f"제외 사유: {reason}")
            if parts:
                info += "  |  " + "  |  ".join(parts)
        self._info_label.setText(info)

        self._prev_btn.setEnabled(self._idx > 0)
        self._next_btn.setEnabled(self._idx < len(self._paths) - 1)

    def _prev(self):
        if self._idx > 0:
            self._idx -= 1
            self._show_current()

    def _next(self):
        if self._idx < len(self._paths) - 1:
            self._idx += 1
            self._show_current()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self._prev()
        elif event.key() == Qt.Key.Key_Right:
            self._next()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Thumbnail card — draggable, double-click preview  (기능 1·4)
# ---------------------------------------------------------------------------

class ThumbnailCard(QFrame):
    double_clicked = pyqtSignal(str)

    def __init__(self, path: Path, pixmap: QPixmap, meta: dict | None = None, parent=None):
        super().__init__(parent)
        self._path = path
        self._pixmap = pixmap
        self._drag_start: QPoint | None = None

        self.setFixedWidth(THUMB_SIZE + 12)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 3)
        layout.setSpacing(2)

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

        # 분석 배지 (기능 4)
        if meta:
            blur = meta.get("blur_score", "")
            faces = meta.get("face_count", 0)
            eyes_ok = not meta.get("eyes_closed", False)
            reason = meta.get("reason", "")
            parts = []
            if blur != "":
                parts.append(f"⚡{blur:.0f}")
            if faces:
                parts.append(f"👤{faces}")
                parts.append("✓" if eyes_ok else "😑")
            if reason:
                parts.append(f"[{reason}]")
            if parts:
                badge = QLabel("  ".join(parts))
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                badge.setFixedWidth(THUMB_SIZE)
                badge.setStyleSheet("font-size: 9px; color: #888;")
                layout.addWidget(badge)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(str(self._path))
        super().mouseDoubleClickEvent(event)

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
    drop_received = pyqtSignal(str)
    card_double_clicked = pyqtSignal(str)

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self._cards: dict[str, ThumbnailCard] = {}
        self._order: list[str] = []

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

    def add_card(self, path: Path, pixmap: QPixmap, meta: dict | None = None):
        path_str = str(path)
        card = ThumbnailCard(path, pixmap, meta)
        card.double_clicked.connect(self.card_double_clicked)
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
        self._reclass_loader: ThumbnailLoader | None = None
        self._org_worker: OrganizeWorker | None = None
        self._output_dir: Path | None = None
        self._dry_run = False
        self._mode = "copy"
        self._kept_paths: list[Path] = []
        self._rej_paths: list[Path] = []
        self._reclass_kept_paths: list[Path] = []
        self._reclass_rej_paths: list[Path] = []
        self._meta: dict = {}
        self._build_ui()
        self._load_settings()  # 기능 6

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(self._build_classify_tab(), "분류")
        self.main_tabs.addTab(self._build_reclass_tab(), "재분류")
        root.addWidget(self.main_tabs)

    def _build_classify_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

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

        layout.addWidget(settings)

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
        layout.addLayout(row)

        self.classify_tabs = QTabWidget()

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.classify_tabs.addTab(self.log_edit, "로그")

        results_w = QWidget()
        rl = QVBoxLayout(results_w)
        rl.setContentsMargins(0, 4, 0, 0)

        self._classify_splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter = self._classify_splitter

        kept_w = QWidget()
        kl = QVBoxLayout(kept_w)
        kl.setContentsMargins(0, 0, 4, 0)
        kl.setSpacing(2)
        kept_header = QHBoxLayout()
        self.kept_label = QLabel("보관 (0)  💡 더블클릭=미리보기 / 드래그=이동")
        self.kept_label.setStyleSheet(
            "font-weight: bold; color: #2e7d32; font-size: 13px; padding: 2px 4px;"
        )
        kept_header.addWidget(self.kept_label, 1)
        self._gps_btn = QPushButton("📍 장소별 정리")
        self._gps_btn.setFixedHeight(26)
        self._gps_btn.setEnabled(False)
        self._gps_btn.clicked.connect(self._on_organize_gps)
        kept_header.addWidget(self._gps_btn)
        kl.addLayout(kept_header)
        self.kept_grid = ThumbnailGrid("kept")
        self.kept_grid.drop_received.connect(self._on_kept_grid_drop)
        self.kept_grid.card_double_clicked.connect(
            lambda p: self._open_preview(p, self.kept_grid)
        )
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
        self.rej_grid.card_double_clicked.connect(
            lambda p: self._open_preview(p, self.rej_grid)
        )
        jl.addWidget(self.rej_grid)
        splitter.addWidget(rej_w)
        splitter.setStretchFactor(1, 1)

        rl.addWidget(splitter, 1)
        self.classify_tabs.addTab(results_w, "결과")

        layout.addWidget(self.classify_tabs, 1)
        return w

    def _build_reclass_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        folders = QGroupBox("폴더 선택")
        fl = QVBoxLayout(folders)

        row = QHBoxLayout()
        lbl = QLabel("보관 폴더:")
        lbl.setFixedWidth(80)
        row.addWidget(lbl)
        self.reclass_best_edit = QLineEdit()
        self.reclass_best_edit.setPlaceholderText("best 폴더 경로")
        row.addWidget(self.reclass_best_edit)
        btn = QPushButton("선택")
        btn.setFixedWidth(60)
        btn.clicked.connect(lambda: self._pick_folder(self.reclass_best_edit))
        row.addWidget(btn)
        fl.addLayout(row)

        row = QHBoxLayout()
        lbl = QLabel("제외 폴더:")
        lbl.setFixedWidth(80)
        row.addWidget(lbl)
        self.reclass_rej_edit = QLineEdit()
        self.reclass_rej_edit.setPlaceholderText("rejected 폴더 경로")
        row.addWidget(self.reclass_rej_edit)
        btn = QPushButton("선택")
        btn.setFixedWidth(60)
        btn.clicked.connect(lambda: self._pick_folder(self.reclass_rej_edit))
        row.addWidget(btn)
        fl.addLayout(row)

        layout.addWidget(folders)

        row = QHBoxLayout()
        self.reclass_load_btn = QPushButton("사진 불러오기")

        self.reclass_load_btn.setFixedHeight(42)
        f = self.reclass_load_btn.font()
        f.setPointSize(11)
        f.setBold(True)
        self.reclass_load_btn.setFont(f)
        self.reclass_load_btn.clicked.connect(self._on_reclass_load)
        row.addWidget(self.reclass_load_btn, 1)
        self.reclass_progress = QProgressBar()
        self.reclass_progress.setRange(0, 0)
        self.reclass_progress.setFixedHeight(42)
        self.reclass_progress.setVisible(False)
        row.addWidget(self.reclass_progress, 2)
        layout.addLayout(row)

        self._reclass_splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter = self._reclass_splitter

        rk_w = QWidget()
        rkl = QVBoxLayout(rk_w)
        rkl.setContentsMargins(0, 0, 4, 0)
        self.reclass_kept_label = QLabel("보관 (0)  💡 더블클릭=미리보기 / 드래그=이동")
        self.reclass_kept_label.setStyleSheet(
            "font-weight: bold; color: #2e7d32; font-size: 13px; padding: 2px 4px;"
        )
        rkl.addWidget(self.reclass_kept_label)
        self.reclass_kept_grid = ThumbnailGrid("kept")
        self.reclass_kept_grid.drop_received.connect(self._on_reclass_kept_drop)
        self.reclass_kept_grid.card_double_clicked.connect(
            lambda p: self._open_preview(p, self.reclass_kept_grid)
        )
        rkl.addWidget(self.reclass_kept_grid)
        splitter.addWidget(rk_w)
        splitter.setStretchFactor(0, 1)

        rr_w = QWidget()
        rrl = QVBoxLayout(rr_w)
        rrl.setContentsMargins(4, 0, 0, 0)
        self.reclass_rej_label = QLabel("제외 (0)")
        self.reclass_rej_label.setStyleSheet(
            "font-weight: bold; color: #c62828; font-size: 13px; padding: 2px 4px;"
        )
        rrl.addWidget(self.reclass_rej_label)
        self.reclass_rej_grid = ThumbnailGrid("rej")
        self.reclass_rej_grid.drop_received.connect(self._on_reclass_rej_drop)
        self.reclass_rej_grid.card_double_clicked.connect(
            lambda p: self._open_preview(p, self.reclass_rej_grid)
        )
        rrl.addWidget(self.reclass_rej_grid)
        splitter.addWidget(rr_w)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)
        return w

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._equalize_splitters)

    def _equalize_splitters(self):
        for sp in (self._classify_splitter, self._reclass_splitter):
            w = sp.width()
            if w > 0:
                sp.setSizes([w // 2, w // 2])

    # ── Settings persistence  (기능 6) ─────────────────────────────────────

    def _load_settings(self):
        s = QSettings("AutoCull", "AutoCull")
        self.input_edit.setText(s.value("input_dir", ""))
        self.output_edit.setText(s.value("output_dir", ""))
        self.blur_spin.setValue(float(s.value("blur_threshold", 100.0)))
        idx = self.mode_combo.findText(s.value("mode", "copy"))
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.dry_run_cb.setChecked(s.value("dry_run", False, type=bool))
        self.log_cb.setChecked(s.value("log_csv", False, type=bool))
        self.recursive_cb.setChecked(s.value("recursive", False, type=bool))

    def closeEvent(self, event):
        s = QSettings("AutoCull", "AutoCull")
        s.setValue("input_dir", self.input_edit.text())
        s.setValue("output_dir", self.output_edit.text())
        s.setValue("blur_threshold", self.blur_spin.value())
        s.setValue("mode", self.mode_combo.currentText())
        s.setValue("dry_run", self.dry_run_cb.isChecked())
        s.setValue("log_csv", self.log_cb.isChecked())
        s.setValue("recursive", self.recursive_cb.isChecked())
        event.accept()

    # ── Folder pickers ─────────────────────────────────────────────────────

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

    # ── Preview  (기능 1) ──────────────────────────────────────────────────

    def _open_preview(self, path_str: str, grid: ThumbnailGrid):
        paths = [Path(p) for p in grid._order]
        try:
            idx = grid._order.index(path_str)
        except ValueError:
            return
        dlg = ImagePreviewDialog(paths, idx, self._meta, self)
        dlg.exec()

    # ── 분류 tab: run ──────────────────────────────────────────────────────

    def _on_run(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not input_dir:
            self.log_edit.append("[오류] 입력 폴더를 선택하세요.")
            self.classify_tabs.setCurrentIndex(0)
            return
        if not output_dir:
            output_dir = input_dir
            self.output_edit.setText(output_dir)

        mode = self.mode_combo.currentText()

        if mode == "remove":
            reply = QMessageBox.warning(
                self, "주의",
                "실행 시 제외된 파일은 영구 삭제됩니다.\n그래도 하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._output_dir = Path(output_dir)
        self._dry_run = self.dry_run_cb.isChecked()
        self._mode = mode
        self._meta = {}
        self._gps_btn.setEnabled(False)

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_edit.clear()
        self.kept_grid.clear_all()
        self.rej_grid.clear_all()
        self.kept_label.setText("보관 (0)")
        self.rej_label.setText("제외 (0)")
        self.classify_tabs.setCurrentIndex(0)

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

        # 메타 로드 (배지용, 기능 4)
        meta_path = self._output_dir / ".autocull_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    self._meta = json.load(f)
            except Exception:
                self._meta = {}

        self._kept_paths = sorted(
            p for p in (best_dir.iterdir() if best_dir.exists() else [])
            if p.suffix.lower() in IMAGE_EXTS
        )
        self._rej_paths = sorted(
            p for p in (rej_dir.iterdir() if rej_dir.exists() else [])
            if p.suffix.lower() in IMAGE_EXTS
        )

        # 재분류 탭 폴더 자동 채우기
        self.reclass_best_edit.setText(str(best_dir))
        self.reclass_rej_edit.setText(str(rej_dir))

        self.kept_label.setText(
            f"보관 ({len(self._kept_paths)}) — 로딩 중...  💡 더블클릭=미리보기 / 드래그=이동"
        )
        self.rej_label.setText(f"제외 ({len(self._rej_paths)}) — 로딩 중...")
        self.classify_tabs.setCurrentIndex(1)

        self._loader = ThumbnailLoader(self._kept_paths, self._rej_paths)
        self._loader.ready.connect(self._on_thumb_ready)
        self._loader.done.connect(self._on_load_done)
        self._loader.start()

    def _on_thumb_ready(self, kind: str, path_str: str, qimg: QImage):
        path = Path(path_str)
        pixmap = QPixmap.fromImage(qimg)
        if kind == "kept":
            self.kept_grid.add_card(path, pixmap, self._meta.get(path.name))
        else:
            self.rej_grid.add_card(path, pixmap, self._meta.get(path.name))

    def _on_load_done(self):
        self.kept_label.setText(
            f"보관 ({len(self._kept_paths)})  💡 더블클릭=미리보기 / 드래그=이동"
        )
        self.rej_label.setText(f"제외 ({len(self._rej_paths)})")
        self.run_btn.setEnabled(True)
        self._gps_btn.setEnabled(True)

    def _on_kept_grid_drop(self, path_str: str):
        if path_str in self.rej_grid._cards:
            self._move_card(path_str, from_grid=self.rej_grid, to_grid=self.kept_grid,
                            best_dir=self._output_dir / "best",
                            rej_dir=self._output_dir / "rejected",
                            kept_label=self.kept_label, rej_label=self.rej_label)

    def _on_rej_grid_drop(self, path_str: str):
        if path_str in self.kept_grid._cards:
            self._move_card(path_str, from_grid=self.kept_grid, to_grid=self.rej_grid,
                            best_dir=self._output_dir / "best",
                            rej_dir=self._output_dir / "rejected",
                            kept_label=self.kept_label, rej_label=self.rej_label)

    # ── GPS organize  (기능 7) ─────────────────────────────────────────────

    def _on_organize_gps(self):
        if self._output_dir is None:
            return
        reply = QMessageBox.question(
            self, "장소별 정리",
            "보관 폴더의 사진을 GPS 위치별 서브폴더로 정리합니다.\n"
            "GPS 정보가 없는 파일은 unknown/ 폴더로 이동됩니다.\n\n"
            "계속하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._gps_btn.setEnabled(False)
        self._org_worker = OrganizeWorker(self._output_dir / "best")
        self._org_worker.done.connect(self._on_organize_done)
        self._org_worker.error.connect(self._on_organize_error)
        self._org_worker.start()

    def _on_organize_done(self, counts: dict):
        summary = ", ".join(f"{loc}({n}장)" for loc, n in sorted(counts.items()))
        self.log_edit.append(f"\n📍 장소별 정리 완료: {summary}")

        # 썸네일 재로딩 (서브폴더 포함)
        best_dir = self._output_dir / "best"
        self._kept_paths = sorted(
            p for p in best_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        self.kept_grid.clear_all()
        self.kept_label.setText(f"보관 ({len(self._kept_paths)}) — 로딩 중...")
        loader = ThumbnailLoader(self._kept_paths, [])
        loader.ready.connect(self._on_thumb_ready)  # signature: (kind, path_str, qimg)
        loader.done.connect(lambda: self.kept_label.setText(
            f"보관 ({len(self._kept_paths)})  💡 더블클릭=미리보기 / 드래그=이동"
        ))
        loader.start()
        self._loader = loader
        self.classify_tabs.setCurrentIndex(1)

    def _on_organize_error(self, msg: str):
        self.log_edit.append(f"[오류] 장소별 정리 실패: {msg}")
        self._gps_btn.setEnabled(True)

    # ── 재분류 tab ─────────────────────────────────────────────────────────

    def _on_reclass_load(self):
        best_dir = Path(self.reclass_best_edit.text().strip())
        rej_dir = Path(self.reclass_rej_edit.text().strip())

        if not best_dir.is_dir() or not rej_dir.is_dir():
            QMessageBox.warning(self, "오류", "보관/제외 폴더를 모두 선택하세요.")
            return

        self.reclass_kept_grid.clear_all()
        self.reclass_rej_grid.clear_all()

        self._reclass_kept_paths = sorted(
            p for p in best_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        self._reclass_rej_paths = sorted(
            p for p in rej_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

        self.reclass_kept_label.setText(
            f"보관 ({len(self._reclass_kept_paths)}) — 로딩 중...  💡 더블클릭=미리보기 / 드래그=이동"
        )
        self.reclass_rej_label.setText(f"제외 ({len(self._reclass_rej_paths)}) — 로딩 중...")

        self.reclass_load_btn.setEnabled(False)
        self.reclass_progress.setVisible(True)

        self._reclass_loader = ThumbnailLoader(self._reclass_kept_paths, self._reclass_rej_paths)
        self._reclass_loader.ready.connect(self._on_reclass_thumb_ready)
        self._reclass_loader.done.connect(self._on_reclass_load_done)
        self._reclass_loader.start()

    def _on_reclass_thumb_ready(self, kind: str, path_str: str, qimg: QImage):
        path = Path(path_str)
        pixmap = QPixmap.fromImage(qimg)
        if kind == "kept":
            self.reclass_kept_grid.add_card(path, pixmap, self._meta.get(path.name))
        else:
            self.reclass_rej_grid.add_card(path, pixmap, self._meta.get(path.name))

    def _on_reclass_load_done(self):
        self.reclass_kept_label.setText(
            f"보관 ({len(self._reclass_kept_paths)})  💡 더블클릭=미리보기 / 드래그=이동"
        )
        self.reclass_rej_label.setText(f"제외 ({len(self._reclass_rej_paths)})")
        self.reclass_load_btn.setEnabled(True)
        self.reclass_progress.setVisible(False)

    def _on_reclass_kept_drop(self, path_str: str):
        if path_str in self.reclass_rej_grid._cards:
            best_dir = Path(self.reclass_best_edit.text().strip())
            rej_dir = Path(self.reclass_rej_edit.text().strip())
            self._move_card(path_str,
                            from_grid=self.reclass_rej_grid, to_grid=self.reclass_kept_grid,
                            best_dir=best_dir, rej_dir=rej_dir,
                            kept_label=self.reclass_kept_label, rej_label=self.reclass_rej_label)

    def _on_reclass_rej_drop(self, path_str: str):
        if path_str in self.reclass_kept_grid._cards:
            best_dir = Path(self.reclass_best_edit.text().strip())
            rej_dir = Path(self.reclass_rej_edit.text().strip())
            self._move_card(path_str,
                            from_grid=self.reclass_kept_grid, to_grid=self.reclass_rej_grid,
                            best_dir=best_dir, rej_dir=rej_dir,
                            kept_label=self.reclass_kept_label, rej_label=self.reclass_rej_label)

    # ── Shared move logic ──────────────────────────────────────────────────

    def _move_card(self, path_str: str,
                   from_grid: ThumbnailGrid, to_grid: ThumbnailGrid,
                   best_dir: Path, rej_dir: Path,
                   kept_label: QLabel, rej_label: QLabel):
        pixmap = from_grid.remove_card(path_str)
        if pixmap is None:
            return

        src = Path(path_str)
        dest_dir = best_dir if to_grid.kind == "kept" else rej_dir
        dest = dest_dir / src.name

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
            from_grid.add_card(src, pixmap)
            return

        to_grid.add_card(dest, pixmap, self._meta.get(dest.name))
        kg = to_grid if to_grid.kind == "kept" else from_grid
        rg = to_grid if to_grid.kind == "rej" else from_grid
        kept_label.setText(
            f"보관 ({len(kg._cards)})  💡 더블클릭=미리보기 / 드래그=이동"
        )
        rej_label.setText(f"제외 ({len(rg._cards)})")


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
