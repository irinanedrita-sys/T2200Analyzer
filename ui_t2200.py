import sys
import traceback
import time
from pathlib import Path
from datetime import date
import html

from PyQt6.QtCore import Qt, QSize, QEvent, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import (
    QPalette, QColor, QFont, QGuiApplication, QIcon, QCursor,
    QTextCursor, QTextBlockFormat
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolButton, QTextEdit, QFileDialog, QLabel, QToolTip,
    QGraphicsDropShadowEffect, QProgressBar, QSizePolicy,
    QListWidget, QListWidgetItem
)

from t2200_analyzer_final import (
    PDFProcessor, CheckboxAnalyzer, load_config,
    YesAnchor, EmptyYesComparator,
)


class AnalysisWorker(QThread):
    progress = pyqtSignal(str)            # строки лога
    finished_ok = pyqtSignal(dict, float, str)  # (result_dict, seconds, filename)
    failed = pyqtSignal(str)              # traceback

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        t0 = time.perf_counter()
        try:
            # Локальные изоляционные экземпляры (чтобы не трогать GUI-объекты)
            cfg = load_config()
            proc = PDFProcessor()
            ya = YesAnchor("yes_label.png", thr=0.75)
            empty = EmptyYesComparator()
            analyzer = CheckboxAnalyzer(cfg)
            analyzer.yes_anchor = ya
            analyzer.empty_yes = empty

            fname = Path(self.file_path).name
            self.progress.emit(f"[start] Analyzing: {fname}")

            # 1) convert
            self.progress.emit("[step] Converting PDF pages to images…")
            images = proc.convert_to_images(self.file_path)
            self.progress.emit(f"[ok] {len(images)} page(s) converted")

            # 2) analyze
            self.progress.emit("[step] Detecting YES anchor / comparing boxes…")
            try:
                results = analyzer.analyze_images(images)
                self.progress.emit("[ok] Checkbox analysis complete")
            finally:
                proc.cleanup()

            # 3) post-process (как в твоём run_analysis)
            yes_total = 0
            no_total = 0
            yes_texts = []
            for qnum, ans in sorted(results.items(), key=lambda kv: int(kv[0])):
                q = int(qnum)
                if ans == "YES":
                    yes_total += 1
                    if q != 1:
                        txt = cfg.question_texts.get(q)
                        if txt:
                            yes_texts.append(txt)
                else:
                    no_total += 1  # FIXED: было no_total = 1

            res = {
                "file": fname,
                "eligibility_q1": results.get(1),
                "yes_total": yes_total,
                "no_total": no_total,
                "yes_texts": yes_texts,
                "date": date.today().isoformat(),
            }
            dt = time.perf_counter() - t0
            self.finished_ok.emit(res, dt, fname)
        except Exception:
            tb = traceback.format_exc()
            self.failed.emit(tb)


class T2200AnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T2200 Form Analyzer")
        self.setWindowIcon(self._icon("main"))
        self.setGeometry(100, 100, 800, 600)

        # Initialize analyzer components once
        self.cfg = load_config()
        self.pdf_processor = PDFProcessor()
        self.yes_anchor = YesAnchor("yes_label.png", thr=0.75)
        self.empty_yes = EmptyYesComparator()
        self.analyzer = CheckboxAnalyzer(self.cfg)
        self.analyzer.yes_anchor = self.yes_anchor
        self.analyzer.empty_yes = self.empty_yes
        self.is_analyzing = False
        self._queued_file = None
        self.enable_checkmarks = True  # Optional flag to enable/disable checkmarks

        # UI
        self.init_ui()
        self.set_light_theme()

        # State
        self.file_path = None
        self._last_result = None
        self.verbose = False
        self._worker = None
        
    # ---------- UI building ----------

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # Left panel
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right results - changed from QTextEdit to QListWidget
        self.report_output = QListWidget()
        self.report_output.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.report_output.setAcceptDrops(True)
        self.report_output.installEventFilter(self)
        
        # Enable word wrap and spacing
        self.report_output.setWordWrap(True)          # ← перенос по ширине
        self.report_output.setUniformItemSizes(False) # ← разрешаем разные высоты
        self.report_output.setSpacing(4)              # ← «межстрочный» зазор между пунктами
        
        self.report_output.setStyleSheet("""
            QListWidget {
                background-color: white;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 8px;
                padding: 16px;
                font-family: Cambria;
                font-size: 15px;
                selection-background-color: #3D8EC4;
                letter-spacing: 0.3px;
                margin-left: 8px;
                margin-right: 8px;
            }
            QListWidget[dragover="true"] {
                border: 2px dashed #3D8EC4;
                background-color: #F5FBFF;
            }
        """)
        main_layout.addWidget(self.report_output, stretch=4)

    def create_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("LeftPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Font for buttons
        button_font = QFont("Segoe UI", 10)
        button_font.setBold(True)

        # Buttons
        self.load_button = self.create_icon_button(
            "Load\nFile", "upload", "Load a T2200 PDF form", button_font)
        self.set_locked(self.load_button, False)
        self.analyze_button = self.create_icon_button(
            "Analyze", "analyze", "Analyze loaded T2200 form", button_font)
        self.save_button = self.create_icon_button(
            "Save\nResults", "save", "Save analysis results", button_font)
        self.copy_button = self.create_icon_button(
            "Copy to\nClipboard", "copy", "Copy results to clipboard", button_font)
        self.clear_button = self.create_icon_button(
            "Clear", "clear", "Clear all results", button_font)
        self.exit_button = self.create_icon_button(
            "Exit", "exit", "Exit application", button_font)

        # Initial soft-lock states
        self.set_locked(self.analyze_button, True, "Load a PDF first")
        self.set_locked(self.save_button, True, "Analyze the form first")
        self.set_locked(self.copy_button, True, "Analyze the form first")

        # Filename label
        self.filename_label = QLabel("No file loaded")
        self.filename_label.setWordWrap(True)
        self.filename_label.setStyleSheet("""
            QLabel {
                color: #495057;
                font-size: 11px;
                background: #F7F9FB;
                border: 1px solid #E1E6EA;
                border-radius: 4px;
                padding: 6px;
                margin-top: 4px;
            }
        """)

        # Layout fill (кнопки сверху вниз)
        layout.addWidget(self.load_button)
        layout.addWidget(self.filename_label)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.copy_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.exit_button)

        # --- Status area (фиксированная высота, строго внизу) ---
        self.status_area = QWidget()
        sa = QVBoxLayout(self.status_area)
        sa.setContentsMargins(0, 6, 0, 0)
        sa.setSpacing(4)

        self.status_label = QLabel("Idle")
        self.status_label.setVisible(False)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.setVisible(False)
        self.busy.setFixedHeight(8)

        # компактный, чтобы не растягивал панель по ширине
        sp_busy = self.busy.sizePolicy()
        sp_busy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        sp_busy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.busy.setSizePolicy(sp_busy)
        self.busy.setFixedWidth(120)

        sa.addWidget(self.status_label)
        sa.addWidget(self.busy)

        # резерв места под статус всегда (панель больше не «подрастает»)
        self.status_area.setFixedHeight(36)
        sp_area = self.status_area.sizePolicy()
        sp_area.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        sp_area.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
        self.status_area.setSizePolicy(sp_area)

        layout.addWidget(self.status_area)

        # Signals
        self.load_button.clicked.connect(self.load_file)
        self.analyze_button.clicked.connect(
            self._guarded(self.analyze_button, self.analyze, "Load a PDF first"))
        self.save_button.clicked.connect(
            self._guarded(self.save_button, self.save_results, "Analyze the form first"))
        self.copy_button.clicked.connect(
            self._guarded(self.copy_button, self._copy_report, "Analyze the form first"))
        self.clear_button.clicked.connect(self.clear_results)
        self.exit_button.clicked.connect(self.close)

        # Panel style
        panel.setStyleSheet("""
            #LeftPanel {
                background-color: #EDF1F4;
                border: 1px solid #D0D5DA;
                border-radius: 6px;
            }
        """)
        panel.setFixedWidth(panel.sizeHint().width())
        return panel

    def create_icon_button(self, text: str, icon_name: str, tooltip: str, font: QFont) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setIcon(self._icon(icon_name))
        btn.setIconSize(QSize(24, 24))
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setFont(font)
        btn.setToolTip(tooltip)
        btn.setToolTipDuration(8000)
        btn.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        btn.setFixedSize(130, 84)

        # стиль как у остальных кнопок
        btn.setStyleSheet("""
            QToolButton {
                background-color: #0E2735;
                color: #F3F6F9;
                border: 2px solid #0A1E29;
                border-radius: 8px;
                padding: 6px 8px;
                font-weight: 600;
                font-size: 11px;
            }
            QToolButton:hover { background-color: #1B3A4A; }
            QToolButton:pressed { background-color: #091A23; }
            QToolButton[locked="true"] {
                background-color: #1A2F3C;
                color: #D3DAE0;
                border-color: #132532;
            }
            QToolButton[locked="true"]:hover { background-color: #182A35; }
        """)

        # тень
        self._add_shadow(btn)
        return btn

    # ---------- Helpers & styling ----------

    def _show_report(self, report: str):
        """Показываем отчёт как список; чекбоксы только у буллетов."""
        self.report_output.clear()
        is_first_line = True
        
        for raw in report.splitlines():
            line = raw.rstrip("\n")
            # Пустые строки — просто добавляем пустую запись (без чекбоксов)
            if not line.strip():
                self.report_output.addItem(QListWidgetItem(""))
                continue

            item = QListWidgetItem(line)

            # --- Заголовок: первая непустая строка —
            if is_first_line:
                is_first_line = False
                f = QFont("Cambria", 13)
                f.setBold(True)
                item.setFont(f)
                # у заголовка чекбокса не будет:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

            # --- Буллеты: чекбокс только у «• ...»
            elif self.enable_checkmarks and line.lstrip().startswith("• "):
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

            self.report_output.addItem(item)
    
    def _collect_checked_points(self) -> list[str]:
        points = []
        for i in range(self.report_output.count()):
            it = self.report_output.item(i)
            txt = it.text()
            # Берём только отмеченные буллеты
            if txt.lstrip().startswith("• ") and \
               (it.flags() & Qt.ItemFlag.ItemIsUserCheckable) and \
               it.checkState() == Qt.CheckState.Checked:
                # уберём лидер "• " при необходимости
                clean = txt.lstrip()[2:].strip() if txt.lstrip().startswith("• ") else txt
                points.append(clean)
        return points
    
    def eventFilter(self, obj, event):
        if obj is self.report_output:
            et = event.type()

            if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                if self.is_analyzing:
                    event.ignore()
                    return True

                md = event.mimeData()
                ok = False
                if md.hasUrls():
                    urls = [u for u in md.urls() if u.isLocalFile()]
                    if urls:
                        fpath = urls[0].toLocalFile()
                        ok = fpath.lower().endswith(".pdf")

                if ok:
                    self.report_output.setProperty("dragover", True)
                    self.report_output.style().unpolish(self.report_output)
                    self.report_output.style().polish(self.report_output)
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return True

            if et == QEvent.Type.DragLeave:
                self.report_output.setProperty("dragover", False)
                self.report_output.style().unpolish(self.report_output)
                self.report_output.style().polish(self.report_output)
                return True

            if et == QEvent.Type.Drop:
                self.report_output.setProperty("dragover", False)
                self.report_output.style().unpolish(self.report_output)
                self.report_output.style().polish(self.report_output)

                md = event.mimeData()
                urls = [u for u in md.urls() if u.isLocalFile()]
                if urls:
                    fpath = urls[0].toLocalFile()
                    if fpath.lower().endswith(".pdf"):
                        if self.is_analyzing:
                            self._queued_file = fpath
                            self._log(f"[queue] Will analyze next: {fpath}")
                        else:
                            self._prepare_new_load(fpath)
                            self._log(f"Dropped file: {fpath}")
                        event.acceptProposedAction()
                        return True

                event.ignore()
                return True

        return super().eventFilter(obj, event)

    def set_locked(self, btn: QToolButton, locked: bool, tooltip_if_locked: str = None):
        """Lock/unlock buttons with visual feedback"""
        btn.setProperty("locked", "true" if locked else "false")
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        if tooltip_if_locked:
            btn.setToolTip(tooltip_if_locked if locked else btn.toolTip())

    def _guarded(self, btn: QToolButton, slot, locked_message: str):
        """Guard against clicking locked buttons"""
        def wrapper():
            locked_prop = btn.property("locked")
            locked_now = (str(locked_prop).lower() == "true")
            if locked_now:
                QToolTip.showText(QCursor.pos(), locked_message)
                return
            slot()
        return wrapper

    def _log(self, msg: str):
        """Append a line to output only if verbose mode is on."""
        if not getattr(self, "verbose", False):
            return
        # For logging, we'll use a temporary approach since we don't have QTextEdit anymore
        print(msg)  # Or implement a different logging mechanism if needed

    def _busy(self, on: bool, text: str = None):
        self.busy.setVisible(bool(on))
        self.status_label.setVisible(bool(on))
        if on and text is not None:
            self.status_label.setText(text)
        QApplication.processEvents()

    def _flash_status(self, text: str, ms: int = 1500):
        from PyQt6.QtCore import QTimer
        self._busy(True, text)
        QTimer.singleShot(ms, lambda: self._busy(False))

    def _on_worker_progress(self, line: str):
        txt = line
        for tag in ("[start] ", "[step] ", "[ok] "):
            if line.startswith(tag):
                txt = line[len(tag):]
                break
        self._busy(True, txt)

    def _on_worker_failed(self, tb: str):
        self.status_label.setText("Error during analysis")
        # Clear and show error message in list widget
        self.report_output.clear()
        error_item = QListWidgetItem("Error during analysis. Please try another PDF.")
        self.report_output.addItem(error_item)
        if getattr(self, "verbose", False):
            for line in tb.splitlines():
                if line.strip():
                    tb_item = QListWidgetItem(line)
                    self.report_output.addItem(tb_item)
        self._finish_analysis_and_maybe_run_next()

    def _on_worker_finished(self, result: dict, dt: float, fname: str):
        self._busy(False)
        self._last_result = result

        report = self._build_report_text(result)
        self._show_report(report)     # ← Now shows as checklist

        # Больше ничего не форматируем курсором
        self._finish_analysis_and_maybe_run_next()

    def _finish_analysis_and_maybe_run_next(self):
        self.is_analyzing = False
        self.set_locked(self.load_button, False)
        self.set_locked(self.analyze_button, False)
        
        if self.report_output.count() > 0:
            self.set_locked(self.save_button, False)
            self.set_locked(self.copy_button, False)

        if self._queued_file:
            next_path = self._queued_file
            self._queued_file = None
            QTimer.singleShot(100, lambda: self._load_next_file(next_path))
        else:
            self._busy(False, "Idle")

    def _load_next_file(self, file_path):
        try:
            self._prepare_new_load(file_path, clear_output=False)
            self.set_locked(self.analyze_button, False)
            self._busy(False)
        except Exception as e:
            # Add error to list widget
            error_item = QListWidgetItem(f"\nError loading next file: {type(e).__name__}: {e}")
            self.report_output.addItem(error_item)
            if getattr(self, "verbose", False):
                import traceback
                for line in traceback.format_exc().splitlines():
                    if line.strip():
                        tb_item = QListWidgetItem(line)
                        self.report_output.addItem(tb_item)
            self._busy(False, "Idle")

    def set_light_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(70, 130, 180))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        self.setPalette(palette)

    def _icon(self, name: str) -> QIcon:
        icons_dir = Path(__file__).resolve().parent / "icons"
        icon_path = icons_dir / f"{name}.png"
        return QIcon(str(icon_path)) if icon_path.exists() else QIcon()

    def _add_shadow(self, w, blur: int = 16, dx: int = 0, dy: int = 2, alpha: int = 80):
        eff = QGraphicsDropShadowEffect(self)
        eff.setBlurRadius(blur)
        eff.setOffset(dx, dy)
        eff.setColor(QColor(0, 0, 0, alpha))
        w.setGraphicsEffect(eff)

    # ---------- Actions ----------

    def load_file(self):
        if self.is_analyzing:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open T2200 Form", "", "PDF Files (*.pdf);;All Files (*)"
            )
            if file_path:
                self._queued_file = file_path
                self._log(f"[queue] Will analyze next: {file_path}")
                self._busy(True, "Queued next file (analyzing current)…")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open T2200 Form", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if file_path:
            self._prepare_new_load(file_path)
            self._log(f"[load] Loaded file: {file_path}")
            self._busy(False, f"Ready: {Path(file_path).name}")
        
    def _prepare_new_load(self, file_path: str, *, clear_output: bool = True):
        if clear_output:
            self.report_output.setProperty("dragover", False)
            self.report_output.style().unpolish(self.report_output)
            self.report_output.style().polish(self.report_output)
            self.report_output.clear()

        try:
            self.pdf_processor.cleanup()
        except Exception:
            pass

        self.file_path = file_path
        self._last_result = None
        self.filename_label.setText(f"Loaded: {Path(file_path).name}")
        self.is_analyzing = False

        self.set_locked(self.analyze_button, False)
        self.set_locked(self.save_button, True, "Analyze the form first")
        self.set_locked(self.copy_button, True, "Analyze the form first")

    def analyze(self):
        if not self.file_path:
            # Add error to list widget
            error_item = QListWidgetItem("Error: Please load a PDF file first")
            self.report_output.addItem(error_item)
            return
        if self.is_analyzing:
            QToolTip.showText(QCursor.pos(), "Please wait — analyzing…")
            return
        self._start_threaded_analysis()
    
    def _start_threaded_analysis(self):
        self.is_analyzing = True
        fname = Path(self.file_path).name
        self._busy(True, f"Analyzing {fname}…")
        self.set_locked(self.load_button, True, "Analyzing…")
        self.set_locked(self.analyze_button, True, "Analyzing…")
        self.set_locked(self.save_button, True, "Save\nResults…")
        self.set_locked(self.copy_button, True, "Copy to\nClipboard…")

        # НЕ затираем правое окно «Analyzing…» — оставляем прежний отчёт

        self._worker = AnalysisWorker(self.file_path)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.start()

    def _build_report_text(self, res: dict) -> str:
        lines = [
            "T2200 Form Analysis — Summary ",
            "=" * 40,
            f"File: {res.get('file', '—')}",
            f"Date: {res.get('date', '—')}",
            "",
            f"Eligibility (Q1): {res.get('eligibility_q1', '—')}",
            f"YES answers: {res.get('yes_total', 0)} | NO answers: {res.get('no_total', 0)}",
            "",
            "Dear customer,",
            ""
        ]
        elig = str(res.get("eligibility_q1", "")).upper()
        if elig == "YES":
            lines.append("Based on Form T2200, you may be eligible to claim:")
            if yes_texts := res.get("yes_texts", []):
                lines.extend([f"• {t}" for t in yes_texts])
            lines.append("\nWhen convenient, please forward the relevant expense documentation so I can finalize your claim.")
        elif elig == "NO":
            lines.append("Based on Form T2200, you are not eligible for employment expense claims.")
        else:
            lines.append("Eligibility could not be determined automatically.")
        return "\n".join(lines)

    def save_results(self):
        if self.report_output.count() == 0:
            error_item = QListWidgetItem("Error: No results to save")
            self.report_output.addItem(error_item)
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    # Save all items text
                    for i in range(self.report_output.count()):
                        item = self.report_output.item(i)
                        f.write(item.text() + "\n")
                self._flash_status("Results saved")
            except Exception as e:
                error_item = QListWidgetItem(f"\nError saving file: {str(e)}")
                self.report_output.addItem(error_item)

    def _copy_report(self):
        if not getattr(self, "_last_result", None):
            error_item = QListWidgetItem("Error: No results to copy (run Analyze first)")
            self.report_output.addItem(error_item)
            return
        try:
            client_text = self._build_client_text(self._last_result)
            QGuiApplication.clipboard().setText(client_text)
            self._flash_status("Copied to clipboard")
        except Exception as e:
            error_item = QListWidgetItem(f"\nCopy failed: {str(e)}")
            self.report_output.addItem(error_item)

    def _build_client_text(self, res: dict) -> str:
        elig = (res.get("eligibility_q1") or "—").upper()
        lines = ["Dear customer,", ""]

        if elig == "YES":
            lines.append("Based on Form T2200, you may be eligible to claim:")
            for p in self._collect_checked_points():
                lines.append(f"• {p}")
            lines.append("\nWhen convenient, please forward the relevant expense documentation so I can finalize your claim.")
        elif elig == "NO":
            lines.append("Based on Form T2200, you are not eligible for employment expense claims.")
        else:
            lines.append("Eligibility could not be determined automatically.")

        return "\n".join(lines)

    def clear_results(self):
        self.report_output.clear()
        self.file_path = None
        self._last_result = None
        self._queued_file = None
        self.filename_label.setText("No file loaded")
        self.set_locked(self.analyze_button, True, "Load a PDF first")
        self.set_locked(self.save_button, True, "Analyze the form first")
        self.set_locked(self.copy_button, True, "Analyze the form first")

        try:
            self.pdf_processor.cleanup()
        except Exception:
            pass

        self.report_output.setProperty("dragover", False)
        self.report_output.style().unpolish(self.report_output)
        self.report_output.style().polish(self.report_output)
        self.is_analyzing = False


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = T2200AnalyzerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()