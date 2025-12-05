import fitz
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import os
import tempfile
import json
import logging
from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass
import webbrowser
from datetime import datetime
from pathlib import Path
from collections import deque
import sys

RECENTS_FILE = Path(tempfile.gettempdir()) / 't2200_recent_files.json'
MAX_RECENTS = 5
DEFAULT_DPI = 300
DEFAULT_PATCH_SIZE = 25
DEFAULT_THRESHOLD = 15.0
DEFAULT_WHITE_THRESHOLD = 245
DEFAULT_DETECTION_THRESHOLD = 3
OFFSET_X_YES_REF = 2
DX_NO_REF = 175
INK_THR = 185
MIN_PIX = 5
MARGIN = 4
MIN_SEP_NO_FROM_YES_REF = int(0.6 * DX_NO_REF)

# Configure logging to console only (no automatic file creation)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Create a handler that only logs to console, not to file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

@dataclass
class CheckboxConfig:
    pages: Dict[int, Dict[int, List[Tuple[int, int]]]]
    question_texts: Dict[int, str]
    ref_width: int = 2550
    ref_height: int = 3300
    detection_threshold: int = DEFAULT_DETECTION_THRESHOLD
    white_threshold: int = DEFAULT_WHITE_THRESHOLD

class PDFProcessor:
    """Handles PDF to image conversion and temporary file management"""

    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.temp_files = []
        self.current_doc = None  # Add this

    def convert_to_images(self, pdf_path: str, dpi: int=DEFAULT_DPI) -> List[str]:
        """Convert each page of PDF to images"""
        try:
            # Close any previously opened document first
            if self.current_doc:
                self.current_doc.close()
                self.current_doc = None
                
            self.current_doc = fitz.open(pdf_path)  # Store reference
            image_paths = []
            for page_num in range(len(self.current_doc)):
                image_path = os.path.join(self.temp_dir, f'page_{page_num + 1}_{os.path.basename(pdf_path)}.png')
                self._convert_page(self.current_doc, page_num, image_path, dpi)
                image_paths.append(image_path)
                self.temp_files.append(image_path)
            return image_paths
        except Exception as e:
            raise PDFProcessingError(f'Failed to process PDF: {str(e)}')

    def _convert_page(self, doc: fitz.Document, page_num: int, output_path: str, dpi: int):
        """Convert single PDF page to image"""
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        pix.save(output_path)

    def cleanup(self):
        # Close document if open
        if self.current_doc:
            self.current_doc.close()
            self.current_doc = None
            
        # Clean up temp files
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {file_path}: {e}")
        self.temp_files = []  # Reset the list

class EmptyYesComparator:
    """Compares checkbox areas with empty reference patches"""

    def __init__(self, patches_dir: str='empty_yes_patches', patch_size: int=DEFAULT_PATCH_SIZE, threshold: float=DEFAULT_THRESHOLD):
        self.dir = patches_dir
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.threshold = threshold
        self.available = os.path.isdir(self.dir)

    def has_patch(self, page_idx: int, qidx: int) -> bool:
        """Check if reference patch exists for given question"""
        if not self.available:
            return False
        path = os.path.join(self.dir, f'p{page_idx}', f'q{qidx}.png')
        return os.path.isfile(path)

    def _load_empty(self, page_idx: int, qidx: int) -> Optional[np.ndarray]:
        """Load reference patch image"""
        path = os.path.join(self.dir, f'p{page_idx}', f'q{qidx}.png')
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return img
        except Exception as e:
            return None

    def is_marked(self, tgt_gray: np.ndarray, cx: int, cy: int, page_idx: int, qidx: int) -> Optional[bool]:
        """Check if checkbox is marked using single point comparison"""
        empty = self._load_empty(page_idx, qidx)
        if empty is None:
            return None
        h, w = tgt_gray.shape[:2]
        y1, y2 = (max(0, cy - self.pad), min(h, cy + self.pad + 1))
        x1, x2 = (max(0, cx - self.pad), min(w, cx + self.pad + 1))
        patch = tgt_gray[y1:y2, x1:x2]
        if patch.shape != empty.shape:
            patch = cv2.resize(patch, (empty.shape[1], empty.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(patch, empty)
        return float(diff.mean()) > self.threshold

    def is_marked_with_search(self, tgt_gray: np.ndarray, cx: int, cy: int, page_idx: int, qidx: int, search: int=4, step: int=2) -> Tuple[Optional[bool], Optional[Tuple]]:
        """Check if checkbox is marked with neighborhood search"""
        empty = self._load_empty(page_idx, qidx)
        if empty is None:
            return (None, None)
        h, w = tgt_gray.shape[:2]
        best = (-1.0, cx, cy)
        for dy in range(-search, search + 1, step):
            for dx in range(-search, search + 1, step):
                x, y = (cx + dx, cy + dy)
                y1, y2 = (max(0, y - self.pad), min(h, y + self.pad + 1))
                x1, x2 = (max(0, x - self.pad), min(w, x + self.pad + 1))
                patch = tgt_gray[y1:y2, x1:x2]
                if patch.shape != empty.shape:
                    patch = cv2.resize(patch, (empty.shape[1], empty.shape[0]), interpolation=cv2.INTER_AREA)
                diff = cv2.absdiff(patch, empty)
                score = float(diff.mean())
                if score > best[0]:
                    best = (score, x, y)
        marked = best[0] > self.threshold
        return (marked, best)

class CheckboxAnalyzer:
    """Handles checkbox detection and analysis"""

    def __init__(self, config: CheckboxConfig):
        self.cfg = config
        self.empty_yes = None
        self.app_debug_mode = False
        self.yes_anchor = None  # Make this explicit

    def reset(self):
        """Reset any state that might persist between analyses"""
        # Clear any cached state if needed
        pass

    def _ink_ratio_debug(self, image: np.ndarray, checkbox_pixels: List[Tuple[int, int]]) -> List[Optional[float]]:
        """Debug function to calculate ink ratios for given points"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        H, W = bw.shape[:2]
        win = 5
        ratios = []
        for x, y in checkbox_pixels:
            if x < 0 or y < 0 or x >= W or (y >= H):
                ratios.append(None)
                continue
            x1, y1 = (max(0, x - win), max(0, y - win))
            x2, y2 = (min(W - 1, x + win), min(H - 1, y + win))
            patch = bw[y1:y2 + 1, x1:x2 + 1]
            total = patch.size
            dark = int((patch > 0).sum())
            ratios.append(dark / total if total else None)
        return ratios

    def _ink_in_box(self, gray: np.ndarray, cx: int, cy: int, box_half: int=8, thr: int=210) -> int:
        """
        Считает количество тёмных пикселей в окне чекбокса (≈ (2*box_half+1)^2).
        Используется, когда и YES, и NO выглядят «чернильными».
        """
        H, W = gray.shape[:2]
        x1 = max(0, cx - box_half)
        x2 = min(W, cx + box_half + 1)
        y1 = max(0, cy - box_half)
        y2 = min(H, cy + box_half + 1)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        return int((roi < thr).sum())

    def _save_overlay(self, image: np.ndarray, points: List[Tuple[int, int]], filename: str, color: Tuple[int, int, int]=(0, 0, 255)) -> Tuple[str, bool]:
        """Save debug image with marked points"""
        dbg = image.copy()
        points = points or []
        for x, y in points:
            cv2.circle(dbg, (x, y), 6, color, 2)
        out_path = os.path.join(tempfile.gettempdir(), filename)
        try:
            cv2.imwrite(out_path, dbg)
            return (out_path, True)
        except Exception:
            return (out_path, False)

    def _save_crops_and_diff(self, tgt_gray, empty, cx, cy, cx_no, cy_no, page_num, qidx):
        """
        Сохраняет:
          - yes_roi.png, no_roi.png
          - empty_ref.png
          - diff_yes.png, diff_no.png
        """
        outdir = Path(tempfile.gettempdir()) / 't2200_debug'
        outdir.mkdir(exist_ok=True)
        pad = empty.shape[0] // 2 if empty is not None else 12
        H, W = tgt_gray.shape[:2]

        def crop_at(x, y):
            y1, y2 = (max(0, y - pad), min(H, y + pad + 1))
            x1, x2 = (max(0, x - pad), min(W, x + pad + 1))
            return tgt_gray[y1:y2, x1:x2]
        yes_roi = crop_at(cx, cy)
        no_roi = crop_at(cx_no, cy_no)
        if empty is not None:
            cv2.imwrite(str(outdir / 'empty_ref.png'), empty)
            def save_diff(name, roi):
                if roi.shape != empty.shape:
                    roi = cv2.resize(roi, (empty.shape[1], empty.shape[0]), interpolation=cv2.INTER_AREA)
                diff = cv2.absdiff(roi, empty)
                cv2.imwrite(str(outdir / f'{name}.png'), diff)
            save_diff('diff_yes', yes_roi)
            save_diff('diff_no', no_roi)
        cv2.imwrite(str(outdir / 'yes_roi.png'), yes_roi)
        cv2.imwrite(str(outdir / 'no_roi.png'), no_roi)

    def analyze_images(self, image_paths: List[str]) -> Dict[int, str]:
        """Analyze checkboxes across all pages"""
        results: Dict[int, str] = {}
        for page_num, img_path in enumerate(image_paths, start=1):
            try:
                page_results = self.analyze_page(img_path, page_num)
                results.update(page_results)
            except Exception as e:
                logger.error(f"Error analyzing page {page_num}: {str(e)}")
                raise
        return results

    def analyze_page(self, image_path: str, page_num: int) -> Dict[int, str]:
        """Analyze one page with coordinate scaling (enhanced calibration)"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'Cannot open image: {image_path}')
        tgt_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        REF_W = self.cfg.ref_width
        REF_H = self.cfg.ref_height
        if (w, h) != (REF_W, REF_H):
            image = cv2.resize(image, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
            tgt_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]
        sx = w / float(REF_W)
        sy = h / float(REF_H)
        page_map = self.cfg.pages.get(page_num, {})
        results: Dict[int, str] = {}
        page_qidx = 0
        dx_no_cal: List[int] = []
        for number, coords in sorted(page_map.items(), key=lambda kv: int(kv[0])):
            try:
                qnum = int(number)
            except Exception:
                qnum = int(str(number).strip())
            page_qidx += 1
            if not coords or not isinstance(coords, (list, tuple)) or (not all((isinstance(p, (list, tuple)) and len(p) == 2 for p in coords))):
                results[qnum] = 'NO'
                continue
            decided = None
            scaled = [(int(round(x0 * sx)), int(round(y0 * sy))) for x0, y0 in coords]
            cx = int(round(sum((x for x, _ in scaled)) / len(scaled)))
            cy = int(round(sum((y for _, y in scaled)) / len(scaled)))
            dx_box_ref = 33
            dx_box = int(round(dx_box_ref * sx))
            if hasattr(self, 'yes_anchor') and getattr(self, 'yes_anchor', None) is not None:
                band = 110 if page_qidx <= 4 else 70
                hit = self.yes_anchor.find_near_row(tgt_gray, y_hint=cy, band=band)
                if hit:
                    x_yes, y_yes = hit
                    cx = x_yes - dx_box
                    cy = y_yes
                H, W = tgt_gray.shape[:2]
                y1, y2 = (max(0, cy - 1), min(H, cy + 2))
                x1, x2 = (max(0, cx - 1), min(W, cx + 2))
                sample = tgt_gray[y1:y2, x1:x2]
            dx_no = int(round(DX_NO_REF * sx))
            cx_no, cy_no = (cx + dx_no, cy)
            search_win = 6 if ((page_num == 1 and page_qidx >= 6) or (page_num == 2 and page_qidx >= 3)) else 4
            yes_marked = None
            best_yes = None
            if self.empty_yes is not None and self.empty_yes.has_patch(page_num, page_qidx):
                yes_marked, best_yes = self.empty_yes.is_marked_with_search(tgt_gray, cx, cy, page_idx=page_num, qidx=page_qidx, search=search_win, step=2)
                no_marked, best_no = self.empty_yes.is_marked_with_search(tgt_gray, cx_no, cy_no, page_idx=page_num, qidx=page_qidx, search=search_win, step=2)
                if page_qidx in (1, 2, 3, 4) and best_yes is not None and (best_no is not None):
                    try:
                        shift = int(best_no[1] - best_yes[1])
                        if shift != 0 and abs(shift) <= int(150 * sx):
                            dx_no_cal.append(shift)
                            dx_no = int(round(sum(dx_no_cal) / len(dx_no_cal)))
                            cx_no = cx + dx_no
                    except Exception:
                        pass
                if best_yes is not None:
                    cx = int(best_yes[1] + OFFSET_X_YES_REF * sx)
                    cy = int(best_yes[2])
                    cx_no = cx + dx_no
                    box_half = 7
                    ink_yes = self._ink_in_box(tgt_gray, cx, cy, box_half=box_half, thr=INK_THR)
                    ink_no = self._ink_in_box(tgt_gray, cx_no, cy_no, box_half=box_half, thr=INK_THR)
                    marked_yes = ink_yes >= MIN_PIX
                    marked_no = ink_no >= MIN_PIX
                    if marked_yes and (not marked_no):
                        decided = True
                    elif marked_no and (not marked_yes):
                        decided = False
                    elif not marked_yes and (not marked_no):
                        decided = False
                    elif ink_yes >= ink_no + MARGIN:
                        decided = True
                    elif ink_no >= ink_yes + MARGIN:
                        decided = False
                    else:
                        decided = False
                    results[qnum] = 'YES' if decided else 'NO'
                    if getattr(self, 'app_debug_mode', False):
                        try:
                            self._save_overlay(image, [(cx, cy), (cx_no, cy_no)], f'p{page_num}_q{page_qidx}_overlay.png')
                        except Exception:
                            pass
                    continue
                if best_yes is not None and best_no is not None:
                    score_yes, x_yes, y_yes = (float(best_yes[0]), best_yes[1], best_yes[2])
                    score_no, x_no, y_no = (float(best_no[0]), best_no[1], best_no[2])
                    T = self.empty_yes.threshold
                    if getattr(self, 'app_debug_mode', False):
                        empty_patch = self.empty_yes._load_empty(page_num, page_qidx)
                        self._save_crops_and_diff(tgt_gray, empty_patch, cx, cy, cx_no, cy_no, page_num, page_qidx)
                        self._save_overlay(image, [(cx, cy), (cx_no, cy_no)], f'p{page_num}_q{page_qidx}_overlay.png')
                    if score_yes >= T and score_no < T:
                        decided = True
                    elif score_no >= T and score_yes < T:
                        decided = False
                    elif score_yes < T and score_no < T:
                        decided = False
                    else:
                        box_half_dyn = 8 if page_num in (2, 3) else 6
                        ink_yes = self._ink_in_box(tgt_gray, cx, cy, box_half=box_half_dyn, thr=200)
                        ink_no = self._ink_in_box(tgt_gray, cx_no, cy_no, box_half=box_half_dyn, thr=200)
                        INK_THRESHOLD = 3
                        marked_yes = ink_yes >= INK_THRESHOLD
                        marked_no = ink_no >= INK_THRESHOLD
                        if not marked_yes and (not marked_no):
                            decided = None
            if decided is None:
                if self.empty_yes is not None and self.empty_yes.has_patch(page_num, page_qidx):
                    marked_one, best_one = self.empty_yes.is_marked_with_search(tgt_gray, cx, cy, page_idx=page_num, qidx=page_qidx, search=search_win, step=2)
                    if best_one is not None:
                        score_one = float(best_one[0])
                        if abs(score_one - self.empty_yes.threshold) <= 2.0:
                            decided = self._is_checked(image, scaled)
                        else:
                            decided = bool(marked_one)
                if decided is None:
                    decided = self._is_checked(image, scaled)
            results[qnum] = 'YES' if decided else 'NO'
        return results

    def _is_checked(self, image: np.ndarray, checkbox_pixels: List[Tuple[int, int]]) -> bool:
        """Determine if checkbox is checked using pixel analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        dark_count = 0
        total = 0
        win = 5
        H, W = bw.shape[:2]
        for x, y in checkbox_pixels:
            if x < 0 or y < 0 or x >= W or (y >= H):
                continue
            x1, y1 = (max(0, x - win), max(0, y - win))
            x2, y2 = (min(W - 1, x + win), min(H - 1, y + win))
            patch = bw[y1:y2 + 1, x1:x2 + 1]
            total += patch.size
            dark_count += int((patch > 0).sum())
        if total == 0:
            return False
        return dark_count / total >= 0.12

class T2200App(tk.Tk):
    """Main application GUI"""

    def __init__(self, config: CheckboxConfig):
        super().__init__()
        self.title('Enhanced T2200 Form Analyzer')
        self.cfg = config
        self.pdf_processor = PDFProcessor()
        self.yes_anchor = YesAnchor('yes_label.png', thr=0.75)
        self.debug_dir = Path(tempfile.gettempdir()) / 't2200_debug'
        self.debug_dir.mkdir(exist_ok=True)
        self.status_var = tk.StringVar()
        self.debug_mode = tk.BooleanVar(value=False)
        self.empty_yes = EmptyYesComparator()
        self.analyzer = CheckboxAnalyzer(config)
        self.analyzer.yes_anchor = self.yes_anchor
        self.analyzer.empty_yes = self.empty_yes
        self.current_results = {}
        self.recent_files = self._load_recent_files()
        self.last_analyzed_path = None
        self.geometry('900x700')
        self.minsize(800, 600)
        self._setup_ui()
        self._create_menu()
        self.analyzer.app_debug_mode = self.debug_mode.get()
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Segoe UI', 10))
        self.style.configure('TLabel', font=('Segoe UI', 10))
        self.style.configure('TFrame', background='#f0f0f0')
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def _create_menu(self):
        menubar = tk.Menu(self)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label='Help', command=self.show_help)
        helpmenu.add_command(label='About', command=self.show_about)
        menubar.add_cascade(label='Help', menu=helpmenu)
        self.config(menu=menubar)

    def _load_recent_files(self):
        try:
            if RECENTS_FILE.exists():
                return json.load(open(RECENTS_FILE, 'r', encoding='utf-8'))
        except Exception:
            pass
        return []

    def _push_recent(self, path):
        rec = [p for p in self.recent_files if p != path]
        rec.insert(0, path)
        del rec[MAX_RECENTS:]
        self.recent_files = rec
        try:
            json.dump(rec, open(RECENTS_FILE, 'w', encoding='utf-8'))
        except Exception:
            pass

    def upload_pdf(self):
        path = filedialog.askopenfilename(filetypes=[('PDF files', '*.pdf'), ('All files', '*.*')])
        if not path:
            return
        self._analyze_pdf_file(path)
        self.clear_btn['state'] = tk.NORMAL
        self.copy_btn['state'] = tk.NORMAL
        self.save_btn['state'] = tk.NORMAL

    def _setup_ui(self):
        """Configure all UI elements"""
        self.main_frame = ttk.Frame(self, padding='10')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        upload_frame = ttk.LabelFrame(self.main_frame, text='PDF Upload', padding='10')
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        self.upload_btn = ttk.Button(upload_frame, text='Upload T2200 PDF', command=self.upload_pdf, style='Accent.TButton')
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn = ttk.Button(upload_frame, text='Clear Results', command=self.clear_results, state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(upload_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.progress.pack_forget()
        results_frame = ttk.LabelFrame(self.main_frame, text='Analysis Results', padding='10')
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.text_area = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=('Segoe UI', 11), padx=10, pady=10)
        self.text_area.pack(fill=tk.BOTH, expand=True)
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        self.copy_btn = ttk.Button(button_frame, text='Copy to Clipboard', command=self.copy_to_clipboard, state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn = ttk.Button(button_frame, text='Save Results', command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.status_var.set('Ready')
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))

    def _analyze_pdf_file(self, file_path: str):
        """Analyze PDF file and generate results"""
        try:
            # Reset state before new analysis
            self.current_results = {}
            self.text_area.delete(1.0, tk.END)
            self._set_ui_state(False)
            
            self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            self.progress['value'] = 0
            self.update()
            
            # Clean up any previous temp files
            self.pdf_processor.cleanup()
            
            self.analyzer.app_debug_mode = self.debug_mode.get()
            self.status_var.set('Converting PDF to images...')
            
            image_paths = self.pdf_processor.convert_to_images(file_path)
            self.status_var.set('Analyzing checkboxes...')
            self.progress['maximum'] = len(image_paths)
            
            results = {}
            for page_num, img_path in enumerate(image_paths, start=1):
                page_results = self.analyzer.analyze_page(img_path, page_num)
                results.update(page_results)
                try:
                    self.progress['value'] = page_num
                    self.update()
                except Exception:
                    pass
            
            self.current_results = results
            self.last_analyzed_path = file_path
            self._push_recent(file_path)
            self._generate_report()
            self._set_ui_state(True)
            self.status_var.set(f'Analysis complete - {os.path.basename(file_path)}')
            
        except Exception as e:
            self.status_var.set('Error occurred - see console for details')
            logger.error(f"Error analyzing PDF: {str(e)}")
            # Ensure cleanup even on error
            self.pdf_processor.cleanup()
        finally:
            self.progress.pack_forget()

    def _generate_report(self):
        """Generate the analysis report based on results"""
        q1 = self.current_results.get(1, 'NO')
        if q1 != 'YES':
            result = 'Dear customer,\n\nBased on Form T2200, you are not eligible to claim employment-related expenses because your employment contract does not require you to pay for your own expenses.'
        else:
            yes_answers = [self.cfg.question_texts[num] for num, ans in sorted(self.current_results.items()) if ans == 'YES' and num != 1 and (num in self.cfg.question_texts)]
            if yes_answers:
                header = 'Dear customer,\n\nBased on Form T2200, you may be eligible to claim the following employment-related expenses:\n'
                body = '\n'.join((f'– {txt}' for txt in yes_answers))
                footer = '\n\nPlease consult with a tax professional to determine which expenses you can actually claim on your tax return.'
                result = f'{header}\n{body}{footer}'
            else:
                result = 'Dear customer,\n\nAlthough your employment contract requires you to pay your own expenses, no other eligible expense categories were marked YES on your T2200 form.\n\nYou may want to review the form with your employer to ensure all applicable boxes are properly marked.'
        
        yes_count = sum((1 for v in self.current_results.values() if v == 'YES'))
        no_count = sum((1 for v in self.current_results.values() if v == 'NO'))
        file_name = os.path.basename(self.last_analyzed_path) if self.last_analyzed_path else 'N/A'
        today_str = datetime.now().strftime('%Y-%m-%d')
        eligibility = 'YES' if q1 == 'YES' else 'NO'
        summary = f'T2200 Form Analysis — Summary\nFile: {file_name}\nDate: {today_str}\n\nEligibility (Q1): {eligibility}\nYES answers: {yes_count} | NO answers: {no_count}\n----------------------------------------\n\n'
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, summary + result)

    def _set_ui_state(self, analysis_complete: bool):
        """Enable/disable UI elements based on state"""
        state = tk.NORMAL if analysis_complete else tk.DISABLED
        self.copy_btn['state'] = state
        self.save_btn['state'] = state
        self.clear_btn['state'] = state

    def copy_to_clipboard(self):
        """Copy results to clipboard"""
        content = self.text_area.get('1.0', tk.END).strip()
        if not content:
            messagebox.showwarning("Empty Report", "Nothing to save, the report is empty.")
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.status_var.set('Results copied to clipboard')

    def save_results(self):
        """Save results to a text file (robust, no empty files)."""
        content = self.text_area.get('1.0', tk.END).strip()
        if not content:
            messagebox.showwarning("Empty Report", "Nothing to save, the report is empty.")
            return

        file_path = filedialog.asksaveasfilename(
            title='Save Results',
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )
        if not file_path:
            return  # user cancelled

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content + '\n')
            self.status_var.set(f'Results saved to {os.path.basename(file_path)}')
        except Exception as e:
            messagebox.showerror('Save Error', f'Failed to save file:\n{str(e)}')
            self.status_var.set('Error saving file')


    def clear_results(self):
        """Clear current results and reset state"""
        self.text_area.delete(1.0, tk.END)
        self.current_results = {}
        self.last_analyzed_path = None
        # Clean up any temporary files
        self.pdf_processor.cleanup()
        self._set_ui_state(False)
        self.status_var.set('Ready')

    def show_help(self):
        """Show help documentation"""
        help_text = "T2200 Form Analyzer Help\n\n1. Click 'Upload T2200 PDF' to select your form\n2. The application will analyze all checkboxes\n3. Review the generated report\n4. Use the buttons to copy or save results\n\nFor best results, use a high-quality scan of the form."
        messagebox.showinfo('Help', help_text)

    def show_about(self):
        """Show about dialog"""
        about_text = 'T2200 Form Analyzer\nVersion 2.0\n\nThis tool analyzes Canadian T2200 forms to determine\neligible employment-related expense claims.\n\nFor official tax advice, consult a qualified professional.'
        messagebox.showinfo('About', about_text)

    def on_close(self):
        """Handle application close"""
        try:
            self.pdf_processor.cleanup()
            # Force garbage collection if needed
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.destroy()


class YesAnchor:
    """Находит координаты слова 'Yes' на странице с помощью template matching."""

    def __init__(self, template_path='yes_label.png', method=cv2.TM_CCOEFF_NORMED, thr=0.75):
        self.method = method
        self.thr = thr
        self.tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.tpl is None:
            logger.warning(f"Could not load template image: {template_path}")
        else:
            self.tw, self.th = (self.tpl.shape[1], self.tpl.shape[0])

    def find_near_row(self, gray_page, y_hint, band=70):
        """
        Ищем 'Yes' в горизонтальной полосе вокруг ожидаемой строки.
        gray_page: страница (GRAY)
        y_hint: ожидаемая высота строки (из конфига/центра 3x3)
        band: полу-высота полосы для поиска
        Возврат: (x_yes, y_yes_center) или None
        """
        if self.tpl is None:
            return None
        H, W = gray_page.shape[:2]
        y1 = max(0, int(y_hint - band))
        y2 = min(H, int(y_hint + band))
        roi = gray_page[y1:y2, :]
        if roi.size == 0:
            return None
        res = cv2.matchTemplate(roi, self.tpl, self.method)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        score = maxVal
        if score < self.thr:
            return None
        x, y = maxLoc
        x_yes = x + self.tw // 2
        y_yes = y1 + y + self.th // 2
        return (x_yes, y_yes)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

class ImageLoadError(Exception):
    """Custom exception for image loading errors"""
    pass

def load_config(config_path: str='config.json') -> CheckboxConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError('config.json: root must be an object')
        if 'pages' not in data or 'question_texts' not in data:
            raise ValueError("config.json must contain 'pages' and 'question_texts' keys")
        pages = {int(k): v for k, v in data['pages'].items()}
        qtexts = {int(k): v for k, v in data['question_texts'].items()}
        if not isinstance(pages, dict) or not isinstance(qtexts, dict):
            raise ValueError("'pages' and 'question_texts' must be objects")
        ref = data.get('ref_size', {})
        return CheckboxConfig(pages=pages, question_texts=qtexts, ref_width=ref.get('width', 2550), ref_height=ref.get('height', 3300), detection_threshold=data.get('detection_threshold', DEFAULT_DETECTION_THRESHOLD), white_threshold=data.get('white_threshold', DEFAULT_WHITE_THRESHOLD))
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return get_default_config()

def get_default_config() -> CheckboxConfig:
    """Return default configuration if config file is missing"""
    return CheckboxConfig(pages={1: {1: [(2152, 1693), (2153, 1693), (2154, 1693), (2152, 1694), (2153, 1694), (2154, 1694), (2152, 1695), (2153, 1695), (2154, 1695)], 2: [(2152, 2205), (2153, 2205), (2154, 2205), (2152, 2206), (2153, 2206), (2154, 2206), (2152, 2207), (2153, 2207), (2154, 2207)], 3: [(2152, 2543), (2153, 2543), (2154, 2543), (2152, 2544), (2153, 2544), (2154, 2544), (2152, 2545), (2153, 2545), (2154, 2545)], 4: [(2152, 2618), (2153, 2618), (2154, 2618), (2152, 2619), (2153, 2619), (2154, 2619), (2152, 2620), (2153, 2620), (2154, 2620)], 5: [(2152, 2855), (2153, 2855), (2154, 2855), (2152, 2856), (2153, 2856), (2154, 2856), (2152, 2857), (2153, 2857), (2154, 2857)], 6: [(2152, 2917), (2153, 2917), (2154, 2917), (2152, 2918), (2153, 2918), (2154, 2918), (2152, 2919), (2153, 2919), (2154, 2919)], 7: [(2152, 2980), (2153, 2980), (2154, 2980), (2152, 2981), (2153, 2981), (2154, 2981), (2152, 2982), (2153, 2982), (2154, 2982)], 8: [(2152, 3042), (2153, 3042), (2154, 3042), (2152, 3043), (2153, 3043), (2154, 3043), (2152, 3044), (2153, 3044), (2154, 3044)]}, 2: {9: [(2152, 211), (2153, 211), (2154, 211), (2152, 212), (2153, 212), (2154, 212), (2152, 213), (2153, 213), (2154, 213)], 10: [(2152, 536), (2153, 536), (2154, 536), (2152, 537), (2153, 537), (2154, 537), (2152, 538), (2153, 538), (2154, 538)], 11: [(2152, 661), (2153, 661), (2154, 661), (2152, 662), (2153, 662), (2154, 662), (2152, 663), (2153, 663), (2154, 663)], 12: [(2152, 848), (2153, 848), (2154, 848), (2152, 849), (2153, 849), (2154, 849), (2152, 850), (2153, 850), (2154, 850)], 13: [(2077, 1073), (2078, 1073), (2079, 1073), (2077, 1074), (2078, 1074), (2079, 1074), (2077, 1075), (2078, 1075), (2079, 1075)], 14: [(2077, 1136), (2078, 1136), (2079, 1136), (2077, 1137), (2078, 1137), (2079, 1137), (2077, 1138), (2078, 1138), (2079, 1138)], 15: [(2077, 1198), (2078, 1198), (2079, 1198), (2077, 1199), (2078, 1199), (2079, 1199), (2077, 1200), (2078, 1200), (2079, 1200)], 16: [(2152, 1604), (2153, 1604), (2154, 1604), (2152, 1605), (2153, 1605), (2154, 1605), (2152, 1606), (2153, 1606), (2154, 1606)], 17: [(2152, 1879), (2153, 1879), (2154, 1879), (2152, 1880), (2153, 1880), (2154, 1880), (2152, 1881), (2153, 1881), (2154, 1881)], 18: [(2152, 2043), (2153, 2044), (2154, 2045), (2152, 2046), (2153, 2047), (2154, 2048), (2152, 2049), (2153, 2050), (2154, 2051)], 19: [(2152, 2393), (2153, 2393), (2154, 2393), (2152, 2394), (2153, 2394), (2154, 2394), (2152, 2395), (2153, 2395), (2154, 2395)], 20: [(2152, 2468), (2153, 2468), (2154, 2468), (2152, 2469), (2153, 2469), (2154, 2469), (2152, 2470), (2153, 2470), (2154, 2470)]}, 3: {21: [(2152, 304), (2153, 304), (2154, 304), (2152, 305), (2153, 305), (2154, 305), (2152, 306), (2153, 306), (2154, 306)], 22: [(2077, 514), (2078, 514), (2079, 514), (2077, 515), (2078, 515), (2079, 515), (2077, 516), (2078, 516), (2079, 516)], 23: [(2077, 576), (2078, 576), (2079, 576), (2077, 577), (2078, 577), (2079, 577), (2077, 578), (2078, 578), (2079, 578)], 24: [(2077, 643), (2078, 643), (2079, 643), (2077, 644), (2078, 644), (2079, 644), (2077, 645), (2078, 645), (2079, 645)], 25: [(2152, 801), (2153, 801), (2154, 801), (2152, 802), (2153, 802), (2154, 802), (2152, 803), (2153, 803), (2154, 803)], 26: [(2152, 1076), (2153, 1076), (2154, 1076), (2152, 1077), (2153, 1077), (2154, 1077), (2152, 1078), (2153, 1078), (2154, 1078)], 27: [(2152, 1193), (2153, 1193), (2154, 1193), (2152, 1194), (2153, 1194), (2154, 1194), (2152, 1195), (2153, 1195), (2154, 1195)], 28: [(2152, 1251), (2153, 1251), (2154, 1251), (2152, 1252), (2153, 1252), (2154, 1252), (2152, 1253), (2153, 1253), (2154, 1253)], 29: [(2152, 1418), (2153, 1418), (2154, 1418), (2152, 1419), (2153, 1419), (2154, 1419), (2152, 1420), (2153, 1420), (2154, 1420)], 30: [(2152, 1568), (2153, 1568), (2154, 1568), (2152, 1569), (2153, 1569), (2154, 1569), (2152, 1570), (2153, 1570), (2154, 1570)], 31: [(2152, 1676), (2153, 1676), (2154, 1676), (2152, 1677), (2153, 1677), (2154, 1677), (2152, 1678), (2153, 1678), (2154, 1678)], 32: [(2152, 1743), (2153, 1743), (2154, 1743), (2152, 1744), (2153, 1744), (2154, 1744), (2152, 1745), (2153, 1745), (2154, 1745)], 33: [(2152, 1901), (2153, 1901), (2154, 1901), (2152, 1902), (2153, 1902), (2154, 1902), (2152, 1903), (2153, 1903), (2154, 1903)], 34: [(2152, 2026), (2153, 2026), (2154, 2026), (2152, 2027), (2153, 2027), (2154, 2027), (2152, 2028), (2153, 2028), (2154, 2028)]}}, question_texts={1: 'Your employment contract required you to pay your own expenses.', 2: 'You were paid in part by commission.', 3: 'You had access to a commission income account for reimbursed expenses.', 4: 'Your commission income was included in box 14 of the T4.', 5: "You were required to rent an office away from your employer's place of business.", 6: 'You were required to hire a substitute or assistant.', 7: 'You were required to incur other specific work-related expenses not covered by other categories.', 8: 'You were required to pay for a cell phone for work purposes.', 9: 'You had to use part of your home for work.', 10: 'You worked from home more than 50% of the time for at least 4 consecutive weeks.', 11: 'You regularly used your home workspace for in-person meetings.', 12: 'Your employer reimbursed some of your work-related home and supply expenses.', 13: 'You were required to provide your own uniform, safety equipment, or protective clothing.', 14: 'You were required to maintain and clean your own uniform or safety gear.', 15: 'You were required to pay for mandatory work-related licenses, dues, or professional fees.', 16: "You regularly traveled to locations outside your employer's business site.", 17: 'You were required to work away for 12 or more consecutive hours.', 18: 'You received or were entitled to a vehicle allowance.', 19: 'You used a company vehicle for work.', 20: 'You covered some expenses for the company vehicle.', 21: 'You paid for reimbursed expenses as part of your job.', 22: 'You were required to purchase special materials or supplies necessary for your work.', 23: 'You were required to pay for training or certification courses for your position.', 24: 'You were required to purchase or maintain specialized equipment or tools for your work.', 25: 'You covered additional job-related expenses without reimbursement.', 26: 'You worked as a tradesperson.', 27: 'You were required to buy and provide tools used in your work.', 28: 'All your tools used in work meet the required conditions.', 29: 'You worked as an apprentice mechanic.', 30: 'You were registered in a mechanic certification program.', 31: 'You were required to purchase and use your own tools.', 32: 'All your listed tools meet the program requirements.', 33: 'You worked in forestry operations.', 34: 'You were required to provide your own power saw for the job.'})

if __name__ == '__main__':
    try:
        config = load_config()
        app = T2200App(config)
        app.mainloop()
    except Exception as e:
        messagebox.showerror('Fatal Error', f'The application encountered a critical error and cannot start:\n{str(e)}\n\nSee the console for details.')
        logger.critical(f"Application failed to start: {str(e)}")