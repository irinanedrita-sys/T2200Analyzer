# T2200Analyzer

Automated analysis tool for CRA Form T2200 using Python and PDF image detection.

## What this tool does

T2200Analyzer reads a scanned or PDF version of CRA Form **T2200** and:
- detects the **YES / NO** checkboxes using image processing;
- builds a **summary of all YES answers** by question;
- generates **client-friendly text** that can be pasted into an email;
- helps accountants quickly see which employment expenses may be claimable.

> **Disclaimer:** This tool does **not** replace professional judgement or CRA guidance.  
> Always review the output before using it in client work or tax filings.

## Project structure

- `ui_t2200.py` – main PyQt6 GUI for the T2200 analyzer (drag & drop, progress, checklist).
- `t2200_analyzer_final.py` – core analysis logic (PDF → images, checkbox detection, report).
- `splash.py` – optional splash-screen for the GUI.
- `yes_label.png` – template image used to locate the "Yes" column on the form.
- `resources/empty_yes_patches/` – reference patches of **empty** YES boxes for comparison.
- `icons/` – application icons (`t2200.ico`, UI icons, etc.).
- `requirements.txt` – Python dependencies.

## Installation

Requirements:
- Python 3.10+ (tested on Windows)
- Ability to install packages with `pip`
- 
## 📸 Screenshots

### Main Window
[![Main Window](screenshots/screenshot1.png)
](https://github.com/irinanedrita-sys/T2200Analyzer/blob/main/screenshots/2025_12_08_13_07_56_T2200_Form_Analyzer.jpg?raw=true)
### Example Analysis Output
![Output](screenshots/screenshot2.png)

Clone or download this repository:

```bash
git clone https://github.com/irinanedrita-sys/T2200Analyzer.git
cd T2200Analyzer
Then install dependencies:

pip install -r requirements.txt

Usage
Option 1 – PyQt6 GUI (recommended)

Run:

python ui_t2200.py


Then:

Click “Load File” and choose a T2200 PDF, or drag & drop a PDF onto the right panel.

Click “Analyze”.

Review the checklist of detected answers and the summary text.

Use “Copy to Clipboard” or “Save Results” to reuse the report.

Option 2 – Tkinter GUI (legacy)

The file t2200_analyzer_final.py contains the original Tkinter-based interface.
It is kept here for reference and testing.

Limitations

Works best with high-quality scans at 300 DPI using the standard CRA T2200 layout.

Non-standard forms or very low-quality scans may require manual review.

This project is an experimental automation tool and is not affiliated with the CRA.

Roadmap / ideas

Export results as structured JSON/CSV.

Batch-processing of multiple T2200 forms at once.

Additional debug overlays for checkbox detection.
