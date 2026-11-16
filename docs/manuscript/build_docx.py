"""Build a review-ready Word file from manuscript.md.

Removes HTML source comments, converts plain-text math to pandoc sub/superscripts, embeds each
figure above its caption, then sets double spacing, 12 pt Times New Roman, continuous line
numbers, and page numbers (common requirements for review copies of Elsevier journals).

Usage: python docs/manuscript/build_docx.py [--pandoc PATH]
Output: docs/manuscript/manuscript.docx (not committed)
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "manuscript.md"
OUTPUT = HERE / "manuscript.docx"

MATH = [
    ("MAE_model / MAE_persistence", "MAE~model~ / MAE~persistence~"),
    ("MAE/MAE_persistence", "MAE/MAE~persistence~"),
    ("λ/2 Σ F_i(θ_i − θ*_i)²", "(λ/2) Σ~i~ F~i~ (θ~i~ − θ\\*~i~)²"),
    ("Σ_{k=1}^{h} φ^k", "Σ φ^k^ (k = 1, …, *h*)"),
]
FIGURES = {
    "**Fig. 1.**": "fig1_tmat_regimes_drift.png",
    "**Fig. 2.**": "fig2_skill_horizon.png",
    "**Fig. 3.**": "fig3_cl_t4.png",
    "**Fig. 4.**": "fig4_diagnostics_t4.png",
    "**Fig. 5.**": "fig5_crossing_f1.png",
}


def prepare(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    for old, new in MATH:
        if old not in text:
            raise SystemExit(f"math pattern not found: {old}")
        text = text.replace(old, new)
    for marker, image in FIGURES.items():
        path = ROOT / "figures" / image
        text = text.replace(marker, f"![]({path}){{width=16cm}}\n\n{marker}", 1)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def line_numbers(document: Document) -> None:
    for section in document.sections:
        numbering = OxmlElement("w:lnNumType")
        numbering.set(qn("w:countBy"), "1")
        numbering.set(qn("w:restart"), "continuous")
        section._sectPr.append(numbering)
        footer = section.footer.paragraphs[0]
        run = footer.add_run()
        for kind, value in (("begin", None), ("instr", " PAGE "), ("end", None)):
            if kind == "instr":
                element = OxmlElement("w:instrText")
                element.set(qn("xml:space"), "preserve")
                element.text = value
            else:
                element = OxmlElement("w:fldChar")
                element.set(qn("w:fldCharType"), kind)
            run._r.append(element)


def styles(document: Document) -> None:
    for style in document.styles:
        if style.type != 1:  # paragraph styles only
            continue
        style.font.name = "Times New Roman"
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman") if style.element.rPr is not None and style.element.rPr.rFonts is not None else None
        if style.name in ("Normal", "Body Text", "First Paragraph", "Compact"):
            style.font.size = Pt(12)
            style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    for run in paragraph.runs:
                        run.font.size = Pt(9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pandoc", default="pandoc")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "manuscript.md"
        md.write_text(prepare(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
        subprocess.run([args.pandoc, str(md), "-f", "markdown+subscript+superscript+pipe_tables", "-o", str(OUTPUT)], check=True)
    document = Document(OUTPUT)
    styles(document)
    line_numbers(document)
    document.save(OUTPUT)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
