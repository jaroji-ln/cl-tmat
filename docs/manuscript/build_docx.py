"""Build the submission files for Environmental Modelling & Software (Elsevier) from manuscript.md.

manuscript.docx: title page; highlights; abstract, keywords, and software and data availability;
main text with numbered sections and editable tables; declarations; appendix; references
(author-year, hanging indent); figure captions with each figure on its own page.
Double spacing, 12 pt Times New Roman, continuous line numbers, page numbers.
HTML source comments are removed and plain-text math becomes sub/superscripts.

highlights.docx: the highlights as a separate file, as Elsevier asks at submission.

Usage: python docs/manuscript/build_docx.py [--pandoc PATH]
Output: docs/manuscript/manuscript.docx, docs/manuscript/highlights.docx (not committed)
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
from docx.shared import Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "manuscript.md"
OUTPUT = HERE / "manuscript.docx"
HIGHLIGHTS = HERE / "highlights.docx"
PAGE_BREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n'
BREAK_BEFORE = ["## Highlights", "## Abstract", "## 1. Introduction", "## References", "## Figure captions"]

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
    text = re.sub(r"^---$", "", text, flags=re.M)
    for old, new in MATH:
        if old not in text:
            raise SystemExit(f"math pattern not found: {old}")
        text = text.replace(old, new)
    for marker, image in FIGURES.items():
        path = ROOT / "figures" / image
        text = text.replace(marker, f"{PAGE_BREAK}![]({path}){{width=16cm}}\n\n{marker}", 1)
    for heading in BREAK_BEFORE:
        if heading not in text:
            raise SystemExit(f"heading not found: {heading}")
        text = text.replace(heading, PAGE_BREAK + heading, 1)
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
        if style.name.startswith("Heading") or style.name == "Title":
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.font.bold = True
            style.font.size = Pt({"Title": 16, "Heading 1": 16, "Heading 2": 13, "Heading 3": 12}.get(style.name, 12))
            style.font.italic = style.name == "Heading 3"
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


def hanging_references(document: Document) -> None:
    inside = False
    for paragraph in document.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            inside = paragraph.text.strip() == "References"
            continue
        if inside and paragraph.text.strip():
            paragraph.paragraph_format.left_indent = Pt(18)
            paragraph.paragraph_format.first_line_indent = Pt(-18)


def highlights(text: str, pandoc: str) -> None:
    block = text.split("## Highlights", 1)[1].split("\n## ", 1)[0]
    bullets = [line for line in re.sub(r"<!--.*?-->", "", block, flags=re.S).splitlines() if line.startswith("- ")]
    for line in bullets:
        if len(line[2:]) > 85:
            raise SystemExit(f"highlight longer than 85 characters: {line}")
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "highlights.md"
        md.write_text("# Highlights\n\n" + "\n".join(bullets) + "\n", encoding="utf-8")
        subprocess.run([pandoc, str(md), "-o", str(HIGHLIGHTS)], check=True)


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
    hanging_references(document)
    line_numbers(document)
    document.save(OUTPUT)
    highlights(SOURCE.read_text(encoding="utf-8"), args.pandoc)
    print(f"wrote {OUTPUT} and {HIGHLIGHTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
