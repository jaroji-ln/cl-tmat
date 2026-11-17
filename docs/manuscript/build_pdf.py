"""Build a camera-ready style PDF of the manuscript with Elsevier's elsarticle class.

Layout: elsarticle "final,5p" (two columns, close to the printed Elsevier journal page), STIX Two
Text with a DejaVu fallback for symbols, no line numbers, single spacing. Tables and figures are
floats placed near their first mention; tables are built from the Markdown pipe tables, figures
use the PDF files in figures/. Highlights are omitted (they are not printed in the journal; see
highlights.docx). This is a preview of the published layout: Elsevier typesets accepted papers
itself, and submission uses manuscript.docx.

Usage: python docs/manuscript/build_pdf.py [--pandoc PATH]
Needs lualatex (TeX Live) and pandoc.
Output: docs/manuscript/manuscript_camera_ready.pdf (not committed)
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "manuscript.md"
OUTPUT = HERE / "manuscript_camera_ready.pdf"
TEXBIN = "/Library/TeX/texbin"

MATH = [
    ("MAE_model / MAE_persistence", "MAE~model~ / MAE~persistence~"),
    ("MAE/MAE_persistence", "MAE/MAE~persistence~"),
    ("λ/2 Σ F_i(θ_i − θ*_i)²", "(λ/2) Σ~i~ F~i~ (θ~i~ − θ\\*~i~)²"),
    ("Σ_{k=1}^{h} φ^k", "Σ φ^k^ (k = 1, …, *h*)"),
]
UNNUMBERED = ["Software and data availability", "CRediT authorship contribution statement", "Declaration of competing interest",
              "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", "Acknowledgements"]
# figure number -> (file, float environment, width)
FIGURES = {
    1: ("fig1_tmat_regimes_drift.pdf", "figure*", "height=0.82\\textheight"),
    2: ("fig2_skill_horizon.pdf", "figure*", "width=\\textwidth"),
    3: ("fig3_cl_t4.pdf", "figure*", "width=0.8\\textwidth"),
    4: ("fig4_diagnostics_t4.pdf", "figure", "width=\\columnwidth"),
    5: ("fig5_crossing_f1.pdf", "figure", "width=\\columnwidth"),
}

PREAMBLE = r"""\documentclass[final,5p,twocolumn]{elsarticle}
\usepackage{fontspec}
\directlua{luaotfload.add_fallback("symbols", {"DejaVuSans:mode=harf;", "DejaVuSansMono:mode=harf;"})}
\setmainfont{STIX Two Text}[RawFeature={fallback=symbols}]
\setmonofont{DejaVu Sans Mono}[Scale=0.8,RawFeature={fallback=symbols}]
\usepackage{graphicx,booktabs,adjustbox,amsmath}
\usepackage{xurl}
\usepackage[colorlinks=true,allcolors=blue!50!black]{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\emergencystretch}{3em}
\sloppy
\journal{Environmental Modelling \& Software}
\begin{document}
"""


def pandoc_latex(md: str, pandoc: str) -> str:
    result = subprocess.run([pandoc, "-f", "markdown+subscript+superscript-implicit_figures", "-t", "latex", "--wrap=none"],
                            input=md, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def inline(md: str, pandoc: str) -> str:
    latex = pandoc_latex(md, pandoc)
    return latex.removeprefix("\\par").strip()


def cell(text: str) -> str:
    text = text.strip().replace("\\*", "*")
    text = re.sub(r"[&%#$_]", lambda m: "\\" + m.group(0), text)
    return re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text)


def table_latex(number: int, caption: str, rows: list[str], note: str, pandoc: str, appendix: bool) -> str:
    split = [[c for c in r.strip().strip("|").split("|")] for r in rows]
    header, align, body = split[0], split[1], split[2:]
    spec = "".join("r" if a.strip().endswith(":") and not a.strip().startswith(":") else "l" for a in align)
    lines = [" & ".join(f"\\textbf{{{cell(h)}}}" for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(c) for c in r) + r" \\" for r in body]
    label = f"A{number}" if appendix else str(number)
    note_tex = f"\n\\par\\smallskip{{\\scriptsize {inline(note, pandoc)}}}" if note else ""
    return (f"\\begin{{table*}}[!tbp]\n\\renewcommand{{\\thetable}}{{{label}}}\\addtocounter{{table}}{{-1}}\\refstepcounter{{table}}\n"
            f"\\caption{{{inline(caption, pandoc)}}}\n\\centering\\footnotesize\n\\begin{{adjustbox}}{{max width=\\textwidth}}\n"
            f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n\\toprule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}"
            + note_tex + "\n\\end{table*}")


def figure_latex(number: int, caption: str, pandoc: str) -> str:
    file, env, size = FIGURES[number]
    path = ROOT / "figures" / file
    return (f"\\begin{{{env}}}[!tbp]\n\\centering\n\\includegraphics[{size},keepaspectratio]{{{path}}}\n"
            f"\\renewcommand{{\\thefigure}}{{{number}}}\\caption{{{inline(caption, pandoc)}}}\n\\end{{{env}}}")


def raw(latex: str) -> str:
    return f"\n```{{=latex}}\n{latex}\n```\n"


def build(pandoc: str) -> str:
    text = re.sub(r"<!--.*?-->", "", SOURCE.read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(r"^---$", "", text, flags=re.M)
    for old, new in MATH:
        text = text.replace(old, new)

    title = re.search(r"^# (.+)$", text, re.M).group(1)
    abstract = text.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0].strip()
    keywords = [k.strip() for k in re.search(r"\*\*Keywords:\*\* (.+)", text).group(1).split(";")]
    body = text[text.index("## Software and data availability"):text.index("## Figure captions")]
    captions = dict((int(n), c.strip()) for n, c in re.findall(r"\*\*Fig\. (\d)\.\*\* (.+)", text))

    # tables: caption paragraph, pipe table, optional italic note
    def replace_table(match: re.Match) -> str:
        number, caption, table, note = match.group(1), match.group(2), match.group(3), match.group(4) or ""
        appendix = number.startswith("A")
        rows = [r for r in table.strip().splitlines() if r.startswith("|")]
        return raw(table_latex(int(number.lstrip("A")), caption.strip(), rows, note.strip().strip("*"), pandoc, appendix))
    body = re.sub(r"\*\*Table (A?\d)\.\*\* (.+?)\n\s*\n((?:\|.*\n)+)(?:\s*\n(\*[^*\n].*\*)\n)?", replace_table, body)

    # figures: after the first paragraph that cites each one
    paragraphs = body.split("\n\n")
    for number, caption in sorted(captions.items()):
        pattern = re.compile(rf"Fig\. {number}(?![0-9])")
        index = next(i for i, p in enumerate(paragraphs) if pattern.search(p) and not p.lstrip().startswith("#"))
        paragraphs[index] += "\n" + raw(figure_latex(number, caption, pandoc))
    body = "\n\n".join(paragraphs)

    # headings: LaTeX numbers sections; declarations unnumbered; appendix via \appendix
    body = re.sub(r"^### \d+\.\d+ (.+)$", r"## \1", body, flags=re.M)
    body = re.sub(r"^## \d+\. (.+)$", r"# \1", body, flags=re.M)
    for heading in UNNUMBERED:
        body = body.replace(f"## {heading}\n", f"# {heading} {{.unnumbered}}\n")
    body = re.sub(r"^## Appendix A\. (.+)$", lambda m: raw("\\appendix\\setcounter{table}{0}") + f"\n# {m.group(1)}", body, flags=re.M)

    references = body.split("## References", 1)
    main_md, refs_md = references[0], references[1]
    refs_tex = pandoc_latex(refs_md, pandoc)

    author_block = (r"\author[a]{[TODO First Author]\corref{cor1}}" "\n" r"\ead{[TODO e-mail]}" "\n"
                    r"\author[b]{[TODO Second Author]}" "\n" r"\cortext[cor1]{Corresponding author.}" "\n"
                    r"\affiliation[a]{organization={[TODO Affiliation]},country={[TODO]}}" "\n"
                    r"\affiliation[b]{organization={[TODO Affiliation]},country={[TODO]}}")
    front = (PREAMBLE + "\\begin{frontmatter}\n" + f"\\title{{{inline(title, pandoc)}}}\n" + author_block + "\n"
             + f"\\begin{{abstract}}\n{inline(abstract, pandoc)}\n\\end{{abstract}}\n"
             + "\\begin{keyword}\n" + " \\sep ".join(inline(k, pandoc) for k in keywords) + "\n\\end{keyword}\n"
             + "\\end{frontmatter}\n")
    refs_block = ("\\section*{References}\n{\\footnotesize\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{2pt}"
                  "\\everypar{\\hangindent=1em\\hangafter=1}\n" + refs_tex + "\n\\par}\n")
    return front + pandoc_latex(main_md, pandoc) + "\n" + refs_block + "\\end{document}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pandoc", default="pandoc")
    args = parser.parse_args()
    tex = build(args.pandoc)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "manuscript.tex"
        source.write_text(tex, encoding="utf-8")
        env = {"PATH": f"{TEXBIN}:/usr/bin:/bin"}
        for _ in range(2):
            run = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", source.name], cwd=tmp, env=env,
                                 capture_output=True, text=True)
            if run.returncode != 0:
                shutil.copy(source, HERE / "manuscript_camera_ready.tex")
                print(run.stdout[-3000:])
                raise SystemExit("lualatex failed; source kept in docs/manuscript/manuscript_camera_ready.tex")
        shutil.copy(Path(tmp) / "manuscript.pdf", OUTPUT)
        log = (Path(tmp) / "manuscript.log").read_text(errors="ignore")
    missing = sorted(set(re.findall(r"Missing character: There is no (.+?) \(", log)))
    print(f"wrote {OUTPUT}" + (f"; missing glyphs: {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
