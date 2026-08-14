"""Render WRITEUP.md to a compact 2-page PDF.

    pip install reportlab
    python scripts/build_writeup_pdf.py            # -> WRITEUP.pdf
    python scripts/build_writeup_pdf.py in.md out.pdf

Handles the subset of Markdown the write-up uses: headings, bullets, **bold**, *italic*, and
`code`. Arabic runs are transliterated, because reportlab's built-in Type1 fonts carry no Arabic
glyphs and would draw solid boxes instead.

reportlab is a documentation-time dependency only; it is deliberately not in requirements.txt,
since nothing the assessment grades depends on regenerating this PDF.
"""
import pathlib
import re
import sys

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = sys.argv[1] if len(sys.argv) > 1 else REPO_ROOT / "WRITEUP.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else REPO_ROOT / "WRITEUP.pdf"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=8.4,
                      leading=11.1, alignment=TA_JUSTIFY, spaceAfter=4.5)
H1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=14,
                    leading=16, spaceAfter=6, alignment=0)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=9.8,
                    leading=12, spaceBefore=8, spaceAfter=3.5, textColor="#1a4d3a")
BULLET = ParagraphStyle("bullet", parent=BODY, spaceAfter=2.5)

# Arabic words the write-up cites; the built-in Type1 fonts cannot draw them.
ARABIC = {"حد": "hadd", "واحد": "wahed"}


def inline(t: str) -> str:
    for ar, latin in ARABIC.items():
        t = t.replace(ar, latin)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="7.6">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    return t


story, pending, bullets = [], [], []


def flush_para():
    if pending:
        story.append(Paragraph(inline(" ".join(pending)), BODY))
        pending.clear()


def flush_bullets():
    if bullets:
        story.append(ListFlowable(
            [ListItem(Paragraph(inline(b), BULLET), leftIndent=9) for b in bullets],
            bulletType="bullet", bulletFontSize=6, leftIndent=11, spaceAfter=4))
        bullets.clear()


for raw in open(SRC, encoding="utf-8").read().split("\n"):
    line = raw.rstrip()
    if line.startswith("# "):
        flush_para(); flush_bullets()
        story += [Paragraph(inline(line[2:]), H1), Spacer(1, 1)]
    elif line.startswith("## "):
        flush_para(); flush_bullets()
        story.append(Paragraph(inline(line[3:]), H2))
    elif line.startswith("- "):
        flush_para()
        bullets.append(line[2:])
    elif not line.strip():
        flush_para(); flush_bullets()
    elif bullets and raw.startswith("  "):
        bullets[-1] += " " + line.strip()
    else:
        pending.append(line.strip())

flush_para(); flush_bullets()

SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=13 * mm, bottomMargin=13 * mm,
                  leftMargin=16 * mm, rightMargin=16 * mm,
                  title="Careem Care Agent - Architectural Write-Up",
                  author="Otakan").build(story)
# The repo path may contain characters the console codepage cannot encode, so report the
# basename and never let a print() failure mask a successful build.
print(f"wrote {pathlib.Path(OUT).name} ({pathlib.Path(OUT).stat().st_size} bytes)")
