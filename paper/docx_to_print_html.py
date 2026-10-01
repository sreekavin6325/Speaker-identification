from __future__ import annotations

import html
import re
from pathlib import Path

from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
DOCX_PATH = PAPER_DIR / "Multi_Model_Speaker_Recognition_IEEE_Paper.docx"
HTML_PATH = PAPER_DIR / "Multi_Model_Speaker_Recognition_IEEE_Paper.html"
ASSET_DIR = PAPER_DIR / "html_assets"


def iter_blocks(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_images(document: Document, paragraph: Paragraph, counter: list[int]) -> list[str]:
    paths: list[str] = []
    for run in paragraph.runs:
        for blip in run._r.xpath(".//a:blip"):
            rel_id = blip.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
            )
            if not rel_id:
                continue
            part = document.part.related_parts[rel_id]
            suffix = Path(part.partname).suffix or ".png"
            counter[0] += 1
            filename = f"figure_{counter[0]:02d}{suffix}"
            path = ASSET_DIR / filename
            path.write_bytes(part.blob)
            paths.append(filename)
    return paths


def paragraph_html(document: Document, paragraph: Paragraph, image_counter: list[int], index: int) -> str:
    images = extract_images(document, paragraph, image_counter)
    if images:
        return "".join(
            f'<div class="figure"><img src="html_assets/{html.escape(name)}" alt="Research figure"></div>'
            for name in images
        )

    text = paragraph.text.strip()
    if not text:
        return '<div class="spacer"></div>'

    safe = html.escape(text).replace("\n", "<br>")
    style_name = paragraph.style.name if paragraph.style is not None else ""

    if index == 0:
        return f'<h1 class="paper-title">{safe}</h1>'
    if index == 1:
        return f'<p class="author">{safe}</p>'
    if index == 2:
        return f'<p class="affiliation">{safe}</p>'
    if text.startswith("Abstract—"):
        return f'<p class="abstract">{safe}</p>'
    if text.startswith("Index Terms—"):
        return f'<p class="keywords">{safe}</p>'
    if re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s", text):
        return f'<h2>{safe}</h2>'
    if text == "REFERENCES" or text == "References":
        return f'<h2>{safe}</h2>'
    if re.match(r"^[A-Z]\.\s", text):
        return f'<h3>{safe}</h3>'
    if text.startswith("TABLE "):
        return f'<p class="table-caption">{safe}</p>'
    if text.startswith("Fig. "):
        return f'<p class="figure-caption">{safe}</p>'
    if text.startswith("[") and re.match(r"^\[\d+\]", text):
        return f'<p class="reference">{safe}</p>'
    if text.startswith("Accuracy =") or text.startswith("SNR(dB)"):
        return f'<p class="equation">{safe}</p>'
    if style_name.startswith("List Bullet"):
        return f'<p class="bullet">{safe}</p>'
    return f'<p class="body">{safe}</p>'


def table_html(table: Table) -> str:
    rows = []
    for row_index, row in enumerate(table.rows):
        cells = []
        tag = "th" if row_index == 0 else "td"
        for cell in row.cells:
            text = html.escape(cell.text.strip()).replace("\n", "<br>")
            cells.append(f"<{tag}>{text}</{tag}>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    extra = " pipeline-table" if len(table.columns) >= 10 else ""
    return f'<table class="data-table{extra}"><tbody>{"".join(rows)}</tbody></table>'


def build_html() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    document = Document(DOCX_PATH)
    output = []
    paragraph_index = 0
    image_counter = [0]
    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            output.append(paragraph_html(document, block, image_counter, paragraph_index))
            if block.text.strip() or image_counter[0] > 0:
                paragraph_index += 1
        else:
            output.append(table_html(block))

    css = r"""
@page { size: Letter; margin: 0.62in 0.66in 0.58in 0.66in; }
* { box-sizing: border-box; }
html { background: white; }
body {
  margin: 0 auto;
  max-width: 7.18in;
  background: white;
  color: #111;
  font-family: "Times New Roman", Times, serif;
  font-size: 10pt;
  line-height: 1.12;
}
.paper-title {
  margin: 0 0 7pt;
  text-align: center;
  font-size: 18pt;
  line-height: 1.06;
  font-weight: 700;
}
.author { margin: 0 0 2pt; text-align: center; font-size: 11pt; text-indent: 0; }
.affiliation { margin: 0 0 8pt; text-align: center; font-size: 9pt; font-style: italic; color: #333; text-indent: 0; }
.abstract, .keywords { margin: 0.04in 0.28in 5pt; text-align: justify; font-size: 9pt; text-indent: 0; }
h2 {
  margin: 9pt 0 4pt;
  text-align: center;
  font-size: 11.5pt;
  line-height: 1.0;
  text-transform: uppercase;
  break-after: avoid-page;
}
h3 {
  margin: 6pt 0 2pt;
  text-align: left;
  font-size: 10.5pt;
  line-height: 1.0;
  break-after: avoid-page;
}
p.body { margin: 0 0 3pt; text-align: justify; text-indent: 0.17in; orphans: 3; widows: 3; }
p.bullet { margin: 0 0 2pt 0.22in; text-align: justify; text-indent: -0.14in; }
p.bullet::before { content: "• "; }
.equation { margin: 5pt 0; text-align: center; font-family: Cambria, "Times New Roman", serif; font-style: italic; }
.table-caption { margin: 6pt 0 3pt; text-align: center; font-size: 8pt; line-height: 1.05; font-weight: 600; break-after: avoid-page; }
.figure { margin: 7pt auto 1pt; text-align: center; break-inside: avoid-page; }
.figure img { max-width: 100%; max-height: 7.1in; object-fit: contain; }
.figure-caption { margin: 2pt 0 6pt; text-align: center; font-size: 8pt; break-before: avoid-page; }
.reference { margin: 0 0 0.8pt 0.19in; text-align: justify; text-indent: -0.19in; font-size: 7.6pt; line-height: 1.0; }
.spacer { height: 2pt; }
.data-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: auto;
  margin: 0 auto 5pt;
  font-size: 7.6pt;
  line-height: 1.08;
  break-inside: avoid-page;
}
.data-table th { background: #1f4e79; color: white; border: 0.5pt solid #fff; padding: 3.5pt; text-align: center; }
.data-table td { border: 0.5pt solid #b8bec4; padding: 3.2pt; vertical-align: middle; }
.data-table tr:nth-child(even) td { background: #f3f6f9; }
.pipeline-table { margin-top: 6pt; font-size: 7pt; }
.pipeline-table th { background: #d9eaf7; color: #1f4e79; border: 0.6pt solid #1f4e79; }
.pipeline-table th:nth-child(even), .pipeline-table td:nth-child(even) { border: none; background: white; color: #1f4e79; font-size: 13pt; }
"""

    content = "\n".join(output)
    final = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Multi-Model Speaker Recognition IEEE Paper</title>
  <style>{css}</style>
</head>
<body>
{content}
</body>
</html>
"""
    HTML_PATH.write_text(final, encoding="utf-8")
    print(HTML_PATH)


if __name__ == "__main__":
    build_html()
