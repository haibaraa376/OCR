from docx import Document
from docx.enum.table import WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _page_size(section, width, height):
    # 按纸页比例选择横竖版。
    section.page_width, section.page_height = (Cm(29.7), Cm(21)) if width > height else (Cm(21), Cm(29.7))
    section.top_margin = section.bottom_margin = Cm(.8)
    section.left_margin = section.right_margin = Cm(.8)


def _add_table(doc, data, page_width, page_height):
    xs, ys = data["columns"], data["rows"]
    rows, cols = len(ys) - 1, len(xs) - 1
    if rows < 1 or cols < 1:
        return
    section = doc.sections[-1]
    scale = min((section.page_width - section.left_margin - section.right_margin) / page_width,
                (section.page_height - section.top_margin - section.bottom_margin) / page_height)
    table = doc.add_table(rows=rows, cols=cols)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for c in range(cols):
        table.columns[c].width = int((xs[c + 1] - xs[c]) * scale)
    for r in range(rows):
        table.rows[r].height = int((ys[r + 1] - ys[r]) * scale)
        table.rows[r].height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        for c in range(cols):
            table.cell(r, c).width = table.columns[c].width
    for item in data["cells"]:
        r, c = item["row"], item["col"]
        cell = table.cell(r, c)
        if item["rowspan"] > 1 or item["colspan"] > 1:
            cell = cell.merge(table.cell(r + item["rowspan"] - 1, c + item["colspan"] - 1))
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.text = item.get("text", "")
        if item.get("text") and xs[c + item["colspan"]] - xs[c] < (ys[r + item["rowspan"]] - ys[r]) * .22:
            direction = OxmlElement("w:textDirection")
            direction.set(qn("w:val"), "tbRl")
            cell._tc.get_or_add_tcPr().append(direction)
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.space_after = Pt(0)
            for run in paragraph.runs:
                run.font.size = Pt(7)


def export_word(pages, destination):
    """每张纸页独立分页，表格和文字均可编辑。"""
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(9)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for index, page in enumerate(pages):
        if index:
            doc.add_page_break()
        _page_size(doc.sections[-1], page["width"], page["height"])
        elements = [(table["bbox"][1], "table", table) for table in page["tables"]]
        elements += [(block["block_bbox"][1], "text", block) for block in page.get("blocks", [])
                     if block.get("block_label") in ("text", "title", "paragraph_title")]
        for _, kind, item in sorted(elements, key=lambda entry: entry[0]):
            if kind == "table":
                _add_table(doc, item, page["width"], page["height"])
            elif item.get("block_content"):
                doc.add_paragraph(item["block_content"])
    doc.save(destination)
