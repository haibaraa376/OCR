"""将空白 PDF 页面图片作背景，叠加可编辑的定位文字框。"""
from pathlib import Path
from docx import Document
from docx.oxml import parse_xml
from docx.shared import Inches
from docx.enum.text import WD_BREAK
from xml.sax.saxutils import escape


def _shape(paragraph, text, x, y, width, height, font_pt, color, image_rel=None):
    # VML 绝对定位框：文字在 w:txbxContent 中
    style = (f"position:absolute;margin-left:{x:.3f}pt;margin-top:{y:.3f}pt;"
             f"width:{width:.3f}pt;height:{height:.3f}pt;z-index:{-1 if image_rel else 2};"
             "mso-position-horizontal-relative:page;mso-position-vertical-relative:page")
    if image_rel:
        body = f'<v:imagedata r:id="{image_rel}"/>'
        attrs = 'filled="t" stroked="f"'
    else:
        body = (f'<v:textbox inset="0,0,0,0" style="mso-fit-shape-to-text:f">'
                f'<w:txbxContent><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="{max(120, int(font_pt*20))}" w:lineRule="exact"/></w:pPr>'
                f'<w:r><w:rPr><w:rFonts w:eastAsia="Microsoft YaHei"/><w:color w:val="{color}"/>'
                f'<w:sz w:val="{max(8, int(font_pt*2))}"/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
                '</w:p></w:txbxContent></v:textbox>')
        attrs = 'filled="f" stroked="f"'
    xml = (f'<w:pict xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           f'xmlns:v="urn:schemas-microsoft-com:vml" '
           f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
           f'<v:rect id="shape_{id(paragraph)}_{abs(hash((text,x,y)))}" style="{style}" {attrs}>{body}</v:rect></w:pict>')
    paragraph.add_run()._r.append(parse_xml(xml))


def export_word(result, alignment, template, template_dir, output_path):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(595 / 72)
    section.page_height = Inches(842 / 72)
    section.top_margin = section.bottom_margin = Inches(0)
    section.left_margin = section.right_margin = Inches(0)
    for index, (page, aligned_page) in enumerate(zip(result["pages"], alignment["pages"])):
        if index:
            doc.add_page_break()
        if aligned_page["status"] != "success":
            continue
        definition = next(p for p in template["pages"] if p["pdf_page"] == aligned_page["template_page"])
        picture = Path(template_dir) / definition["image"]
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = 0
        image_part = paragraph.part.get_or_add_image(str(picture))[0]
        _shape(paragraph, "", 0, 0, 595, 842, 0, "000000", image_part)
        factor = 72 / 200
        for field in page["fields"]:
            right = field["bbox"][2]
            for item in field["items"]:
                a, b, c, d = item["bbox_template"]
                size = min(8, max(4, (d-b)*factor*.72))
                width = max(4, (right-max(a,field["bbox"][0])-2)*factor)
                # 长识别串缩小字号；文本框限制在所属字段内
                size = min(size, max(4, width / max(1, len(item["text_raw"])) * 1.7))
                _shape(paragraph, item["text_raw"], max(a,field["bbox"][0])*factor,
                       b*factor, width, max(9, (d-b+4)*factor), size,
                       "0032D2" if item["review_status"] == "ok" else "DC5A00")
        for item in page.get("unassigned", []):
            a, b, c, d = item["bbox_template"]
            _shape(paragraph, item["text_raw"], a*factor, b*factor,
                   max(5, (c-a+4)*factor), max(9, (d-b+4)*factor), 6, "DC5A00")
    doc.save(output_path)
