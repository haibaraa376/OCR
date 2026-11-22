"""解析 PDF 模板并生成标准坐标系。"""
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf


def _fields(kind):
    # 固定日志表格的 200 DPI 像素坐标；边界留白以避开印刷线。
    columns = (
        [("date_shift", "日期班次", 99, 154, "text"), ("operation", "作业名称及纯钻时间", 156, 379, "text"),
         ("depth_from", "钻进深度由", 382, 454, "number"), ("depth_to", "钻进深度至", 457, 529, "number"),
         ("depth_total", "钻进深度共", 532, 605, "number"), ("bit_diameter", "钻头直径", 609, 680, "number"),
         ("core_length", "岩心长度", 684, 777, "number"), ("recovery", "采取率", 780, 876, "number"),
         ("run_number", "回次编号", 879, 974, "text"), ("lithology", "岩层名称及特征", 978, 1553, "long_text")]
        if kind == "left" else
        [("section", "柱状剖面", 99, 304, "text"), ("layer_depth", "层深和层厚", 307, 379, "number"),
         ("drilling_condition", "钻进情况", 382, 568, "long_text"), ("casing_diameter", "套管直径", 572, 646, "number"),
         ("casing_depth", "套管累计深度", 649, 729, "number"), ("penetration_count", "标贯贯入击数", 733, 810, "number"),
         ("rod_length", "钻杆长度", 814, 953, "number"), ("water_level", "含水量和水位", 957, 1178, "number"),
         ("sample_number", "送试验件编号", 1182, 1324, "text"), ("remarks", "注", 1328, 1553, "long_text")])
    y0, y1 = (478, 1991) if kind == "left" else (514, 2027)
    return [{"id": f"{kind}.{key}", "name": name, "bbox": [x0, y0, x1, y1],
             "type": field_type, "multiple_records": True} for key, name, x0, x1, field_type in columns]


def _ink_profile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray < 205).astype(np.uint8)
    # 表格页有贯穿大部分页面的竖线；扫描版也适用。
    vertical = np.sum(binary, axis=0)
    return int(np.sum(vertical > image.shape[0] * 0.43))


def build_template(pdf_path, output_dir=None):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or pdf_path.parent)
    rendered = output_dir / "rendered"
    rendered.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "template.json"
    if manifest.exists():
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if Path(cached.get("source_pdf", "")) == pdf_path.resolve() and all(
                (output_dir / p["image"]).exists() for p in cached["pages"] if p.get("image")):
            for page in cached["pages"]:
                if page["type"] in ("left", "right"):
                    page["fields"] = _fields(page["type"])
                    page["image"] = f"rendered/{page['type']}.png"
                elif page["type"] == "unknown":
                    page["image"] = None
            manifest.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
            for unused in rendered.glob("page_*.png"):
                unused.unlink()
            return cached
    doc = pymupdf.open(pdf_path)
    pages = []
    images = []
    for index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(200 / 72, 200 / 72), alpha=False)
        image = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        images.append(image)
        words = [{"text": w[4], "bbox": [round(v, 2) for v in w[:4]]} for w in page.get_text("words")]
        lines = []
        for drawing in page.get_drawings():
            for item in drawing["items"]:
                if item[0] == "l":
                    lines.append([[round(item[1].x, 2), round(item[1].y, 2)],
                                  [round(item[2].x, 2), round(item[2].y, 2)]])
        pages.append({"pdf_page": index + 1, "size_pt": [page.rect.width, page.rect.height],
                      "size_px": [image.shape[1], image.shape[0]], "dpi": 200,
                      "printed_words": words, "printed_lines": lines,
                      "type": "unknown", "fields": []})

    # 按竖线结构找出表格页；首张非表格页不能仅凭页序猜测为封面。
    scores = [_ink_profile(im) for im in images]
    table_indices = [i for i, score in enumerate(scores) if score >= 8]
    groups = {}
    for i in table_indices:
        im = cv2.resize(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY), (240, 340))
        mask = (im < 205).astype(np.uint8)
        key = tuple(np.sum(mask, axis=0) > 125)
        groups.setdefault(key, []).append(i)
    # 同一版式按页面顺序复用代表页；左右类别由表格结构和交替位置共同确认。
    ordered = sorted(groups.values(), key=lambda g: min(g))
    if len(ordered) == 2 and all(len(g) >= 2 for g in ordered):
        for kind, group in zip(("left", "right"), ordered):
            for i in group:
                pages[i]["type"] = kind
                pages[i]["reference_page"] = group[0] + 1
    # 标题“钻探日志”是封面识别依据。
    for i, page in enumerate(pages):
        if page["type"] == "unknown" and "钻探日志" in "".join(w["text"] for w in page["printed_words"]):
            page["type"] = "cover"
            page["reference_page"] = i + 1

    for page, image in zip(pages, images):
        kind = page["type"]
        if kind in ("left", "right"):
            page["fields"] = _fields(kind)
            page["image"] = f"rendered/{kind}.png"
            if page["reference_page"] == page["pdf_page"]:
                cv2.imwrite(str(output_dir / page["image"]), image)
        elif kind == "cover":
            page["image"] = "rendered/cover.png"
            cv2.imwrite(str(output_dir / page["image"]), image)
        else:
            page["image"] = None
    data = {"source_pdf": str(pdf_path.resolve()), "coordinate_system": "rendered_pixels_200dpi",
            "pages": pages}
    (output_dir / "template.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
