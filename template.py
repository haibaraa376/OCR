"""解析 PDF 模板并生成标准坐标系。"""
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf


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
        name = kind if kind in ("cover", "left", "right") and page.get("reference_page") == page["pdf_page"] else f"page_{page['pdf_page']:02d}"
        page["image"] = f"rendered/{name}.png"
        cv2.imwrite(str(output_dir / page["image"]), image)
    data = {"source_pdf": str(pdf_path.resolve()), "coordinate_system": "rendered_pixels_200dpi",
            "pages": pages}
    (output_dir / "template.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
