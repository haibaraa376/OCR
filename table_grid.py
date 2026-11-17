import cv2
import numpy as np


def _cluster(values, tolerance, minimum=1):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [int(np.median(group)) for group in groups if len(group) >= minimum]


def _lines(binary, horizontal):
    h, w = binary.shape
    length = (w if horizontal else h)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, length // 35), 1) if horizontal else (1, max(20, length // 35)))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # 小间隙闭合，保留局部边框。
    gap = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, length // 160), 1) if horizontal else (1, max(5, length // 160)))
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, gap)


def _coordinates(mask, horizontal, cell_boxes, offset, limit):
    projection = (mask > 0).sum(axis=1 if horizontal else 0)
    span = mask.shape[1 if horizontal else 0]
    strong = np.where(projection >= max(15, span * .10))[0].tolist()
    # PP 单元格框只辅助补足局部线，避免文字框生成额外行。
    candidates = _cluster(strong, max(3, limit // 300))
    edges = []
    for box in cell_boxes:
        edges.extend([box[1 if horizontal else 0] - offset, box[3 if horizontal else 2] - offset])
    for edge in _cluster(edges, max(5, limit // 100), 3):
        support = projection[max(0, edge - 4):min(limit, edge + 5)].max(initial=0)
        if 0 <= edge < limit and (horizontal or support > span * .06) and all(abs(edge - item) > max(5, limit // 100) for item in candidates):
            candidates.append(edge)
    return sorted(candidates)


def _covered(mask, x1, y1, x2, y2, horizontal):
    h, w = mask.shape
    if horizontal:
        y = int(np.clip((y1 + y2) / 2, 0, h - 1))
        a, b = max(0, int(x1 + 2)), min(w, int(x2 - 2))
        strip = mask[max(0, y - 3):min(h, y + 4), a:b]
        return bool(strip.size and np.mean(np.any(strip > 0, axis=0)) > .38)
    x = int(np.clip((x1 + x2) / 2, 0, w - 1))
    a, b = max(0, int(y1 + 2)), min(h, int(y2 - 2))
    strip = mask[a:b, max(0, x - 3):min(w, x + 4)]
    return bool(strip.size and np.mean(np.any(strip > 0, axis=1)) > .38)


def _cells(xs, ys, horizontal, vertical, boxes):
    rows, cols = len(ys) - 1, len(xs) - 1
    used = np.zeros((rows, cols), bool)
    cells = []
    for r in range(rows):
        for c in range(cols):
            if used[r, c]:
                continue
            end_c = c + 1
            while end_c < cols and not used[r, end_c] and not _covered(vertical, xs[end_c], ys[r], xs[end_c], ys[r + 1], False) and any(
                b[0] < xs[end_c] - 3 < xs[end_c] + 3 < b[2] and b[1] < (ys[r] + ys[r + 1]) / 2 < b[3] for b in boxes
            ):
                end_c += 1
            end_r = r + 1
            while end_r < rows and not used[end_r, c:end_c].any() and all(
                not _covered(horizontal, xs[k], ys[end_r], xs[k + 1], ys[end_r], True)
                for k in range(c, end_c)
            ) and any(b[1] < ys[end_r] - 3 < ys[end_r] + 3 < b[3] and
                      b[0] < (xs[c] + xs[end_c]) / 2 < b[2] for b in boxes):
                end_r += 1
            used[r:end_r, c:end_c] = True
            cells.append({"row": r, "col": c, "rowspan": end_r - r, "colspan": end_c - c,
                          "bbox": [xs[c], ys[r], xs[end_c], ys[end_r]], "text": ""})
    return cells


def _fill_text(cells, ocr):
    boxes = ocr.get("rec_boxes", [])
    texts = ocr.get("rec_texts", [])
    for box, text in zip(boxes, texts):
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for cell in cells:
            a, b, c, d = cell["bbox"]
            if a <= x <= c and b <= y <= d:
                cell.setdefault("_items", []).append((y, x, text))
                break
    for cell in cells:
        cell["text"] = "\n".join(item[2] for item in sorted(cell.pop("_items", [])))


def recover_tables(image, result):
    """PP 定位表格，OpenCV 恢复网格；坐标均为当前纸页像素坐标。"""
    blocks = [block for block in result.get("parsing_res_list", []) if block.get("block_label") == "table"]
    pp_tables = result.get("table_res_list", [])
    tables = []
    for block in blocks:
        x1, y1, x2, y2 = map(int, block["block_bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 - x1 < 30 or y2 - y1 < 30:
            continue
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 12)
        horizontal, vertical = _lines(binary, True), _lines(binary, False)
        match = max(pp_tables, key=lambda t: sum(
            x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2
            for b in t.get("cell_box_list", [])), default={})
        boxes = match.get("cell_box_list", [])
        xs = _coordinates(vertical, False, boxes, x1, x2 - x1)
        ys = _coordinates(horizontal, True, boxes, y1, y2 - y1)
        # 页面边缘补全，不靠 OCR 文字推断表格行。
        for coords, size in ((xs, x2 - x1), (ys, y2 - y1)):
            if not coords or coords[0] > size * .03:
                coords.insert(0, 0)
            if coords[-1] < size * .97:
                coords.append(size)
        if len(xs) < 2 or len(ys) < 2:
            continue
        local_boxes = [[b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1] for b in boxes]
        cells = _cells(xs, ys, horizontal, vertical, local_boxes)
        xs, ys = [x + x1 for x in xs], [y + y1 for y in ys]
        for cell in cells:
            a, b, c, d = cell["bbox"]
            cell["bbox"] = [a + x1, b + y1, c + x1, d + y1]
        _fill_text(cells, result.get("overall_ocr_res", {}))
        tables.append({"bbox": [x1, y1, x2, y2], "columns": xs, "rows": ys, "cells": cells})
    return tables
