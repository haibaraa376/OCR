"""按模板字段识别，保存原始 OCR 与待核对状态。"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR


def recognize(alignment, template, template_dir, output_dir):
    ocr = PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det",
                    text_recognition_model_name="PP-OCRv6_medium_rec", device="gpu:0",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    result = {"source_image": alignment["source_image"], "coordinate_system": "rendered_pixels_200dpi",
              "model": "PP-OCRv6_medium_det+PP-OCRv6_medium_rec", "pages": []}
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(alignment["pages"], 1):
        record = {"side": page["side"], "alignment_status": page["status"], "fields": [], "unassigned": []}
        result["pages"].append(record)
        if page["status"] != "success":
            record["reason"] = page["reason"]
            continue
        definition = next(p for p in template["pages"] if p["pdf_page"] == page["template_page"])
        fields = definition["fields"]
        aligned = cv2.imread(str(Path(output_dir) / page["aligned_image"]))
        ref = cv2.imread(str(Path(template_dir) / page["template_image"]))
        preview = aligned.copy()
        filled = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)).convert("RGBA")
        entries = {f["id"]: {"field_id": f["id"], "name": f["name"], "type": f["type"],
                              "bbox": f["bbox"], "items": [], "status": "empty_or_not_detected"} for f in fields}
        record["fields"] = list(entries.values())
        candidates = []
        for field in fields:
            x0, y0, x1, y1 = field["bbox"]
            pad = 9 if field["type"] == "number" else 6
            cx0, cy0 = max(0, x0-pad), max(0, y0-3)
            cx1, cy1 = min(aligned.shape[1], x1+pad), min(aligned.shape[0], y1+3)
            crop = aligned[cy0:cy1, cx0:cx1]
            scale = 2 if field["type"] == "number" else 1
            if scale == 2:
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            try:
                prediction = ocr.predict(crop)
                data = prediction[0].json.get("res", {}) if prediction else {}
                for text, score, poly in zip(data.get("rec_texts", []), data.get("rec_scores", []), data.get("rec_polys", [])):
                    points = np.asarray(poly, np.float32) / scale + [cx0, cy0]
                    bbox = [int(points[:, 0].min()), int(points[:, 1].min()),
                            int(points[:, 0].max()), int(points[:, 1].max())]
                    candidates.append({"text_raw": str(text), "confidence": float(score),
                                       "bbox_template": bbox})
            except Exception as error:
                entries[field["id"]]["status"] = "ocr_failed"
                entries[field["id"]]["reason"] = str(error)
        # 外扩裁剪可能重复识别，优先保留置信度较高的同位置结果。
        accepted = []
        for item in sorted(candidates, key=lambda v: -v["confidence"]):
            a, b, c, d = item["bbox_template"]
            if any(item["text_raw"] == old["text_raw"] and
                   abs((a+c)-(old["bbox_template"][0]+old["bbox_template"][2])) < 16 and
                   abs((b+d)-(old["bbox_template"][1]+old["bbox_template"][3])) < 16 for old in accepted):
                continue
            accepted.append(item)
        for item in accepted:
            a, b, c, d = item["bbox_template"]
            mx, my = (a+c)/2, (b+d)/2
            owners = [f for f in fields if f["bbox"][0] <= mx <= f["bbox"][2] and
                      f["bbox"][1] <= my <= f["bbox"][3]]
            if len(owners) != 1:
                item.update(field_id=None, review_status="unassigned_cross_column")
                record["unassigned"].append(item)
                continue
            owner = owners[0]
            item["field_id"] = owner["id"]
            inside = max(0, min(c, owner["bbox"][2]) - max(a, owner["bbox"][0]))
            if inside / max(1, c-a) < .8:
                status = "cross_column_review"
            elif c-a > .9*(owner["bbox"][2]-owner["bbox"][0]):
                status = "long_text_review"
            elif item["confidence"] < .65:
                status = "low_confidence_review"
            else:
                status = "ok"
            item["review_status"] = status
            entries[owner["id"]]["items"].append(item)
        for field in fields:
            entry = entries[field["id"]]
            entry["items"].sort(key=lambda v: (v["bbox_template"][1], v["bbox_template"][0]))
            if entry["items"]:
                entry["status"] = "recognized"
            x0, y0, x1, y1 = field["bbox"]
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 170, 0), 2)
            for item in entry["items"]:
                a, b, c, d = item["bbox_template"]
                color = (0, 50, 210, 255) if item["review_status"] == "ok" else (220, 90, 0, 255)
                size = min(20, max(9, d-b))
                while size > 8:
                    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
                    if ImageDraw.Draw(filled).textbbox((0, 0), item["text_raw"], font=font)[2] <= x1-max(a, x0+2)-2:
                        break
                    size -= 1
                # 用字段宽度裁切绘制层，防止预览文字覆盖相邻列。
                layer = Image.new("RGBA", (x1-x0, max(1, d-b+5)), (0, 0, 0, 0))
                ImageDraw.Draw(layer).text((max(0, a-x0), 0), item["text_raw"], font=font, fill=color)
                filled.alpha_composite(layer, (x0, b))
                cv2.rectangle(preview, (a, b), (c, d), (0, 180, 0) if item["review_status"] == "ok" else (0, 120, 255), 2)
        for item in record["unassigned"]:
            a, b, c, d = item["bbox_template"]
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", min(18, max(9, d-b)))
            ImageDraw.Draw(filled).text((a, b), item["text_raw"], font=font, fill=(220, 90, 0, 255))
            cv2.rectangle(preview, (a, b), (c, d), (0, 120, 255), 2)
        cv2.imwrite(str(images_dir / f"ocr_boxes_{index:02d}.png"), preview)
        cv2.imwrite(str(images_dir / f"filled_{index:02d}.png"), cv2.cvtColor(np.asarray(filled.convert("RGB")), cv2.COLOR_RGB2BGR))
    return result
