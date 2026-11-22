"""在标准模板字段区域调用本机 PP-OCRv6。"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR


def recognize(alignment, template, template_dir, output_dir):
    # 模型目录已存在于当前 OCR 环境的 PaddleX 缓存中。
    ocr = PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det",
                    text_recognition_model_name="PP-OCRv6_medium_rec", device="gpu:0",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    result = {"source_image": alignment["source_image"], "coordinate_system": "rendered_pixels_200dpi",
              "model": "PP-OCRv6_medium_det+PP-OCRv6_medium_rec", "pages": []}
    for index, page in enumerate(alignment["pages"], 1):
        record = {"side": page["side"], "alignment_status": page["status"], "fields": []}
        result["pages"].append(record)
        if page["status"] != "success":
            record["reason"] = page["reason"]
            continue
        definition = next(p for p in template["pages"] if p["pdf_page"] == page["template_page"])
        aligned = cv2.imread(str(Path(output_dir) / page["aligned_image"]))
        ref = cv2.imread(str(Path(template_dir) / page["template_image"]))
        boxes_preview = aligned.copy()
        fill = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(fill)
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        for field in definition["fields"]:
            x0, y0, x1, y1 = field["bbox"]
            crop = aligned[y0:y1, x0:x1]
            entry = {"field_id": field["id"], "name": field["name"], "type": field["type"],
                     "bbox": field["bbox"], "items": []}
            record["fields"].append(entry)
            if crop.size == 0:
                entry["status"] = "invalid_region"
                continue
            try:
                predictions = ocr.predict(crop)
                payload = predictions[0].json if predictions else {}
                data = payload.get("res", payload)
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                polygons = data.get("rec_polys", [])
                for text, score, polygon in zip(texts, scores, polygons):
                    points = np.asarray(polygon, np.float32)
                    points[:, 0] += x0
                    points[:, 1] += y0
                    bbox = [int(points[:, 0].min()), int(points[:, 1].min()),
                            int(points[:, 0].max()), int(points[:, 1].max())]
                    # 接触裁剪边界的文字可能跨字段，明确标记供人工检查。
                    uncertain = bbox[0] <= x0 + 3 or bbox[2] >= x1 - 3
                    entry["items"].append({"text_raw": str(text), "confidence": float(score),
                                           "bbox_template": bbox, "field_assignment_uncertain": uncertain})
                entry["items"].sort(key=lambda item: (item["bbox_template"][1] // 20, item["bbox_template"][0]))
                entry["status"] = "recognized" if entry["items"] else "empty_or_not_detected"
            except Exception as error:
                entry["status"] = "ocr_failed"
                entry["reason"] = str(error)
            cv2.rectangle(boxes_preview, (x0, y0), (x1, y1), (0, 180, 0), 2)
            for item in entry["items"]:
                a, b, c, d = item["bbox_template"]
                cv2.rectangle(boxes_preview, (a, b), (c, d), (0, 0, 255), 2)
                if not item["field_assignment_uncertain"]:
                    draw.text((a, b), item["text_raw"], font=font, fill=(0, 50, 210))
        cv2.imwrite(str(Path(output_dir) / f"ocr_boxes_{index:02d}.png"), boxes_preview)
        cv2.imwrite(str(Path(output_dir) / f"filled_{index:02d}.png"), cv2.cvtColor(np.asarray(fill), cv2.COLOR_RGB2BGR))
    return result
