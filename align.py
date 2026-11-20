"""双页分割、模板匹配和原图到标准模板的坐标变换。"""
from pathlib import Path

import cv2
import numpy as np


def split_pages(photo):
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 中缝通常是一条较宽的浅色带；只在画面中央搜索。
    col = np.mean(gray[int(h * .12):int(h * .88)], axis=0)
    smooth = cv2.GaussianBlur(col.reshape(1, -1), (151, 1), 0).ravel()
    lo, hi = int(w * .42), int(w * .58)
    seam = int(lo + np.argmax(smooth[lo:hi]))
    margin = max(12, int(w * .015))
    boxes = [(0, 0, min(w, seam + margin), h), (max(0, seam - margin), 0, w, h)]
    return [(photo[y0:y1, x0:x1], (x0, y0, x1, y1)) for x0, y0, x1, y1 in boxes], seam


def _structure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2, (8, 8)).apply(gray)
    return gray


def _geometry_score(warped, template):
    # 仅用长直线评估模板与照片表格是否重合，避免手写字影响判断。
    def lines(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 41, 12)
        v = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, im.shape[0] // 12))))
        h = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, im.shape[1] // 12), 1)))
        return cv2.bitwise_or(v, h)
    a, b = lines(warped), lines(template)
    if cv2.countNonZero(a) < 100 or cv2.countNonZero(b) < 100:
        return 0.0
    near = cv2.dilate(a, np.ones((17, 17), np.uint8))
    return float(cv2.countNonZero(cv2.bitwise_and(near, b)) / cv2.countNonZero(b))


def _table_corners(image):
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 30, 100)
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .35:
            continue
        points = cv2.approxPolyDP(contour, .015 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(points) == 4 and cv2.isContourConvex(points.astype(np.float32)):
            points = points.astype(np.float32)
            order = np.argsort(points[:, 1])
            top = points[order[:2]][np.argsort(points[order[:2], 0])]
            bottom = points[order[2:]][np.argsort(points[order[2:], 0])]
            return np.array([top[0], top[1], bottom[1], bottom[0]], np.float32)
    return None


def align_one(crop, box, template, full_shape):
    detector = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = detector.detectAndCompute(_structure(crop), None)
    k2, d2 = detector.detectAndCompute(_structure(template), None)
    good = []
    H = None
    inlier_count = 0
    if d1 is not None and d2 is not None:
        pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        good = [a for a, b in pairs if a.distance < .76 * b.distance]
        if len(good) >= 12:
            src = np.float32([k1[m.queryIdx].pt for m in good])
            dst = np.float32([k2[m.trainIdx].pt for m in good])
            H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inlier_count = int(inliers.sum()) if inliers is not None else 0
    th, tw = template.shape[:2]
    ch, cw = crop.shape[:2]
    candidates = []
    if H is not None and inlier_count >= 10:
        corners = cv2.perspectiveTransform(np.float32([[[0, 0], [cw, 0], [cw, ch], [0, ch]]]), H)[0]
        area = abs(cv2.contourArea(corners)) / (tw * th)
        if .45 < area < 1.8 and cv2.isContourConvex(corners.astype(np.float32)):
            candidates.append((H, "features"))
    # 印刷表格外框提供四个稳定角点，扫描模板与电子模板均可使用。
    source_corners, target_corners = _table_corners(crop), _table_corners(template)
    if source_corners is not None and target_corners is not None:
        candidates.append((cv2.getPerspectiveTransform(source_corners, target_corners), "table_frame"))
    if not candidates:
        return None, "特征配准和表格外框定位均失败"
    evaluated = []
    for transform, method in candidates:
        picture = cv2.warpPerspective(crop, transform, (tw, th), borderValue=(255, 255, 255))
        evaluated.append((_geometry_score(picture, template), transform, picture, method))
    score, H, warped, method = max(evaluated, key=lambda c: c[0])
    if score < .25:
        return None, f"表格线重合度不足（{score:.3f}）"
    # crop 像素坐标加上裁剪偏移，得到完整照片坐标；矩阵组合后可直接映射原图。
    x0, y0 = box[:2]
    offset = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    full_H = H @ offset
    full_H /= full_H[2, 2]
    return {"matrix": full_H.tolist(), "aligned": warped, "score": score,
            "matches": len(good), "inliers": inlier_count, "method": method}, None


def run_alignment(photo_path, template_data, template_dir, output_dir):
    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise ValueError(f"无法读取照片：{photo_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops, seam = split_pages(photo)
    refs = [p for p in template_data["pages"] if p.get("reference_page") == p["pdf_page"] and p["type"] in ("left", "right")]
    results = []
    for side, (crop, box) in zip(("left", "right"), crops):
        candidates = []
        failures = []
        for page in refs:
            ref = cv2.imread(str(Path(template_dir) / page["image"]))
            result, error = align_one(crop, box, ref, photo.shape)
            if result:
                candidates.append((result["score"], page, result, ref))
            else:
                failures.append(f"PDF 第 {page['pdf_page']} 页：{error}")
        # 左右页依版式匹配；仅在其对应模板达到校验门槛时报告成功。
        candidates = [c for c in candidates if c[1]["type"] == side]
        record = {"side": side, "crop_box_original": list(box), "status": "failed"}
        if candidates:
            _, page, result, ref = max(candidates, key=lambda c: c[0])
            n = len(results) + 1
            cv2.imwrite(str(output_dir / f"page_{n:02d}.png"), result["aligned"])
            overlay = cv2.addWeighted(result["aligned"], .5, ref, .5, 0)
            cv2.imwrite(str(output_dir / f"overlay_{n:02d}.png"), overlay)
            record.update({"status": "success", "template_type": page["type"],
                           "template_page": page["pdf_page"], "template_image": page["image"],
                           "matrix_original_to_template": result["matrix"],
                           "line_overlap": result["score"], "feature_matches": result["matches"],
                           "ransac_inliers": result["inliers"], "aligned_image": f"page_{n:02d}.png",
                           "method": result["method"],
                           "overlay_image": f"overlay_{n:02d}.png"})
        else:
            record["reason"] = "; ".join(failures)
        results.append(record)
    return {"source_image": str(Path(photo_path).resolve()), "source_size": [photo.shape[1], photo.shape[0]],
            "seam_x_original": seam, "coordinate_system": "original_photo_to_rendered_template_pixels_200dpi",
            "pages": results}
