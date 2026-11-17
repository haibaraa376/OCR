import cv2
import numpy as np


def _rectify(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(contour) < 0.7 * w * h:
            continue
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4:
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        bx, by, bw, bh = cv2.boundingRect(polygon)
        if bx > w * .08 or by > h * .08 or bx + bw < w * .92 or by + bh < h * .92:
            continue
        ordered = np.zeros((4, 2), np.float32)
        ordered[0] = points[np.argmin(points.sum(axis=1))]
        ordered[2] = points[np.argmax(points.sum(axis=1))]
        ordered[1] = points[np.argmin(np.diff(points, axis=1))]
        ordered[3] = points[np.argmax(np.diff(points, axis=1))]
        widths = [np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])]
        heights = [np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])]
        nw, nh = int(max(widths)), int(max(heights))
        if nw < 100 or nh < 100:
            break
        target = np.float32([[0, 0], [nw - 1, 0], [nw - 1, nh - 1], [0, nh - 1]])
        return cv2.warpPerspective(image, cv2.getPerspectiveTransform(ordered, target), (nw, nh), borderValue=(255, 255, 255))
    return image


def split_pages(image):
    """依据中缝空白及两侧内容分割双页，单页原样返回。"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if w / h < 1.25:
        return [_rectify(image)]
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 12) > 0
    profile = ink[int(h * .12):int(h * .88)].mean(axis=0)
    smooth = np.convolve(profile, np.ones(max(9, w // 100)) / max(9, w // 100), mode="same")
    lo, hi = int(w * .38), int(w * .62)
    center = lo + int(np.argmin(smooth[lo:hi]))
    left = np.median(smooth[int(w * .15):int(w * .35)])
    right = np.median(smooth[int(w * .65):int(w * .85)])
    if min(left, right) < .025 or smooth[center] > min(left, right) * .55:
        return [_rectify(image)]
    return [_rectify(image[:, :center]), _rectify(image[:, center:])]


def load_pages(path):
    # Windows 下 OpenCV 的 imread 对中文路径可能失败。
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return split_pages(image)
