import cv2
import numpy as np


def _rectify(image):
    """用面积最大的闭合四边形校正；扩大输出画布，保留表格外文字。"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 12)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < .45 * w * h:
            break
        poly = cv2.approxPolyDP(contour, .015 * cv2.arcLength(contour, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        points = poly.reshape(4, 2).astype(np.float32)
        order = np.argsort(np.arctan2(points[:, 1] - points[:, 1].mean(), points[:, 0] - points[:, 0].mean()))
        points = points[order]
        points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
        nw = max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3]))
        nh = max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1]))
        target = np.float32([[0, 0], [nw, 0], [nw, nh], [0, nh]])
        matrix = cv2.getPerspectiveTransform(points, target)
        corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
        mapped = cv2.perspectiveTransform(corners, matrix)[0]
        low, high = np.floor(mapped.min(axis=0)), np.ceil(mapped.max(axis=0))
        # 过大的外推说明边界不可靠，宁可保留原图。
        if np.any(high-low > np.array([w, h])*1.5):
            continue
        shift = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1.]])
        matrix = shift @ matrix
        corrected = cv2.warpPerspective(image, matrix, tuple((high-low).astype(int)), borderValue=(255, 255, 255))
        return corrected, matrix
    return image, np.eye(3)


def split_pages(image, with_metadata=False):
    """寻找两块内容之间持续的低墨迹区域；不确定时保留单页。"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 12) > 0
    profile = ink[int(h*.12):int(h*.88)].mean(axis=0)
    size = max(3, w//100)
    smooth = np.convolve(profile, np.ones(size)/size, mode='same')
    cuts = [0, w]
    if w/h > 1.2:
        # 从低墨迹区间中选择最宽者，避免在某个窄的空白列中切页。
        indices = np.where(smooth < max(.002, np.quantile(smooth, .6)*.12))[0]
        groups = np.split(indices, np.where(np.diff(indices)>1)[0]+1)
        gaps = [g for g in groups if len(g)>w*.025 and g[0]>w*.2 and g[-1]<w*.8]
        if gaps:
            gap = max(gaps, key=len)
            cut = int((gap[0]+gap[-1])/2)
            if profile[:cut].mean()>.015 and profile[cut:].mean()>.015:
                cuts = [0, cut, w]
    pages = []
    for index, (a, b) in enumerate(zip(cuts, cuts[1:])):
        corrected, transform = _rectify(image[:, a:b])
        crop_transform = np.array([[1, 0, -a], [0, 1, 0], [0, 0, 1.]])
        forward = transform @ crop_transform
        pages.append({'image': corrected, 'page_index': index+1,
                      'original_size': [w, h], 'crop_bbox': [a, 0, b, h],
                      'source_to_page': forward.tolist(),
                      'page_to_source': np.linalg.inv(forward).tolist()})
    return pages if with_metadata else [p['image'] for p in pages]


def load_pages(path, with_metadata=False):
    # imdecode + fromfile 支持 Windows 中文路径。
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片: {path}')
    return split_pages(image, with_metadata)
