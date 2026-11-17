import cv2
import numpy as np


def _line_mask(binary, horizontal):
    """长核过滤文字，短核修补断线；长度随表格分辨率缩放。"""
    extent = binary.shape[1 if horizontal else 0]
    length, gap = max(3, round(extent*.035)), max(1, round(extent*.004))
    shape = lambda n: (n, 1) if horizontal else (1, n)
    repaired = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(shape(gap)[::-1], np.uint8))
    return cv2.morphologyEx(repaired, cv2.MORPH_OPEN, np.ones(shape(length)[::-1], np.uint8))


def _boundaries(mask, cross, horizontal):
    """只接受长线或接触至少两个交点的局部线；PP 框不创建边界。"""
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    extent = mask.shape[1 if horizontal else 0]
    perpendicular = mask.shape[0 if horizontal else 1]
    tolerance = max(1, perpendicular*.012)
    dilated = cv2.dilate(cross, np.ones((3, 3), np.uint8))
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        length, thickness = (w, h) if horizontal else (h, w)
        intersections = cv2.connectedComponents(((labels[y:y+h, x:x+w]==i) & (dilated[y:y+h, x:x+w]>0)).astype(np.uint8))[0]-1
        if length < extent*.045 or thickness > max(length*.12, perpendicular*.012):
            continue
        if intersections < 2 and length < extent*.28:
            continue
        coordinate = centers[i][1 if horizontal else 0]
        candidates.append((coordinate, length))
    groups = []
    for coordinate, weight in sorted(candidates):
        if groups and coordinate - groups[-1][-1][0] < tolerance:
            groups[-1].append((coordinate, weight))
        else:
            groups.append([(coordinate, weight)])
    return [int(round(sum(a*b for a,b in g)/sum(b for a,b in g))) for g in groups]


def _support(mask, coordinate, start, end, horizontal):
    """返回边界覆盖比例，允许线条在细小范围内偏移。"""
    radius = max(1, round(min(mask.shape)*.003))
    margin = min(round((end-start)*.08), radius*2)
    if horizontal:
        strip = mask[max(0, coordinate-radius):coordinate+radius+1, start+margin:end-margin]
        axis = 0
    else:
        strip = mask[start+margin:end-margin, max(0, coordinate-radius):coordinate+radius+1]
        axis = 1
    return float(np.any(strip, axis=axis).mean()) if strip.size else 0.


def _topology(xs, ys, hm, vm, boxes):
    """缺边连接形成连通区域；仅接受矩形、且内部没有实线的合并。"""
    nr, nc = len(ys)-1, len(xs)-1
    hb = [[_support(hm, y, xs[c], xs[c+1], True) for c in range(nc)] for y in ys]
    vb = [[_support(vm, x, ys[r], ys[r+1], False) for x in xs] for r in range(nr)]
    # 并查集把可跨越的基础格连起来；每个编号对应 row * 列数 + col。
    parent = list(range(nr*nc))
    uncertain = set()
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def connect(a, b, support, x, y):
        # 低覆盖且有 PP 跨界证据时合并；有少量墨迹但不确定的边保留标记。
        crosses = any(q[0]<x<q[2] and q[1]<y<q[3] for q in boxes)
        if support < .12 or (support < .3 and crosses):
            parent[find(a)] = find(b)
        elif support < .55:
            uncertain.update((a,b))
    for r in range(nr):
        for c in range(nc):
            i=r*nc+c
            if c+1<nc:
                connect(i,i+1,vb[r][c+1],xs[c+1],(ys[r]+ys[r+1])/2)
            if r+1<nr:
                connect(i,i+nc,hb[r+1][c],(xs[c]+xs[c+1])/2,ys[r+1])
    groups={}
    for i in range(nr*nc):
        groups.setdefault(find(i),[]).append(divmod(i,nc))
    cells=[]
    for group in groups.values():
        r0,c0=np.min(group,axis=0).tolist(); r1,c1=(np.max(group,axis=0)+1).tolist()
        rectangular = len(group)==(r1-r0)*(c1-c0)
        internal = any(hb[r][c]>.55 for r in range(r0+1,r1) for c in range(c0,c1)) or any(vb[r][c]>.55 for r in range(r0,r1) for c in range(c0+1,c1))
        regions=[]
        remaining=set(group)
        # L 形连通区域不能直接合并：拆成内部无实线的最大矩形。
        # 优先保留长条单元格，避免把整个连通分量退化成一堆小格。
        while remaining:
            best=None; best_score=-1
            for a,b in sorted(remaining):
                for d in range(a+1,r1+1):
                    for e in range(b+1,c1+1):
                        if any((r,c) not in remaining for r in range(a,d) for c in range(b,e)):
                            break
                        if any(hb[r][c]>.4 for r in range(a+1,d) for c in range(b,e)) or any(vb[r][c]>.4 for r in range(a,d) for c in range(b+1,e)):
                            continue
                        score=(xs[e]-xs[b])*(ys[d]-ys[a])
                        if score>best_score:
                            best=(a,b,d,e);best_score=score
            a,b,d,e=best
            regions.append(best)
            remaining.difference_update((r,c) for r in range(a,d) for c in range(b,e))
        for a,b,d,e in regions:
            borders={'top':hb[a][b:e], 'bottom':hb[d][b:e],
                     'left':[vb[r][b] for r in range(a,d)], 'right':[vb[r][e] for r in range(a,d)]}
            cells.append({'row':a,'col':b,'rowspan':d-a,'colspan':e-b,
                          'bbox':[xs[b],ys[a],xs[e],ys[d]],
                          'borders':{k:any(v>.4 for v in values) for k,values in borders.items()},
                          'border_support':borders,
                          'uncertain':not rectangular or internal or any(r*nc+c in uncertain for r in range(a,d) for c in range(b,e)), 'text':''})
    return cells, hb, vb


def _fill_text(cells, ocr):
    """按文字框中心分配文本；只填内容，不参与行列或合并判断。"""
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


def recover_tables(image, result, debug_dir=None):
    """结构仅依赖几何证据；文字分配在拓扑恢复完成后执行。"""
    tables=[]
    overlay=image.copy(); line_view=np.zeros_like(image)
    for block in result.get('parsing_res_list', []):
        if block.get('block_label')!='table':
            continue
        x1,y1,x2,y2=map(int,block['block_bbox'])
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(image.shape[1],x2),min(image.shape[0],y2)
        gray=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
        if min(gray.shape)<10:
            continue
        window=max(3,round(min(gray.shape)*.03)//2*2+1)
        binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,window,7)
        hm,vm=_line_mask(binary,True),_line_mask(binary,False)
        xs,ys=_boundaries(vm,hm,False),_boundaries(hm,vm,True)
        if len(xs)<2 or len(ys)<2:
            continue
        candidates=[b for t in result.get('table_res_list',[]) for b in t.get('cell_box_list',[])
                    if x1<=(b[0]+b[2])/2<=x2 and y1<=(b[1]+b[3])/2<=y2]
        boxes=[[a-x1,b-y1,c-x1,d-y1] for a,b,c,d in candidates]
        cells,hb,vb=_topology(xs,ys,hm,vm,boxes)
        for cell in cells:
            a,b,c,d=cell['bbox']; cell['bbox']=[a+x1,b+y1,c+x1,d+y1]
            color=(0,140,255) if cell['uncertain'] else (0,190,0)
            cv2.rectangle(overlay,(a+x1,b+y1),(c+x1,d+y1),color,2)
        _fill_text(cells,result.get('overall_ocr_res',{}))
        line_view[y1:y2,x1:x2,1]=cv2.max(hm,line_view[y1:y2,x1:x2,1])
        line_view[y1:y2,x1:x2,2]=cv2.max(vm,line_view[y1:y2,x1:x2,2])
        tables.append({'bbox':[xs[0]+x1,ys[0]+y1,xs[-1]+x1,ys[-1]+y1],
                       'columns':[x+x1 for x in xs], 'rows':[y+y1 for y in ys],
                       'horizontal_edges':hb,'vertical_edges':vb,'cells':cells})
    if debug_dir is not None:
        for name,img in [('lines.png',line_view),('grid_overlay.png',overlay)]:
            cv2.imencode('.png',img)[1].tofile(str(debug_dir/name))
    return tables
