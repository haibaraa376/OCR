import math

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_ROW_HEIGHT_RULE, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _element(parent, name, **attributes):
    """补充 python-docx 未直接提供的 Word XML 设置，qn 用于添加命名空间。"""
    node = OxmlElement('w:'+name)
    for key,value in attributes.items():
        node.set(qn('w:'+key),str(value))
    parent.append(node)
    return node


def _page_size(section, width, height):
    # 保持原图比例，短边为 A4 短边；每页独立设置纸张与边距。
    short = Cm(21)
    ratio = min(width,height)
    section.page_width = int(short*width/ratio)
    section.page_height = int(short*height/ratio)
    section.top_margin = section.bottom_margin = Cm(.5)
    section.left_margin = section.right_margin = Cm(.5)
    return min((section.page_width-Cm(1))/width,(section.page_height-Cm(1))/height)


def _add_table(doc, data, scale):
    xs,ys=data['columns'],data['rows']
    table=doc.add_table(rows=len(ys)-1,cols=len(xs)-1)
    table.autofit=False
    props=table._tbl.tblPr
    # 浮动表格使用纸页坐标定位，避免上下段落把表格挤离原位置。
    twips=lambda pixel: round(pixel*scale/635)
    _element(props,'tblpPr',horzAnchor='page',vertAnchor='page',
             tblpX=round(Cm(.5)/635)+twips(xs[0]),tblpY=round(Cm(.5)/635)+twips(ys[0]),
             leftFromText=0,rightFromText=0,topFromText=0,bottomFromText=0)
    width=props.find(qn('w:tblW')); width.set(qn('w:type'),'dxa'); width.set(qn('w:w'),str(twips(xs[-1]-xs[0])))
    margins=_element(props,'tblCellMar')
    for side in ('top','bottom','left','right'):
        _element(margins,side,w=0,type='dxa')
    # 不使用 Table Grid，真实边框由每个单元格分别指定。
    for c in range(len(xs)-1):
        table.columns[c].width=int((xs[c+1]-xs[c])*scale)
    for r,row in enumerate(table.rows):
        row.height=int((ys[r+1]-ys[r])*scale)
        row.height_rule=WD_ROW_HEIGHT_RULE.EXACTLY
        _element(row._tr.get_or_add_trPr(),'cantSplit')
        for c,cell in enumerate(row.cells):
            cell.width=table.columns[c].width
    overflow=[]
    for item in data['cells']:
        r,c=item['row'],item['col']; er=r+item['rowspan']; ec=c+item['colspan']
        cell=table.cell(r,c)
        if er>r+1 or ec>c+1:
            cell=cell.merge(table.cell(er-1,ec-1))
        cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        borders=_element(cell._tc.get_or_add_tcPr(),'tcBorders')
        for side,exists in item['borders'].items():
            _element(borders,side,val='single' if exists else 'nil',sz=4,color='000000')
        text=item.get('text',''); cell.text=text
        width_pt=(xs[ec]-xs[c])*scale/12700
        height_pt=(ys[er]-ys[r])*scale/12700
        # 仅对较短的窄列文本推断竖排，避免把长正文错误旋转。
        vertical=item.get('text_direction')=='vertical' or (width_pt<22 and height_pt>width_pt*3 and len(text.replace('\n',''))<35)
        if vertical:
            _element(cell._tc.get_or_add_tcPr(),'textDirection',val='tbRl')
        size=8.
        def fits(font):
            across=max(1,int((height_pt if vertical else width_pt)/font))
            lines=sum(max(1,math.ceil(len(line)/across)) for line in text.split('\n'))
            return lines*font*1.08 <= (width_pt if vertical else height_pt)
        while size>5 and not fits(size):
            size-=.5
        if text and not fits(size):
            # 不静默截断：确实容不下时允许该行扩展，并在结构数据中记录。
            overflow.append(item)
            for rr in range(r,er):
                table.rows[rr].height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
        for paragraph in cell.paragraphs:
            fmt=paragraph.paragraph_format
            fmt.space_before=fmt.space_after=Pt(0)
            fmt.line_spacing=Pt(size*1.08)
            for run in paragraph.runs:
                run.font.size=Pt(size)
    # Word 的纵向合并由起始格和多个续接格组成。
    # 只给起始格设边框会导致长竖线在后续行消失，因此逐个 OOXML 格设置。
    owners={}
    for item in data['cells']:
        for rr in range(item['row'],item['row']+item['rowspan']):
            for cc in range(item['col'],item['col']+item['colspan']):
                owners[rr,cc]=item
    for rr,tr in enumerate(table._tbl.tr_lst):
        cc=0
        for tc in tr.tc_lst:
            span=tc.grid_span
            item=owners[rr,cc]
            props=tc.get_or_add_tcPr()
            old=props.find(qn('w:tcBorders'))
            if old is not None:
                props.remove(old)
            borders=_element(props,'tcBorders')
            sides={'left':item['borders']['left'], 'right':item['borders']['right'],
                   'top':item['borders']['top'] if rr==item['row'] else False,
                   'bottom':item['borders']['bottom'] if rr==item['row']+item['rowspan']-1 else False}
            for side,exists in sides.items():
                _element(borders,side,val='single' if exists else 'nil',sz=4,color='000000')
            cc+=span
    return overflow


def export_word(pages, destination):
    """按页面位置重建可编辑对象，返回文字溢出位置供调用者提示。"""
    doc=Document(); warnings=[]
    style=doc.styles['Normal']; style.font.name='宋体'; style.font.size=Pt(8)
    style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    style.paragraph_format.space_after=Pt(0)
    for index,page in enumerate(pages):
        section=doc.sections[0] if not index else doc.add_section(WD_SECTION_START.NEW_PAGE)
        scale=_page_size(section,page['width'],page['height'])
        for table in page['tables']:
            for cell in _add_table(doc,table,scale):
                warnings.append({'page':index+1,'bbox':cell['bbox'],'reason':'文字可能需要扩展行高'})
        # 表格外的文字和草图用框架定位；整张表格始终是 Word 表格。
        for block in page.get('blocks',[]):
            if block.get('block_label')=='table':
                continue
            bbox=block.get('block_bbox')
            if not bbox:
                continue
            x,y,right,bottom=bbox
            paragraph=doc.add_paragraph()
            _element(paragraph._p.get_or_add_pPr(),'framePr',
                           hAnchor='page',vAnchor='page',wrap='none',
                           x=round((Cm(.5)+x*scale)/635),y=round((Cm(.5)+y*scale)/635),
                           w=max(1,round((right-x)*scale/635)))
            if block.get('image_path'):
                paragraph.add_run().add_picture(block['image_path'],width=int((right-x)*scale))
            else:
                paragraph.add_run(block.get('block_content',''))
    doc.save(destination)
    return warnings
