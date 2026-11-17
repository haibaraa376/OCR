import argparse
import json
import cv2
import numpy as np
from pathlib import Path

from paddleocr import PPStructureV3

from preprocess import load_pages
from table_grid import recover_tables
from word_export import export_word


def run(paths):
    output = Path("output")
    output.mkdir(exist_ok=True)
    # 保留现有 GPU 结构识别及 PP-OCRv6 文字识别配置。
    pipeline = PPStructureV3(
        device="gpu:0",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_chart_recognition=False,
        use_formula_recognition=False,
    )
    for path in paths:
        source = Path(path)
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if source.is_dir() else [source]
        for file in files:
            # 保留输入相对路径，避免不同目录下同名图片互相覆盖。
            relative = file.resolve().relative_to(Path.cwd().resolve()) if file.resolve().is_relative_to(Path.cwd().resolve()) else Path(file.drive.replace(':','')) / Path(*file.parts[1:])
            sample = output / relative.parent / file.name
            # 常用样例保持简短目录名；已存在其他来源时改用完整相对路径。
            preferred = output / file.stem
            existing = preferred / 'intermediate' / 'structure.json'
            if not existing.exists() or json.loads(existing.read_text(encoding='utf-8')).get('source') == str(file.resolve()):
                sample = preferred
            intermediate = sample / 'intermediate'
            intermediate.mkdir(parents=True, exist_ok=True)
            pages = []
            for page in load_pages(file, with_metadata=True):
                image = page.pop('image')
                folder = intermediate / f"page_{page['page_index']:02d}"
                folder.mkdir(exist_ok=True)
                cv2.imencode('.jpg',image)[1].tofile(str(folder/'preprocessed.jpg'))
                result = next(iter(pipeline.predict(image)))
                data = result.json['res']
                (folder/'model_result.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
                # 先建立几何结构，再分配 OCR 文字，文字准确率不影响网格。
                height,width = image.shape[:2]
                page.update(width=width,height=height,tables=recover_tables(image,data,folder),
                            blocks=data.get('parsing_res_list',[]))
                for number,block in enumerate(page['blocks']):
                    if block.get('block_label') in ('image','figure','chart'):
                        x1,y1,x2,y2=map(int,block['block_bbox'])
                        crop=image[max(0,y1):min(height,y2),max(0,x1):min(width,x2)]
                        if crop.size:
                            asset=folder/f'figure_{number:02d}.png'
                            cv2.imencode('.png',crop)[1].tofile(str(asset))
                            block['image_path']=str(asset.resolve())
                # 核心结构检查：每个基础格只能归属一个单元格。
                for table in page['tables']:
                    if any(b <= a for key in ('rows', 'columns')
                           for a, b in zip(table[key], table[key][1:])):
                        raise ValueError('网格边界必须严格递增')
                    occupied=np.zeros((len(table['rows'])-1,len(table['columns'])-1),dtype=np.uint8)
                    for cell in table['cells']:
                        r,c=cell['row'],cell['col']; er=r+cell['rowspan']; ec=c+cell['colspan']
                        if not (0<=r<er<=occupied.shape[0] and 0<=c<ec<=occupied.shape[1]):
                            raise ValueError('单元格索引越界')
                        occupied[r:er,c:ec]+=1
                    if not np.all(occupied==1):
                        raise ValueError('单元格有重叠或遗漏')
                pages.append(page)
            destination=sample/'result.docx'
            warnings=export_word(pages,destination)
            with (intermediate/'structure.json').open('w',encoding='utf-8') as stream:
                json.dump({'source':str(file.resolve()),'pages':pages,'layout_warnings':warnings},stream,ensure_ascii=False,indent=2)
            print(f'已生成: {destination}；纸页 {len(pages)}，排版溢出提示 {len(warnings)}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="有线表格文档结构恢复")
    parser.add_argument("images", nargs="+", help="JPG/PNG 图片或图片文件夹")
    run(parser.parse_args().images)
