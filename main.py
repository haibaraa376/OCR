import argparse
import json
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
    )
    for path in paths:
        source = Path(path)
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if source.is_dir() else [source]
        for file in files:
            pages = []
            for image in load_pages(file):
                result = next(iter(pipeline.predict(image)))
                # Paddle 原始结果含 LayoutBlock 对象，JSON 属性转为统一字典。
                data = result.json["res"]
                height, width = image.shape[:2]
                pages.append({"width": width, "height": height,
                              "tables": recover_tables(image, data),
                              "blocks": data.get("parsing_res_list", [])})
            with (output / f"{file.stem}_structure.json").open("w", encoding="utf-8") as stream:
                json.dump({"source": str(file), "pages": pages}, stream, ensure_ascii=False, indent=2)
            destination = output / f"{file.stem}_rebuilt.docx"
            export_word(pages, destination)
            print(f"已生成: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="有线表格文档结构恢复")
    parser.add_argument("images", nargs="+", help="JPG/PNG 图片或图片文件夹")
    run(parser.parse_args().images)
