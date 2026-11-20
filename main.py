"""第一阶段入口：生成模板并将双页日志照片配准。"""
import argparse
import json
from pathlib import Path

from align import run_alignment
from template import build_template


def main():
    parser = argparse.ArgumentParser(description="钻探日志模板解析与双页配准")
    parser.add_argument("--template", default="templates/log.pdf", help="PDF 模板路径")
    parser.add_argument("--image", required=True, help="双页日志照片路径")
    parser.add_argument("--output", default=None, help="输出目录")
    args = parser.parse_args()
    pdf = Path(args.template)
    image = Path(args.image)
    out = Path(args.output) if args.output else Path("output") / image.stem
    data = build_template(pdf)
    result = run_alignment(image, data, pdf.parent, out)
    (out / "alignment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for page in result["pages"]:
        print(f"{page['side']}: {page['status']} " +
              (f"PDF 第 {page['template_page']} 页，线条重合度 {page['line_overlap']:.3f}" if page['status'] == "success" else page["reason"]))
    print(f"结果：{out / 'alignment.json'}")


if __name__ == "__main__":
    main()
