from paddleocr import PPStructureV3

# 初始化通用文档结构识别模型
pipeline = PPStructureV3(
    device="gpu:0",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    use_chart_recognition=False
)

# 识别技术文档
results = pipeline.predict("日志文字识别样例/12-LZD-04/DSC03651.JPG")
for result in results:
    # 输出版面检测可视化图片
    result.save_to_img(save_path="output")
    # 输出表格和文字的结构化结果
    result.save_to_json(save_path="output")
    # 自动生成Word文档
    result.save_to_word(save_path="output")
print("文档解析完成")
