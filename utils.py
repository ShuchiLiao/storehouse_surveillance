import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def put_chinese_text(image, text, position, font_path, font_size, color):
    # 将 OpenCV 图像转换为 Pillow 图像
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建一个可以在图片上绘制的对象
    draw = ImageDraw.Draw(image_pil)
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    # 在 Pillow 图片上绘制中文文本
    draw.text(position, text, font=font, fill=color)
    # 将图片转换回 OpenCV 格式
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)\

    return image