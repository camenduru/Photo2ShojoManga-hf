from PIL import Image, ImageOps
import numpy as np
import cv2
from rembg import remove

def background_removal(input_image_path):
    """
    指定された画像から背景を除去し、透明部分を白背景にブレンドして返す関数
    """
    try:
        input_image = Image.open(input_image_path)
    except IOError:
        print(f"Error: Cannot open {input_image_path}")
        return None

    # 背景除去処理
    result = remove(input_image)
    result_path = "tmp.png"
    result.save(result_path)
    
    return result_path

def resize_image_aspect_ratio(image):
    # 元の画像サイズを取得
    original_width, original_height = image.size

    # アスペクト比を計算
    aspect_ratio = original_width / original_height

    # 標準のアスペクト比サイズを定義
    sizes = {
        1: (1024, 1024),  # 正方形
        4/3: (1152, 896),  # 横長画像
        3/2: (1216, 832),
        16/9: (1344, 768),
        21/9: (1568, 672),
        3/1: (1728, 576),
        1/4: (512, 2048),  # 縦長画像
        1/3: (576, 1728),
        9/16: (768, 1344),
        2/3: (832, 1216),
        3/4: (896, 1152)
    }

    # 最も近いアスペクト比を見つける
    closest_aspect_ratio = min(sizes.keys(), key=lambda x: abs(x - aspect_ratio))
    target_width, target_height = sizes[closest_aspect_ratio]

    # リサイズ処理
    resized_image = image.resize((target_width, target_height), Image.LANCZOS)

    return resized_image


def base_generation(size, color):
    canvas = Image.new("RGBA", size, color) 
    return canvas     