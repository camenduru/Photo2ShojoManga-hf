from PIL import Image, ImageOps
import numpy as np
import cv2
from rembg import remove

def remove_background(input_image):
    rgba_image = remove(input_image)
    # アルファチャネルをマスクとして使用
    alpha_channel = rgba_image[:, :, 3]

    # 白い背景画像を作成
    background = np.ones_like(rgba_image, dtype=np.uint8) * 255
    # マスクを適用
    background_masked = cv2.bitwise_and(background, background, mask=alpha_channel)
    return background_masked

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