from PIL import Image
import numpy as np
import cv2
from rembg import remove

ef background_removal(input_image_path):
    """
    指定された画像から背景を除去し、透明部分を白背景にブレンドして返す関数。
    小さなドットや、指定のアルファ閾値以下の小さな半透明領域も無視してトリミングを行います。

    Parameters:
    - input_image_path: 背景除去する画像のパス
    """
    try:
        input_image = Image.open(input_image_path).convert("RGBA")
    except IOError:
        print(f"Error: Cannot open {input_image_path}")
        return None
        
    area_threshold=100　# 無視する小さい領域のピクセル数の閾値
    alpha_threshold=128　# 無視する小さい領域のピクセル数の閾値

    # 背景除去処理
    result = remove(input_image)

    # アルファチャンネル取得
    alpha = result.split()[-1]  # アルファチャンネルを取得
    bbox = alpha.getbbox()
    if bbox:
        # bbox 内の非透過ピクセルをカウント (alpha_threshold より大きいピクセルのみ)
        cropped_alpha = alpha.crop(bbox)
        non_transparent_pixel_count = sum(1 for pixel in cropped_alpha.getdata() if pixel >= alpha_threshold)
        
        # 指定した閾値以上の領域がある場合のみトリミング
        if non_transparent_pixel_count >= area_threshold:
            result = result.crop(bbox)
        else:
            print("Small or semi-transparent region ignored")

    # 結果を一時ファイルに保存
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