import os

import requests
from tqdm import tqdm
import shutil

from PIL import Image, ImageOps
import numpy as np
import cv2

def dl_cn_model(model_dir):
    folder = model_dir
    file_name = 'diffusion_pytorch_model.safetensors'
    url = "https://huggingface.co/2vXpSwA7/iroiro-lora/resolve/main/test_controlnet2/CN-anytest_v4-marged.safetensors"
    file_path = os.path.join(folder, file_name)
    if not os.path.exists(file_path):
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {file_name}')
        else:
            print(f'Failed to download {file_name}')
    else:
        print(f'{file_name} already exists.')

def dl_cn_config(model_dir):
  folder = model_dir
  file_name = 'config.json'
  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
     config_path = os.path.join(os.getcwd(), file_name)
     shutil.copy(config_path, file_path)

def dl_tagger_model(model_dir):
    model_id = 'SmilingWolf/wd-vit-tagger-v3'
    files = [
        'config.json', 'model.onnx', 'selected_tags.csv', 'sw_jax_cv_config.json'
    ]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for file in files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            url = f'https://huggingface.co/{model_id}/resolve/main/{file}'
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f'Downloaded {file}')
            else:
                print(f'Failed to download {file}')
        else:
            print(f'{file} already exists.')


def dl_lora_model(model_dir):
    models = {
        "tcd-animaginexl-3_1.safetensors": "https://huggingface.co/furusu/SD-LoRA/resolve/main/tcd-animaginexl-3_1.safetensors",
        "tori29umai_line.safetensors": "https://huggingface.co/tori29umai/Egara_Lora/resolve/main/tori29umai_line.safetensors",
    }
    
    # 各モデルを確認し、必要に応じてダウンロード
    for file_name, url in models.items():
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f'Downloaded {file_name}')
            else:
                print(f'Failed to download {file_name}')
        else:
            print(f'{file_name} already exists.')