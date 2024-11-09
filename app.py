import spaces
import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL

from PIL import Image
import os
import time
from utils.dl_utils import dl_cn_model, dl_cn_config, dl_tagger_model, dl_lora_model
from utils.image_utils import resize_image_aspect_ratio, base_generation, background_removal
from utils.prompt_utils import execute_prompt, remove_color, remove_duplicates
from utils.tagger import modelLoad, analysis

import pygit2
import shutil
import os

# クローンするリポジトリのURLと保存先のパス
repo_url = "https://github.com/jabir-zheng/TCD.git"
repo_dir = "TCD"

# Git リポジトリをクローン
repo = pygit2.clone_repository(repo_url, repo_dir)

# 現在の作業ディレクトリを取得
current_dir = os.getcwd()

# Git関連のファイル（.git）を無視するための関数
def ignore_git_files(path, names):
    return [name for name in names if name == '.git']

# クローンしたリポジトリの内容を現在のディレクトリにコピー
shutil.copytree(repo_dir, current_dir, ignore=ignore_git_files)

# コピーが完了したことを表示
print(f"Repository contents (excluding .git) have been copied to {current_dir}")

from scheduling_tcd import TCDScheduler

path = os.getcwd()
cn_dir = f"{path}/controlnet"
tagger_dir = f"{path}/tagger"
lora_dir = f"{path}/lora"
os.makedirs(cn_dir, exist_ok=True)
os.makedirs(tagger_dir, exist_ok=True)
os.makedirs(lora_dir, exist_ok=True)

dl_cn_model(cn_dir)
dl_cn_config(cn_dir)
dl_tagger_model(tagger_dir)
dl_lora_model(lora_dir)

# グローバル変数でpipeを管理
pipe = None
current_lora_model = None

def load_model(lora_model):
    global pipe, current_lora_model
    # 既にロードされたpipeがあり、同じLoRAモデルの場合は再利用
    if pipe is not None and current_lora_model == lora_model:
        return pipe  # キャッシュされたpipeを返す

    # 新しいpipeの生成
    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype, use_safetensors=True)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1", controlnet=controlnet, vae=vae, torch_dtype=dtype
    )
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    pipe.enable_model_cpu_offload()

    # LoRAモデルの設定
    if lora_model == "とりにく風":
        pipe.load_lora_weights(lora_dir, weight_name="tcd-animaginexl-3_1.safetensors", adapter_name="tcd-animaginexl-3_1")
        pipe.load_lora_weights(lora_dir, weight_name="tori29umai_line.safetensors", adapter_name="tori29umai_line")        
        pipe.set_adapters(["tcd-animaginexl-3_1", "tori29umai_line"], adapter_weights=[1.0, 1.0])
    elif lora_model == "プレーン":
        pipe.load_lora_weights(lora_dir, weight_name="tcd-animaginexl-3_1.safetensors", adapter_name="tcd-animaginexl-3_1")    
        pipe.set_adapters(["tcd-animaginexl-3_1"], adapter_weights=[1.0])

    # 現在のLoRAモデルを保存
    current_lora_model = lora_model
    return pipe

@spaces.GPU(duration=120)
def predict(lora_model, input_image_path, prompt, negative_prompt, controlnet_scale):
    # pipeをグローバル変数から取得
    pipe = load_model(lora_model)
    
    # 画像読み込みとリサイズ
    input_image = Image.open(input_image_path)
    base_image = base_generation(input_image.size, (255, 255, 255, 255)).convert("RGB")
    resize_image = resize_image_aspect_ratio(input_image)
    resize_base_image = resize_image_aspect_ratio(base_image)
    generator = torch.manual_seed(0)
    last_time = time.time()
    
    # プロンプト生成
    prompt = "masterpiece, best quality, monochrome, greyscale, lineart, white background, " + prompt  
    
    execute_tags = ["realistic", "nose", "asian"]
    prompt = execute_prompt(execute_tags, prompt)
    prompt = remove_duplicates(prompt)        
    prompt = remove_color(prompt)
    print(prompt)

    # 画像生成
    output_image = pipe(
        image=resize_base_image,
        control_image=resize_image,
        strength=1.0,
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=float(controlnet_scale),
        generator=generator,
        num_inference_steps=4,
        guidance_scale=0,
        eta=0.3, 
    ).images[0]
    print(f"Time taken: {time.time() - last_time}")
    output_image = output_image.resize(input_image.size, Image.LANCZOS)
    return output_image

class Img2Img:
    def __init__(self):
        self.demo = self.layout()
        self.tagger_model = None
        self.input_image_path = None
        self.bg_removed_image = None

    def process_prompt_analysis(self, input_image_path):
        if self.tagger_model is None:
            self.tagger_model = modelLoad(tagger_dir)
        tags = analysis(input_image_path, tagger_dir, self.tagger_model)
        prompt = remove_color(tags)
        execute_tags = ["realistic", "nose", "asian"]
        prompt = execute_prompt(execute_tags, prompt)
        prompt = remove_duplicates(prompt)
        return prompt

    def layout(self):
        css = """
        #intro{
            max-width: 32rem;
            text-align: center;
            margin: 0 auto;
        }
        """
        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column():
                    # LoRAモデル選択ドロップダウン
                    self.lora_model = gr.Dropdown(label="Image Style",  choices=["とりにく風", "プレーン"], value="とりにく風")
                    self.input_image_path = gr.Image(label="Input image", type='filepath')
                    self.bg_removed_image_path = gr.Image(label="Background Removed Image", type='filepath')
                    
                    # 自動背景除去トリガー
                    self.input_image_path.change(
                        fn=self.auto_background_removal,
                        inputs=[self.input_image_path],
                        outputs=[self.bg_removed_image_path]
                    )

                    self.prompt = gr.Textbox(label="Prompt", lines=3)
                    self.negative_prompt = gr.Textbox(label="Negative prompt", lines=3, value="nose, asian, realistic, lowres, error, extra digit, fewer digits, cropped, worst quality,low quality, normal quality, jpeg artifacts, blurry")
                    prompt_analysis_button = gr.Button("Prompt analysis")
                    self.controlnet_scale = gr.Slider(minimum=0.4, maximum=1.0, value=0.55, step=0.01, label="Photo fidelity")                 
                    generate_button = gr.Button(value="Generate", variant="primary")

                with gr.Column():
                    self.output_image = gr.Image(type="pil", label="Output image")

            prompt_analysis_button.click(
                fn=self.process_prompt_analysis,
                inputs=[self.bg_removed_image_path],
                outputs=self.prompt
            )

            generate_button.click(
                fn=predict,
                inputs=[self.lora_model, self.bg_removed_image_path, self.prompt, self.negative_prompt, self.controlnet_scale],
                outputs=self.output_image
            )
        return demo

    def auto_background_removal(self, input_image_path):
        if input_image_path is not None:
            bg_removed_image = background_removal(input_image_path)
            return bg_removed_image
        return None

img2img = Img2Img()
img2img.demo.queue()
img2img.demo.launch(share=True)
