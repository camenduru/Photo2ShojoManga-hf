import spaces
import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from PIL import Image
import os
import time

from utils.dl_utils import dl_cn_model, dl_cn_config, dl_tagger_model, dl_lora_model
from utils.image_utils import resize_image_aspect_ratio, base_generation

from utils.prompt_utils import execute_prompt, remove_color, remove_duplicates
from utils.tagger import modelLoad, analysis

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

def load_model(lora_dir, cn_dir):
    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype, use_safetensors=True)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(lora_dir, weight_name="syoujomannga_line.safetensors")
    return pipe

@spaces.GPU(duration=120)
def predict(input_image_path, prompt, negative_prompt, controlnet_scale):
    pipe = load_model(lora_dir, cn_dir) 
    input_image = Image.open(input_image_path)
    base_image = base_generation(input_image.size, (255, 255, 255, 255)).convert("RGB")
    resize_image = resize_image_aspect_ratio(input_image)
    resize_base_image = resize_image_aspect_ratio(base_image)
    generator = torch.manual_seed(0)
    last_time = time.time()
    prompt = "masterpiece, best quality, monochrome, greyscale, lineart, white background, " + prompt
    execute_tags = ["realistic", "nose", "asian"]
    prompt = execute_prompt(execute_tags, prompt)
    prompt = remove_duplicates(prompt)        
    prompt = remove_color(prompt)
    print(prompt)

    output_image = pipe(
        image=resize_base_image,
        control_image=resize_image,
        strength=1.0,
        prompt=prompt,
        negative_prompt = negative_prompt,
        controlnet_conditioning_scale=float(controlnet_scale),
        generator=generator,
        num_inference_steps=30,
        eta=1.0,
    ).images[0]
    print(f"Time taken: {time.time() - last_time}")
    output_image = output_image.resize(input_image.size, Image.LANCZOS)
    return output_image

class Img2Img:
    def __init__(self):
        self.demo = self.layout()
        self.tagger_model = None
        self.input_image_path = None
        self.canny_image = None


    def process_prompt_analysis(self, input_image_path):
        if self.tagger_model is None:
            self.tagger_model = modelLoad(tagger_dir)
        tags = analysis(input_image_path, tagger_dir, self.tagger_model)
        tags_list = remove_color(tags)
        return tags_list


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
                    self.input_image_path = gr.Image(label="Input image", type='filepath')
                    self.prompt = gr.Textbox(label="Prompt", lines=3)
                    self.negative_prompt = gr.Textbox(label="Negative prompt", lines=3, value="sketch, lowres, error, extra digit, fewer digits, cropped, worst quality,low quality, normal quality, jpeg artifacts, blurry")
                    prompt_analysis_button = gr.Button("Prompt analysis")
                    self.controlnet_scale = gr.Slider(minimum=0.5, maximum=1.25, value=1.0, step=0.01, label="Lineart fidelity")                 
                    generate_button = gr.Button(value="Generate", variant="primary")
                with gr.Column():
                    self.output_image = gr.Image(type="pil", label="Output image")

            prompt_analysis_button.click(
                        self.process_prompt_analysis,
                        inputs=[self.input_image_path],
                        outputs=self.prompt
            )

            generate_button.click(
                fn=predict,
                inputs=[self.input_image_path, self.prompt, self.negative_prompt, self.controlnet_scale],
                outputs=self.output_image
            )
        return demo

img2img = Img2Img()
img2img.demo.queue()
img2img.demo.launch(share=True)
