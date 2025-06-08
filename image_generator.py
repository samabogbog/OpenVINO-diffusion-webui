import os
import json
import torch
from datetime import datetime
from optimum.intel.openvino import OVDiffusionPipeline
import gc

_pipeline_cache = {}
_cached_model_name = None
_cached_width = None
_cached_height = None

def get_pipeline(model_name: str, width: int, height: int, device: str = "GPU", torch_dtype: torch.dtype = torch.float16):

    global _pipeline_cache, _cached_model_name, _cached_width, _cached_height

    if (model_name != _cached_model_name or
        width != _cached_width or
        height != _cached_height):

        if model_name in _pipeline_cache:
            del _pipeline_cache[model_name]
        gc.collect()

        _cached_model_name = model_name
        _cached_width = width
        _cached_height = height
    
    if model_name in _pipeline_cache:
        return _pipeline_cache[model_name]

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model {model_name} not found! Please ensure the model is in the 'models' directory.")

    pipeline = OVDiffusionPipeline.from_pretrained(model_path, device=device, torch_dtype=torch_dtype)
    _pipeline_cache[model_name] = pipeline
    print("Pipeline loaded successfully.")
    return pipeline

def get_models():

    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    return [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

styles = []
if os.path.exists("style/style.json"):
    with open("style/style.json", "r") as f:
        try:
            styles = json.load(f)
            print("Styles loaded from style/style.json")
        except json.JSONDecodeError:
            print("Error: Could not decode style/style.json. Please check its format.")
else:
    print("Warning: style/style.json not found. No custom styles will be available.")

def apply_style(prompt: str, style_name: str, style_weight: float, user_negative_prompt: str):

    base_negative = ""
    styled_prompt = prompt
    if style_name != "None":
        for style in styles:
            if style["name"] == style_name:
                styled_prompt = style["prompt"].format(prompt=prompt)
                styled_prompt = f"({styled_prompt}:{style_weight})"
                base_negative = style["negative_prompt"]
                break
    full_negative = ", ".join(filter(None, [base_negative, user_negative_prompt]))
    return styled_prompt, full_negative

def generate_images(model_name: str, prompt: str, style_name: str, style_weight: float, 
                    user_negative_prompt: str, width: int, height: int, 
                    num_images: int, batch_size: int, num_steps: int, guidance_scale: float):

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if width % 64 != 0 or height % 64 != 0:
        raise ValueError("Width and height must be divisible by 64! Please adjust the dimensions.")

    styled_prompt, full_negative = apply_style(prompt, style_name, style_weight, user_negative_prompt)

    pipeline = get_pipeline(model_name, width, height)
    generated_images = []

    print(f"Starting image generation: {num_images} images...")
    for i in range(0, num_images, batch_size):
        current_batch_size = min(batch_size, num_images - i)
        batch_prompts = [styled_prompt] * current_batch_size
        batch_negative_prompts = [full_negative] * current_batch_size

        with torch.no_grad():
            images = pipeline(
                batch_prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                width=width,
                height=height,
                negative_prompt=batch_negative_prompts
            ).images
        generated_images.extend(images)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    saved_files = []
    for idx, image in enumerate(generated_images):
        filename = os.path.join(output_dir, f"{timestamp}_{idx}.png")
        image.save(filename)
        saved_files.append(filename)
    print(f"{len(saved_files)} image(s) have been generated.")
    return saved_files