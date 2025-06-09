import os
import json
import torch
from datetime import datetime
from optimum.intel.openvino import OVDiffusionPipeline
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DDPMScheduler,
    KDPM2DiscreteScheduler
)
import gc

_pipeline_cache = {}
_cached_model_name = None
_cached_width = None
_cached_height = None
_cached_sampler = None

SAMPLER_MAP = {
    "DPM++ 2M Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True),
    "DPM Solver Multistep": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "DDPM": DDPMScheduler,
    "KDPM2 Discrete": KDPM2DiscreteScheduler,
}

def get_pipeline(model_name: str, width: int, height: int, sampler_name: str, device: str = "GPU", torch_dtype: torch.dtype = torch.float32):
    global _pipeline_cache, _cached_model_name, _cached_width, _cached_height, _cached_sampler

    if model_name != _cached_model_name:
        _pipeline_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    if (model_name != _cached_model_name or
        width != _cached_width or
        height != _cached_height or
        sampler_name != _cached_sampler):
        
        if _cached_model_name is not None and model_name == _cached_model_name:
            print(f"Pipeline parameters changed for model {model_name}. Re-initializing pipeline.")
            
        _cached_model_name = model_name
        _cached_width = width
        _cached_height = height
        _cached_sampler = sampler_name
    
    if model_name in _pipeline_cache:
        return _pipeline_cache[model_name]

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model {model_name} not found! Please ensure the model is in the 'models' directory.")
    
    if sampler_name not in SAMPLER_MAP:
        raise ValueError(f"Invalid sampler name: {sampler_name}. Available samplers: {list(SAMPLER_MAP.keys())}")
    
    pipeline = OVDiffusionPipeline.from_pretrained(model_path, device=device, torch_dtype=torch_dtype)

    try:
        scheduler_config = DPMSolverMultistepScheduler.load_config(os.path.join(model_path, "scheduler"))
        
        sampler_constructor = SAMPLER_MAP[sampler_name]
        
        if callable(sampler_constructor):
            scheduler = sampler_constructor(scheduler_config)
        else:
            scheduler = sampler_constructor.from_config(scheduler_config)
            
        pipeline.scheduler = scheduler
        print(f"Successfully set {sampler_name} as the scheduler.")
    except Exception as e:
        print(f"Warning: Could not set {sampler_name} scheduler. Falling back to default. Error: {e}")

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
        except json.JSONDecodeError:
            print("Error: Could not decode style/style.json. Please check its format.")
else:
    print("Warning: style/style.json not found. No custom styles will be available.")

def apply_style(prompt: str, style_name: str, user_negative_prompt: str):

    base_negative = ""
    styled_prompt = prompt
    if style_name != "None":
        for style in styles:
            if style["name"] == style_name:
                styled_prompt = style["prompt"].format(prompt=prompt)

                base_negative = style["negative_prompt"]
                break
    full_negative = ", ".join(filter(None, [base_negative, user_negative_prompt]))
    return styled_prompt, full_negative

def generate_images(model_name: str, prompt: str, style_name: str, 
                    user_negative_prompt: str, width: int, height: int, 
                    num_images: int, batch_size: int, num_steps: int, 
                    guidance_scale: float, sampler_name: str):

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if width % 64 != 0 or height % 64 != 0:
        raise ValueError("Width and height must be divisible by 64! Please adjust the dimensions.")

    styled_prompt, full_negative = apply_style(prompt, style_name, user_negative_prompt) 

    pipeline = get_pipeline(model_name, width, height, sampler_name)
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