import gradio as gr
from image_generator import get_models, generate_images, styles, SAMPLER_MAP

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎨 OpenVINO Diffusion Web UI")
        
        available_models = get_models()
        default_model = available_models[0] if available_models else None

        style_choices = ["None"] + [s["name"] for s in styles]
        sampler_choices = list(SAMPLER_MAP.keys())
        default_sampler = "DPM++ 2M Karras"

        with gr.Row():
            with gr.Column(scale=3):
                model_name = gr.Dropdown(
                    label="🤖 Model",
                    choices=available_models,
                    value=default_model,
                    interactive=True
                )
                prompt = gr.Textbox(label="📝 Prompt", lines=3, placeholder="e.g., A futuristic city at sunset")
                user_negative_prompt = gr.Textbox(
                    label="🚫 Negative Prompt",
                    placeholder="e.g., low quality, blurry, deformed, bad anatomy",
                    lines=1
                )
                with gr.Row():
                    style_name = gr.Dropdown(
                        label="🎨 Style",
                        choices=style_choices,
                        value="None"
                    )
                    sampler_name = gr.Dropdown(
                        label="🎲 Sampler",
                        choices=sampler_choices,
                        value=default_sampler,
                        interactive=True
                    )
                
                guidance_scale = gr.Slider(label="🧭 Guidance Scale", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                
                with gr.Row():
                    width = gr.Slider(label="📏 Width", minimum=256, maximum=1280, value=1024, step=64)
                    height = gr.Slider(label="📐 Height", minimum=256, maximum=1280, value=1024, step=64)
                with gr.Row():
                    num_images = gr.Slider(label="🖼️ Number of Images", minimum=1, maximum=4, value=2, step=1)
                    batch_size = gr.Slider(label="⚡ Batch Size", minimum=1, maximum=4, value=2, step=1)
                with gr.Row():
                    num_steps = gr.Slider(label="🔢 Inference Steps", minimum=1, maximum=50, value=30, step=1)
                    
                generate_btn = gr.Button("🚀 Generate", variant="primary")
            with gr.Column(scale=4):
                gallery = gr.Gallery(label="🖼️ Generated Images", columns=2, preview=True, height=720)
                status = gr.Textbox(label="📊 Status")

        def run_generation_wrapper(model_name_val, prompt_val, style_name_val, 
                                 user_negative_prompt_val, width_val, height_val, 
                                 num_images_val, batch_size_val, num_steps_val, 
                                 guidance_scale_val, sampler_name_val):
            try:
                if not model_name_val:
                    raise gr.Error("Please select a model from the dropdown!")
                
                files = generate_images(
                    model_name_val, prompt_val, style_name_val, 
                    user_negative_prompt_val, width_val, height_val, 
                    num_images_val, batch_size_val, num_steps_val, 
                    guidance_scale_val, sampler_name_val
                )
                return files, "✅ Generation completed successfully!"
            except gr.Error as ge:
                print(f"Gradio Error: {ge}")
                return [], str(ge)
            except Exception as e:
                print(f"An unexpected error occurred during generation: {e}")
                return [], f"❌ Error: {str(e)}"

        generate_btn.click(
            fn=run_generation_wrapper,
            inputs=[
                model_name, prompt, style_name, user_negative_prompt, 
                width, height, num_images, batch_size, num_steps, 
                guidance_scale, sampler_name
            ],
            outputs=[gallery, status]
        )
    return demo

if __name__ == "__main__":
    print("Launching Gradio UI directly for testing...")
    ui_demo = create_gradio_interface()
    ui_demo.launch()