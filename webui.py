from gradio_ui import create_gradio_interface

if __name__ == "__main__":
    print("Starting OpenVINO Diffusion Web UI...")
    demo = create_gradio_interface()
    
    demo.launch(
        share=False,
        inbrowser=True
    )
