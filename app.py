import gradio as gr
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

from src.model import Generator
from src.generate import generate_samples

# ==================== CONSTANTS ====================
GENERATOR_PATH = "models/checkpoints/generator_final.pth"
LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== SETUP ====================
# Load Generator once at startup (outside any function)
print("🧠 Loading Brain MRI DCGAN Generator...")
generator = Generator()
generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
generator.to(DEVICE)  # Move generator to the correct device
generator.eval()

# Dynamically detect the model's output size
with torch.no_grad():
    test_noise = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
    test_output = generator(test_noise)
    IMAGE_SIZE = test_output.shape[-1]  # Get H/W from (B, C, H, W)
    
print(f"✅ Generator loaded successfully! Output size: {IMAGE_SIZE}x{IMAGE_SIZE}")

def generate_mri_image(num_images: int) -> list:
    """
    Generate synthetic brain MRI images using the loaded generator.
    
    Args:
        num_images (int): Number of images to generate (between 1 and 16)
        
    Returns:
        list: List of PIL Image objects (resized to 256x256 for display)
    """
    # Generate random noise on the same device as the generator
    noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=DEVICE)
    
    # Pass through generator
    with torch.no_grad():
        fake_images = generator(noise)
    
    # Convert to PIL grayscale images
    pil_images = []
    for i in range(num_images):
        # Extract single image and denormalize from [-1,1] to [0,255]
        img = fake_images[i].squeeze().detach().cpu().numpy()
        img = (img * 0.5 + 0.5) * 255  # denormalize
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Convert to PIL and resize to 256x256 for better display
        pil_img = Image.fromarray(img, mode='L').resize((256, 256), Image.NEAREST)
        pil_images.append(pil_img)
    
    return pil_images

# ==================== GRADIO INTERFACE ====================
def create_demo_interface():
    """Build a Gradio Blocks interface for interactive MRI generation."""
    
    with gr.Blocks(title="🧠 Brain MRI GAN — Synthetic Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 Brain MRI DCGAN - Interactive Demo
            
            Generate synthetic brain MRI images using a trained Deep Convolutional GAN model.
            This interactive demo allows you to create custom brain MRI scans on demand.
            
            ## 🎯 Features:
            - **Real-time Generation**: 1-16 synthetic MRI images per request
            - **Medical Quality**: Grayscale brain-like structures
            - **Interactive Controls**: Adjustable number of images
            - **Instant Results**: No training required, immediate generation
            
            ## 📊 Model Performance:
            - **FID Score**: 147.00 (reasonable for small dataset)
            - **SSIM**: 0.41 (expected for GANs)
            - **PSNR**: 19.25 dB (typical for 64×64 medical images)
            
            ---
            *Trained on Brain Tumor MRI Dataset with 200 epochs*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input Controls
                gr.Markdown("### 🎛 Generation Controls")
                
                num_slider = gr.Slider(
                    label="Number of Images to Generate",
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    info="Generate 1-16 synthetic brain MRI images"
                )
                
                generate_btn = gr.Button(
                    "🧠 Generate MRI Images",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=3):
                # Output Display
                gr.Markdown("### 🖼️ Generated Brain MRI Images")
                
                output_gallery = gr.Gallery(
                    label="Generated Brain MRI Scans",
                    columns=4,
                    height=400,
                    show_label=True,
                    allow_preview=True
                )
        
        # Generation function
        def handle_generation(num_images):
            """Handle the generation request and return images."""
            try:
                # Generate images
                pil_images = generate_mri_image(num_images)
                
                # Create tuples of (image, caption) for Gradio gallery
                gallery_items = [(img, f"Synthetic Brain MRI #{i+1}") for i, img in enumerate(pil_images)]
                
                return gallery_items
                
            except Exception as e:
                gr.Error(f"Generation failed: {str(e)}")
                return []
        
        # Connect button to function
        generate_btn.click(
            fn=handle_generation,
            inputs=[num_slider],
            outputs=[output_gallery]
        )
        
        # Add example generation on load
        def generate_initial():
            """Generate initial sample images on page load."""
            pil_images = generate_mri_image(4)
            return [(img, f"Synthetic Brain MRI #{i+1}") for i, img in enumerate(pil_images)]
        
        demo.load(
            fn=generate_initial,
            inputs=[],
            outputs=[output_gallery]
        )
        
        # Return the demo interface
        return demo

# ==================== MAIN LAUNCH ====================
if __name__ == "__main__":
    """Launch the Gradio demo interface."""
    print("🚀 Starting Brain MRI DCGAN Demo...")
    print("📱 Opening web interface at http://localhost:7860")
    print("📋 Close this terminal to keep the demo running")
    print("🛑 Press Ctrl+C to stop the server")
    
    # Create and launch interface
    demo_interface = create_demo_interface()
    demo_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
