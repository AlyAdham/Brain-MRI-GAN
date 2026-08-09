import torch
import os
import numpy as np
from torchvision.utils import save_image
from PIL import Image

# Import our custom modules
from src.model import Generator


def generate_samples(
    generator_path="models/checkpoints/generator_final.pth",
    num_images=16,
    latent_dim=100,
    image_size=128,
    save_dir="outputs/generated/",
    return_tensors=False
):
    """
    Generate synthetic brain MRI images using trained generator.
    
    Args:
        generator_path (str): Path to trained generator weights
        num_images (int): Number of images to generate (default: 16)
        latent_dim (int): Dimension of latent noise vector (default: 100)
        image_size (int): Target image size (default: 64)
        save_dir (str): Directory to save generated images (default: "outputs/generated/")
        return_tensors (bool): Whether to return tensor data (default: False)
        
    Returns:
        None or torch.Tensor: Generated images tensor if return_tensors=True
    """
    print(f"🧠 Generating {num_images} synthetic brain MRI images...")
    
    # 1. Load Generator from src.model
    print(f"📂 Loading generator from {generator_path}...")
    generator = Generator(latent_dim)
    generator.load_state_dict(torch.load(generator_path, map_location='cpu'))
    generator.eval()
    
    # 2. Generate random noise tensor
    print(f"🎲 Creating random noise vectors...")
    noise = torch.randn(num_images, latent_dim, 1, 1)
    
    # 3. Pass through generator to get fake images
    print(f"🎨 Generating images from noise...")
    with torch.no_grad():
        fake_images = generator(noise)
    
    # 4. Denormalize from [-1, 1] to [0, 1] range
    print(f"🔧 Denormalizing images...")
    fake_images = (fake_images + 1) / 2.0
    fake_images = torch.clamp(fake_images, 0, 1)
    
    # 5. Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 6. Save individual images
    print(f"💾 Saving individual images...")
    for i in range(num_images):
        # Extract single image
        img_tensor = fake_images[i]
        
        # Save as individual file
        filename = f"individual_{i+1:04d}.png"
        save_image(img_tensor, os.path.join(save_dir, filename), normalize=True)
    
    # 7. Save grid of all images
    print(f"🖼️  Creating image grid...")
    nrow = 4  # 4x4 grid for 16 images
    grid_filename = "generated_grid.png"
    save_image(fake_images, os.path.join(save_dir, grid_filename), nrow=nrow, normalize=True)
    
    # 8. Print completion message
    print(f"✅ Generated {num_images} brain MRI images saved to {save_dir}")
    print(f"   • Individual images: individual_0001.png to individual_{num_images:04d}.png")
    print(f"   • Grid image: {grid_filename}")
    
    # 9. Return tensors if requested (for app.py integration)
    if return_tensors:
        return fake_images
    else:
        return None


if __name__ == "__main__":
    """
    Main entry point for on-demand brain MRI generation.
    Run this script to generate new synthetic brain MRI images.
    """
    print("🧠 Brain MRI DCGAN - On-Demand Generation")
    print("=" * 60)
    
    # Generate samples with default parameters
    generate_samples()
    
    print("\n" + "=" * 60)
    print("✅ Generation complete!")
    print("📁 Check outputs/generated/ for your new brain MRI images")
