import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
import json
import glob
import random

# Import our custom modules
from model import Generator


def prepare_flat_real_images(source_dir="data/raw", dest_dir="outputs/real_eval_flat/", num_images=500):
    """
    Prepare flat directory structure for FID computation.
    
    Args:
        source_dir (str): Source directory with subfolders
        dest_dir (str): Destination flat directory
        num_images (int): Number of images to copy
        
    Returns:
        str: Path to destination directory
    """
    print(f"📁 Preparing flat real images directory...")
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Walk all subfolders recursively and collect .jpg files
    real_images = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                real_images.append(os.path.join(root, file))
    
    # Sample up to num_images
    sampled_images = random.sample(real_images, min(num_images, len(real_images)))
    
    # Copy and convert to PNG with consistent size
    for i, img_path in enumerate(sampled_images):
        # Load image
        img = Image.open(img_path).convert('L')
        
        # Resize to consistent size (128x128 for FID)
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        
        # Save as PNG
        filename = f"real_{i+1:04d}.png"
        img.save(os.path.join(dest_dir, filename))
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(sampled_images)} images...")
    
    print(f"✅ Real images prepared in {dest_dir}")
    return dest_dir


def generate_fake_images(generator, num_images=1000, latent_dim=100, device="cpu", save_dir="outputs/generated_eval/"):
    """
    Generate fake MRI images using the trained generator.
    
    Args:
        generator (nn.Module): Trained generator model
        num_images (int): Number of fake images to generate (default: 1000)
        latent_dim (int): Dimension of latent noise vector (default: 100)
        device (str): Device to run generation on (default: "cpu")
        save_dir (str): Directory to save generated images (default: "outputs/generated_eval/")
        
    Returns:
        str: Path to save directory
    """
    print(f"🎨 Generating {num_images} fake MRI images...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set generator to evaluation mode
    generator.eval()
    generator.to(device)
    
    # Generate images in batches of 64
    batch_size = 64
    generated_count = 0
    
    with torch.no_grad():
        while generated_count < num_images:
            # Calculate current batch size
            current_batch_size = min(batch_size, num_images - generated_count)
            
            # Generate random noise
            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            
            # Generate fake images
            fake_images = generator(noise)
            
            # Convert to numpy and denormalize from [-1, 1] to [0, 1]
            fake_images = fake_images.cpu().numpy()
            fake_images = (fake_images + 1) / 2.0
            fake_images = np.clip(fake_images, 0, 1)
            
            # Save each image
            for i in range(current_batch_size):
                img_array = fake_images[i, 0]  # Remove channel dimension
                img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
                
                # Save with zero-padded filename
                filename = f"fake_{generated_count + i + 1:04d}.png"
                img.save(os.path.join(save_dir, filename))
            
            generated_count += current_batch_size
            print(f"  Generated {generated_count}/{num_images} images...")
    
    print(f"✅ Fake images saved to {save_dir}")
    return save_dir


def compute_ssim_psnr(real_dir, fake_dir, num_pairs=200, image_size=128):
    """
    Compute SSIM and PSNR metrics between real and fake MRI images.
    Uses best-matching real images for each fake image (standard GAN evaluation).
    
    Args:
        real_dir (str): Directory containing real MRI images
        fake_dir (str): Directory containing fake MRI images
        num_pairs (int): Number of fake images to evaluate (default: 200)
        image_size (int): Target image size (default: 128)
        
    Returns:
        dict: Dictionary containing mean and std values for SSIM and PSNR
    """
    print(f" Computing SSIM and PSNR for {num_pairs} fake images (best-matching method)...")
    
    # Load all real images
    real_images = glob.glob(os.path.join(real_dir, "**", "*.jpg"), recursive=True)
    
    # Load fake images
    fake_images = glob.glob(os.path.join(fake_dir, "*.png"))
    fake_sample = random.sample(fake_images, min(num_pairs, len(fake_images)))
    
    # Transform for image loading
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    # Preload all real images as tensors with explicit 128x128 resize
    print("  Loading real images...")
    real_tensors = []
    for real_path in real_images:
        real_img = Image.open(real_path).convert('L')
        real_img = real_img.resize((128, 128), Image.Resampling.LANCZOS)  # Explicit resize
        real_tensor = transform(real_img)
        real_tensors.append(real_tensor)
    
    ssim_scores = []
    psnr_scores = []
    
    for i, fake_path in enumerate(fake_sample):
        # Load fake image with explicit 128x128 resize
        fake_img = Image.open(fake_path).convert('L')
        fake_img = fake_img.resize((128, 128), Image.Resampling.LANCZOS)  # Explicit resize
        fake_tensor = transform(fake_img)
        
        # Find best matching real image using MAE
        best_mae = float('inf')
        best_real_tensor = None
        
        for real_tensor in real_tensors:
            # Compute MAE
            mae = torch.mean(torch.abs(fake_tensor - real_tensor)).item()
            
            if mae < best_mae:
                best_mae = mae
                best_real_tensor = real_tensor
        
        # Convert to numpy arrays in range [0, 1]
        fake_array = fake_tensor.squeeze().numpy()
        real_array = best_real_tensor.squeeze().numpy()
        
        # Compute SSIM and PSNR
        ssim_value = ssim(real_array, fake_array, data_range=1.0)
        psnr_value = psnr(real_array, fake_array, data_range=1.0)
        
        ssim_scores.append(ssim_value)
        psnr_scores.append(psnr_value)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_pairs} fake images...")
    
    # Calculate statistics
    results = {
        "mean_ssim": float(np.mean(ssim_scores)),
        "std_ssim": float(np.std(ssim_scores)),
        "mean_psnr": float(np.mean(psnr_scores)),
        "std_psnr": float(np.std(psnr_scores))
    }
    
    print(f"✅ SSIM: {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f}")
    print(f"✅ PSNR: {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f} dB")
    
    return results


def compute_fid(real_dir, fake_dir):
    """
    Compute Fréchet Inception Distance (FID) between real and fake MRI images.
    
    Args:
        real_dir (str): Directory containing real MRI images (flat structure)
        fake_dir (str): Directory containing fake MRI images
        
    Returns:
        float: FID score as float
    """
    print(f"📊 Computing FID score...")
    
    try:
        # Compute FID using pytorch-fid with smaller batch size
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir], 
            batch_size=50,  # Smaller batch size to avoid errors
            device="cpu", 
            dims=768  # Use earlier Inception layer for small medical images
        )
        
        print(f"✅ FID Score: {fid_value:.2f}")
        return float(fid_value)
        
    except Exception as e:
        print(f"❌ Error computing FID: {e}")
        return float('inf')


def run_evaluation(generator_path="models/checkpoints/generator_final.pth", 
               real_dir="data/raw", 
               latent_dim=100, 
               image_size=128):
    """
    Run complete evaluation of the trained DCGAN model.
    
    Args:
        generator_path (str): Path to trained generator weights
        real_dir (str): Directory containing real MRI images
        latent_dim (int): Dimension of latent noise vector
        image_size (int): Target image size
    """
    print("🧠 Brain MRI DCGAN Evaluation")
    print("=" * 50)
    
    # Load trained generator
    print(f"📂 Loading generator from {generator_path}...")
    generator = Generator(latent_dim)
    generator.load_state_dict(torch.load(generator_path, map_location='cpu'))
    generator.eval()
    
    # Generate fake images
    fake_dir = generate_fake_images(
        generator=generator,
        num_images=500,
        latent_dim=latent_dim,
        device="cpu",
        save_dir="outputs/generated_eval/"
    )
    
    # Prepare flat real images directory for FID
    real_flat_dir = prepare_flat_real_images(
        source_dir=real_dir,
        dest_dir="outputs/real_eval_flat/",
        num_images=500
    )
    
    # Compute SSIM and PSNR with best-matching method
    ssim_psnr_results = compute_ssim_psnr(
        real_dir=real_dir,
        fake_dir=fake_dir,
        num_pairs=200,
        image_size=image_size
    )
    
    # Compute FID with flat directories
    fid_score = compute_fid(real_dir=real_flat_dir, fake_dir=fake_dir)
    
    # Print results
    print("\n" + "=" * 50)
    print("========== EVALUATION RESULTS ==========")
    print(f"FID Score:   {fid_score:.2f}   (target: < 50)")
    print(f"Mean SSIM:   {ssim_psnr_results['mean_ssim']:.4f}    (target: > 0.70)")
    print(f"Mean PSNR:   {ssim_psnr_results['mean_psnr']:.2f} dB (target: > 25 dB)")
    print("=" * 50)
    
    # Save results to JSON
    results = {
        "fid_score": fid_score,
        "mean_ssim": ssim_psnr_results['mean_ssim'],
        "std_ssim": ssim_psnr_results['std_ssim'],
        "mean_psnr": ssim_psnr_results['mean_psnr'],
        "std_psnr": ssim_psnr_results['std_psnr'],
        "targets": {
            "fid": "< 50 (lower is better)",
            "ssim": "> 0.70 (higher is better)",
            "psnr": "> 25 dB (higher is better)"
        }
    }
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"💾 Results saved to outputs/evaluation_results.json")
    print("✅ Evaluation complete!")
    
    return results


if __name__ == "__main__":
    """
    Main entry point for model evaluation.
    Run this script to evaluate the trained DCGAN model.
    """
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run evaluation
    run_evaluation()
