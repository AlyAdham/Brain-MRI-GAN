import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

def load_and_display_images(image_paths, title, max_cols=5):
    """
    Load and display a grid of images.
    
    Args:
        image_paths (list): List of image file paths
        title (str): Title for the plot
        max_cols (int): Maximum number of columns in grid
    """
    if not image_paths:
        print(f"No images found for {title}")
        return
    
    # Calculate grid dimensions
    n_images = len(image_paths)
    n_cols = min(max_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Load and display images
    for i, img_path in enumerate(image_paths):
        if i >= n_rows * n_cols:
            break
            
        row = i // n_cols
        col = i % n_cols
        
        # Load image
        img = Image.open(img_path).convert('L')
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Display
        axes[row, col].imshow(img_array, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        
        # Add filename
        filename = os.path.basename(img_path)
        axes[row, col].set_title(filename[:10], fontsize=8)
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_training_progress():
    """Analyze training progress by showing images from different epochs."""
    print("🔍 Analyzing Training Progress...")
    
    # Get training images
    train_dir = "outputs/generated"
    train_images = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.png')])
    
    if len(train_images) < 10:
        print("Not enough training images found")
        return
    
    # Sample images from different epochs
    epoch_samples = []
    epochs_to_check = [1, 10, 25, 50, 75, 100]
    
    for epoch in epochs_to_check:
        epoch_file = f"epoch_{epoch:03d}.png"
        if epoch_file in [os.path.basename(f) for f in train_images]:
            epoch_samples.append([f for f in train_images if epoch_file in f][0])
    
    if epoch_samples:
        load_and_display_images(epoch_samples, "Training Progress (Selected Epochs)")
    else:
        print("No epoch images found")

def analyze_evaluation_samples():
    """Analyze evaluation samples to check quality."""
    print("\n🔍 Analyzing Evaluation Samples...")
    
    # Get evaluation images
    eval_dir = "outputs/generated_eval"
    eval_images = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('.png')]
    
    if len(eval_images) < 10:
        print("Not enough evaluation images found")
        return
    
    # Random sample for visualization
    random.seed(42)  # For reproducibility
    sample_images = random.sample(eval_images, min(15, len(eval_images)))
    
    load_and_display_images(sample_images, "Generated MRI Samples (Random Selection)")

def analyze_real_samples():
    """Show some real MRI images for comparison."""
    print("\n🔍 Analyzing Real MRI Samples...")
    
    # Get real images
    real_dir = "data/raw"
    real_images = []
    
    # Walk through subdirectories
    for root, dirs, files in os.walk(real_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                real_images.append(os.path.join(root, file))
    
    if len(real_images) < 10:
        print("Not enough real images found")
        return
    
    # Random sample for visualization
    random.seed(42)  # For reproducibility
    sample_images = random.sample(real_images, min(15, len(real_images)))
    
    load_and_display_images(sample_images, "Real MRI Samples (Random Selection)")

def main():
    """Main function to visualize results."""
    print("🧠 Brain MRI DCGAN Results Visualization")
    print("=" * 60)
    
    # Set random seed for consistent sampling
    random.seed(42)
    
    # Analyze different aspects
    analyze_real_samples()
    analyze_training_progress()
    analyze_evaluation_samples()
    
    print("\n" + "=" * 60)
    print("✅ Visualization complete!")
    print("\n📋 Analysis Guide:")
    print("• Compare real vs generated images for structural similarity")
    print("• Look for brain-like features: circular shapes, tissue texture, contrast")
    print("• Training progress should show improvement from early to late epochs")
    print("• Generated images should have consistent grayscale values")
    print("• Check for mode collapse (all images looking too similar)")

if __name__ == "__main__":
    main()
