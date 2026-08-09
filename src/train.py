import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import random

# Import our custom modules
from model import Generator, Discriminator, weights_init
from dataset import get_dataloader

# ==================== HYPERPARAMETERS ====================
DATA_DIR = "data/raw"
IMAGE_SIZE = 128
BATCH_SIZE = 32  # 128x128 images need smaller batch
LATENT_DIM = 100
NUM_EPOCHS = 200
LR = 0.0002
BETA1 = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "models/checkpoints"
GENERATED_DIR = "outputs/generated"
FIXED_NOISE_PATH = "outputs/fixed_noise.pt"

# RTX 5070 optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)


def save_loss_plot(G_losses, D_losses):
    """
    Save the training loss curves to a plot.
    
    Args:
        G_losses (list): List of generator losses
        D_losses (list): List of discriminator losses
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G Loss")
    plt.plot(D_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/plots/loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


def train():
    """
    Main training function for the DCGAN.
    Trains the Generator and Discriminator on the Brain MRI dataset.
    """
    
    # Create necessary directories
    create_directories()
    
    # Print training information
    print("=" * 60)
    print("🧠 Brain MRI DCGAN Training")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"📊 Batch Size: {BATCH_SIZE}")
    print(f"🖼️  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"🎯 Latent Dimension: {LATENT_DIM}")
    print(f"📚 Epochs: {NUM_EPOCHS}")
    print(f"⚡ Learning Rate: {LR}")
    print("=" * 60)
    
    # 1. Create dataloader
    print("📂 Loading dataset...")
    dataloader = get_dataloader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, num_workers=6, pin_memory=True)
    print(f"✅ Dataset loaded: {len(dataloader)} batches")
    
    # 2. Initialize models
    print("🏗️  Initializing models...")
    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Load from epoch 120 checkpoint to continue training
    print("📂 Loading checkpoint from epoch 120...")
    generator.load_state_dict(torch.load('models/checkpoints/generator_epoch_120.pth', map_location=DEVICE))
    discriminator.load_state_dict(torch.load('models/checkpoints/discriminator_epoch_120.pth', map_location=DEVICE))
    print("✅ Checkpoint loaded, continuing training...")
    
    # Apply weight initialization (skip since we're loading pretrained)
    # generator.apply(weights_init)
    # discriminator.apply(weights_init)
    
    # Print model info
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"📊 Generator parameters: {gen_params:,}")
    print(f"📊 Discriminator parameters: {disc_params:,}")
    
    # 3. Define loss function - use BCELoss for DCGAN
    criterion = nn.BCELoss()
    
    # 4. Define optimizers for DCGAN
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    # 5. Create fixed noise for consistent image generation across epochs
    print("🎲 Creating fixed noise for consistent visualization...")
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)
    torch.save(fixed_noise, FIXED_NOISE_PATH)
    print(f"✅ Fixed noise saved to {FIXED_NOISE_PATH}")
    
    # 6. Initialize loss lists
    G_losses = []
    D_losses = []
    
    # 7. Main training loop for DCGAN
    print("🚀 Starting DCGAN training...")
    print("-" * 60)
    
    START_EPOCH = 120  # Continue from checkpoint
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        # Progress bar for this epoch
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                   desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for i, (real_images, _) in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(DEVICE)
            
            # Create soft labels with one-sided label smoothing
            # Real labels: uniform(0.85, 1.0) instead of hard 1.0
            # Fake labels: uniform(0.0, 0.15) instead of hard 0.0
            real_labels = torch.FloatTensor(batch_size, 1).uniform_(0.85, 1.0).to(DEVICE)
            fake_labels = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.15).to(DEVICE)
            
            # 10% probability: flip labels for discriminator only (not generator)
            if random.random() < 0.10:
                real_labels, fake_labels = fake_labels, real_labels
            
            # ==================== TRAIN DISCRIMINATOR ====================
            
            # a. Zero discriminator gradients
            discriminator.zero_grad()
            
            # b. Train with real images
            output_real = discriminator(real_images).view(-1, 1)
            loss_D_real = criterion(output_real, real_labels)
            
            # c. Train with fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = generator(noise).detach()  # .detach() so gradients don't flow to G
            output_fake = discriminator(fake_images).view(-1, 1)
            loss_D_fake = criterion(output_fake, fake_labels)
            
            # d. Total discriminator loss
            loss_D = loss_D_real + loss_D_fake
            
            # e. Backpropagation and optimizer step
            loss_D.backward()
            optimizer_D.step()
            
            # ==================== TRAIN GENERATOR ====================
            
            # f. Zero generator gradients
            generator.zero_grad()
            
            # g. Generate fake images (do NOT detach this time)
            fake_images = generator(noise)
            output = discriminator(fake_images).view(-1, 1)
            loss_G = criterion(output, real_labels)  # Generator wants discriminator to say these are real
            
            # h. Backpropagation and optimizer step
            loss_G.backward()
            optimizer_G.step()
            
            # i. Store losses
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            
            # Update progress bar with DCGAN metrics
            pbar.set_postfix({
                'D_Loss': f'{loss_D.item():.4f}',
                'G_Loss': f'{loss_G.item():.4f}',
                'D_Real': f'{output_real.mean().item():.3f}',
                'D_Fake': f'{output_fake.mean().item():.3f}'
            })
        
        # 8. End of epoch operations
        actual_epoch = epoch + 1
        print(f"Epoch [{actual_epoch}/{NUM_EPOCHS}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
        
        # Generate and save sample images
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
            save_image(fake_images, 
                      f"{GENERATED_DIR}/epoch_{actual_epoch:03d}.png",
                      nrow=8, normalize=True)
        
        # Save checkpoints every 10 epochs
        if actual_epoch % 10 == 0:
            torch.save(generator.state_dict(), 
                      f"{CHECKPOINT_DIR}/generator_epoch_{actual_epoch}.pth")
            torch.save(discriminator.state_dict(), 
                      f"{CHECKPOINT_DIR}/discriminator_epoch_{actual_epoch}.pth")
            print(f"💾 Checkpoint saved at epoch {actual_epoch}")
    
    # 9. Training completed
    print("-" * 60)
    print("🎉 Training completed!")
    
    # Plot and save loss curves
    print("📊 Saving loss curves...")
    save_loss_plot(G_losses, D_losses)
    
    # Save final models
    print("💾 Saving final models...")
    torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_final.pth")
    
    print("✅ Training complete! Models and plots saved.")
    print(f"📁 Generated images: {GENERATED_DIR}/")
    print(f"📁 Model checkpoints: {CHECKPOINT_DIR}/")
    print(f"📁 Loss plot: outputs/plots/loss_curve.png")


if __name__ == "__main__":
    """
    Main entry point for training.
    Run this script to start the DCGAN training process.
    """
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Start training
    train()
