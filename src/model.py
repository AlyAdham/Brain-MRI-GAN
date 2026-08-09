import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for WGAN-GP that creates synthetic MRI images from random noise.
    
    Takes a latent noise vector and upsamples it through a series of 
    transposed convolution layers to produce a 128x128 grayscale MRI image.
    """
    
    def __init__(self, latent_dim=100):
        """
        Initialize the Generator network.
        
        Args:
            latent_dim (int): Dimension of the input noise vector (default: 100)
        """
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            # Output: 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 1 x 128 x 128
            nn.Tanh()  # Output values in range [-1, 1]
        )
    
    def forward(self, x):
        """
        Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Input noise tensor of shape (batch_size, latent_dim, 1, 1)
            
        Returns:
            torch.Tensor: Generated image tensor of shape (batch_size, 1, 128, 128)
        """
        return self.main(x)


class Discriminator(nn.Module):
    """
    Critic network for WGAN-GP that scores images (not classifier).
    
    Takes a 128x128 grayscale MRI image and downsamples it through a series of
    convolution layers to produce an unbounded scalar score (higher = more real).
    """
    
    def __init__(self):
        """Initialize the Critic network."""
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: 1 x 128 x 128
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 32 x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Output: 1 x 1 x 1
            nn.Sigmoid()  # Output probability in range [0, 1] for DCGAN
        )
    
    def forward(self, x):
        """
        Forward pass through the critic.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, 128, 128)
            
        Returns:
            torch.Tensor: Unbounded score tensor of shape (batch_size, 1)
        """
        return self.main(x).view(-1, 1)  # Flatten to (batch_size, 1)


def weights_init(m):
    """
    Custom weights initialization for DCGAN networks.
    
    Args:
        m (nn.Module): PyTorch module to initialize
    """
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        # Initialize Conv2d and ConvTranspose2d layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm2d layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Test the Generator and Discriminator networks.
    This block will only run when the script is executed directly.
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create models
    print("Creating Generator and Discriminator...")
    generator = Generator(latent_dim=100)
    discriminator = Discriminator()
    
    # Apply custom weights initialization
    print("Applying custom weights initialization...")
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Print model parameter counts
    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)
    
    print(f"\nModel Parameter Counts:")
    print(f"Generator: {gen_params:,} trainable parameters")
    print(f"Discriminator: {disc_params:,} trainable parameters")
    print(f"Total: {gen_params + disc_params:,} trainable parameters")
    
    # Test forward passes
    print("\nTesting forward passes...")
    
    # Test Generator
    batch_size = 1
    latent_dim = 100
    noise = torch.randn(batch_size, latent_dim, 1, 1)
    
    with torch.no_grad():
        gen_output = generator(noise)
        print(f"Generator input shape: {noise.shape}")
        print(f"Generator output shape: {gen_output.shape}")
        print(f"Generator output range: [{gen_output.min():.3f}, {gen_output.max():.3f}]")
    
    # Test Discriminator
    # Use the generator output as input to discriminator
    with torch.no_grad():
        disc_output = discriminator(gen_output)
        print(f"\nDiscriminator input shape: {gen_output.shape}")
        print(f"Discriminator output shape: {disc_output.shape}")
        print(f"Discriminator output range: [{disc_output.min():.3f}, {disc_output.max():.3f}]")
    
    # Verify expected output shapes
    expected_gen_shape = torch.Size([batch_size, 1, 128, 128])
    expected_disc_shape = torch.Size([batch_size, 1])
    
    print(f"\nShape Verification:")
    print(f"Generator output correct: {gen_output.shape == expected_gen_shape}")
    print(f"Discriminator output correct: {disc_output.shape == expected_disc_shape}")
    
    if gen_output.shape == expected_gen_shape and disc_output.shape == expected_disc_shape:
        print("✅ All tests passed! WGAN-GP model architecture is correct.")
    else:
        print("❌ Tests failed! Check model architecture.")
