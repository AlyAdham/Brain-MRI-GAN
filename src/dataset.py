import torch
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from glob import glob


class BrainMRIDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading Brain MRI images.
    
    This class loads MRI images from subdirectories (glioma/, meningioma/, pituitary/, notumor/)
    and applies transformations to prepare them for GAN training.
    """
    
    def __init__(self, root_dir, image_size=64):
        """
        Initialize the BrainMRI dataset.
        
        Args:
            root_dir (str): Path to the data directory containing subfolders
            image_size (int): Target size for resizing images (default: 64)
        """
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Get all class folders
        self.classes = [d for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Ensure consistent ordering
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load all image paths and their corresponding labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            # Get all .jpg files recursively
            image_paths = glob(os.path.join(class_path, '**', '*.jpg'), recursive=True)
            
            for img_path in image_paths:
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get an image and its label by index.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (image_tensor, label_index)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')  # Convert to RGB first
        image_tensor = self.transform(image)
        
        return image_tensor, label


def get_dataloader(root_dir, image_size=64, batch_size=64, num_workers=2, pin_memory=False):
    """
    Create and return a DataLoader for the Brain MRI dataset.
    
    Args:
        root_dir (str): Path to the data directory
        image_size (int): Target size for resizing images (default: 64)
        batch_size (int): Number of samples per batch (default: 64)
        num_workers (int): Number of worker processes for data loading (default: 2)
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer (default: False)
        
    Returns:
        torch.utils.data.DataLoader: Configured dataloader
    """
    dataset = BrainMRIDataset(root_dir=root_dir, image_size=image_size)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    """
    Test the dataset and dataloader functionality.
    This block will only run when the script is executed directly.
    """
    
    # Create dataloader with test parameters
    dataloader = get_dataloader(
        root_dir="data/raw", 
        image_size=64, 
        batch_size=16
    )
    
    # Get dataset information
    dataset = dataloader.dataset
    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class names: {dataset.classes}")
    
    # Get first batch
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")  # Should be [batch_size, 1, 64, 64]
        print(f"  Labels shape: {labels.shape}")  # Should be [batch_size]
        print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Save sample images
        # Denormalize from [-1, 1] to [0, 1] for saving
        denormalized_images = (images + 1) / 2
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs/plots", exist_ok=True)
        
        # Save grid of images
        save_image(
            denormalized_images, 
            "outputs/plots/sample_real.png",
            nrow=4,  # 4x4 grid for 16 images
            normalize=False
        )
        
        print(f"Saved sample images to outputs/plots/sample_real.png")
        break  # Only process first batch for testing
