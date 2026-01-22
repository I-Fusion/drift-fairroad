"""
Data Preprocessing for Convolutional Autoencoder FL

Loads clean and noisy image pairs for federated learning.
Each client gets a subset of image pairs.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CAEImageDataset(Dataset):
    """Dataset for clean/noisy image pairs"""

    def __init__(self, clean_paths, noisy_paths, transform=None):
        """
        Args:
            clean_paths: List of paths to clean images
            noisy_paths: List of paths to noisy images
            transform: Torchvision transforms
        """
        assert len(clean_paths) == len(noisy_paths), "Mismatch in clean/noisy image counts"

        self.clean_paths = clean_paths
        self.noisy_paths = noisy_paths
        self.transform = transform

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        # Load images
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        noisy_img = Image.open(self.noisy_paths[idx]).convert('RGB')

        # Apply transforms
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, clean_img  # (input, target)


class CAEDataPreprocessor:
    """Preprocessor for distributing image pairs across FL clients"""

    def __init__(self, clean_dir, noisy_dir, img_size=224, num_clients=3):
        """
        Args:
            clean_dir: Directory containing clean images
            noisy_dir: Directory containing noisy images
            img_size: Target image size (default: 224x224)
            num_clients: Number of federated clients
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.img_size = img_size
        self.num_clients = num_clients

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])

        # Load all image paths
        self.clean_paths, self.noisy_paths = self._load_image_pairs()

        logger.info(f"Loaded {len(self.clean_paths)} clean/noisy image pairs")

    def _load_image_pairs(self):
        """Load and match clean/noisy image pairs"""
        clean_paths = []
        noisy_paths = []

        # Get all clean images
        clean_files = sorted(os.listdir(self.clean_dir))
        clean_files = [f for f in clean_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for clean_file in clean_files:
            clean_path = os.path.join(self.clean_dir, clean_file)

            # Find corresponding noisy image
            # Noisy files have format: originalname.jpg_adv.png (keeps full extension)
            noisy_file = clean_file + '_adv.png'
            noisy_path = os.path.join(self.noisy_dir, noisy_file)

            if os.path.exists(noisy_path):
                clean_paths.append(clean_path)
                noisy_paths.append(noisy_path)
            else:
                logger.warning(f"No noisy pair found for {clean_file}")

        return clean_paths, noisy_paths

    def get_client_data(self, client_id):
        """
        Get data for specific client.

        Args:
            client_id: Client ID (0, 1, 2, ...)

        Returns:
            dataset: PyTorch Dataset for this client
            num_samples: Number of samples assigned to this client
        """
        total_images = len(self.clean_paths)
        images_per_client = total_images // self.num_clients

        # Calculate start and end indices for this client
        start_idx = client_id * images_per_client

        # Last client gets any remaining images
        if client_id == self.num_clients - 1:
            end_idx = total_images
        else:
            end_idx = start_idx + images_per_client

        # Get subset for this client
        client_clean_paths = self.clean_paths[start_idx:end_idx]
        client_noisy_paths = self.noisy_paths[start_idx:end_idx]

        # Create dataset
        dataset = CAEImageDataset(
            client_clean_paths,
            client_noisy_paths,
            transform=self.transform
        )

        logger.info(f"Client {client_id}: {len(dataset)} image pairs (indices {start_idx}-{end_idx})")

        return dataset, len(dataset)

    def get_dataloader(self, client_id, batch_size=32, shuffle=True):
        """
        Get DataLoader for specific client.

        Args:
            client_id: Client ID
            batch_size: Batch size for training
            shuffle: Whether to shuffle data

        Returns:
            dataloader: PyTorch DataLoader
            num_samples: Number of samples
        """
        dataset, num_samples = self.get_client_data(client_id)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Use 0 for compatibility
            pin_memory=True
        )

        return dataloader, num_samples


def test_preprocessing():
    """Test data preprocessing"""
    print("Testing CAE Data Preprocessing...")

    # Initialize preprocessor
    preprocessor = CAEDataPreprocessor(
        clean_dir='../data/payload/images/clean',
        noisy_dir='../data/payload/images/noise',
        img_size=224,
        num_clients=3
    )

    # Test client 0
    dataloader, num_samples = preprocessor.get_dataloader(client_id=0, batch_size=8)

    print(f"\nClient 0 has {num_samples} image pairs")
    print(f"Number of batches: {len(dataloader)}")

    # Get one batch
    for noisy_batch, clean_batch in dataloader:
        print(f"\nBatch shapes:")
        print(f"  Noisy: {noisy_batch.shape}")
        print(f"  Clean: {clean_batch.shape}")
        print(f"  Noisy range: [{noisy_batch.min():.3f}, {noisy_batch.max():.3f}]")
        print(f"  Clean range: [{clean_batch.min():.3f}, {clean_batch.max():.3f}]")
        break

    print("\n✓ Preprocessing test passed!")


if __name__ == '__main__':
    test_preprocessing()
