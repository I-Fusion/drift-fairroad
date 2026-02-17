"""
Convolutional Autoencoder for Federated Learning

Architecture:
- Encoder: Compresses images into latent representation
- Decoder: Reconstructs images from latent space
- Loss: MSE (Mean Squared Error)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network - compresses images to latent space"""

    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()

        # Input: (3, 224, 224)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # (32, 112, 112)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (64, 56, 56)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (128, 28, 28)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (256, 14, 14)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # (512, 7, 7)
        self.bn5 = nn.BatchNorm2d(512)

        # Flatten to latent space
        self.fc = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        # Encoder path
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Flatten and compress to latent space
        x = x.view(x.size(0), -1)
        latent = self.fc(x)

        return latent


class Decoder(nn.Module):
    """Decoder network - reconstructs images from latent space"""

    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()

        # Expand from latent space
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)

        # Decoder path (transpose convolutions)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # (256, 14, 14)
        self.bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # (128, 28, 28)
        self.bn2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # (64, 56, 56)
        self.bn3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # (32, 112, 112)
        self.bn4 = nn.BatchNorm2d(32)

        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # (3, 224, 224)

    def forward(self, latent):
        # Expand from latent space
        x = self.fc(latent)
        x = x.view(x.size(0), 512, 7, 7)

        # Decoder path
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))  # Sigmoid for pixel values in [0, 1]

        return x


class ConvolutionalAutoencoder(nn.Module):
    """Complete Convolutional Autoencoder"""

    def __init__(self, latent_dim=128):
        super(ConvolutionalAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)

    def decode(self, latent):
        """Reconstruct from latent representation"""
        return self.decoder(latent)


class CAEFederatedModel:
    """Wrapper for Convolutional Autoencoder in Federated Learning"""

    def __init__(self, latent_dim=128, learning_rate=0.001):
        """
        Initialize CAE model for FL.

        Args:
            latent_dim: Dimension of latent space
            learning_rate: Learning rate for optimizer
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        # Create model
        self.model = ConvolutionalAutoencoder(latent_dim)

        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_step(self, noisy_images, clean_images):
        """
        Single training step.

        Args:
            noisy_images: Batch of noisy input images
            clean_images: Batch of clean target images

        Returns:
            loss: Reconstruction loss value
        """
        self.model.train()

        # Move to device
        noisy_images = noisy_images.to(self.device)
        clean_images = clean_images.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        reconstructed = self.model(noisy_images)

        # Compute loss (reconstruct clean images from noisy input)
        loss = self.criterion(reconstructed, clean_images)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, noisy_images, clean_images):
        """
        Evaluate reconstruction quality.

        Args:
            noisy_images: Batch of noisy input images
            clean_images: Batch of clean target images

        Returns:
            mse_loss: Mean squared error
        """
        self.model.eval()

        with torch.no_grad():
            noisy_images = noisy_images.to(self.device)
            clean_images = clean_images.to(self.device)

            reconstructed = self.model(noisy_images)
            mse_loss = self.criterion(reconstructed, clean_images)

        return mse_loss.item()

    def reconstruct(self, noisy_images):
        """
        Reconstruct clean images from noisy input.

        Args:
            noisy_images: Batch of noisy images

        Returns:
            reconstructed: Batch of reconstructed images
        """
        self.model.eval()

        with torch.no_grad():
            noisy_images = noisy_images.to(self.device)
            reconstructed = self.model(noisy_images)

        return reconstructed

    def get_weights(self):
        """Get model state dict for federated aggregation"""
        return self.model.state_dict()

    def set_weights(self, state_dict):
        """Set model weights from federated aggregation"""
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_dim': self.latent_dim,
        }, path)

    def load(self, path):
        """Load model checkpoint (server checkpoints have only model_state_dict and round)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
