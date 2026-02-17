"""
Small Convolutional Autoencoder for Federated Learning (for testing / debugging).

Same interface as cae_model.py but with fewer channels and smaller latent dim
to reduce payload size (~2–3 MB vs ~48 MB) and avoid transfer timeouts.
Input: (3, 224, 224) unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderSmall(nn.Module):
    """Small encoder: 3 -> 8 -> 16 -> 32 -> 64 -> 128, then fc to latent."""

    def __init__(self, latent_dim=32):
        super(EncoderSmall, self).__init__()
        # (3,224,224) -> (8,112,112) -> (16,56,56) -> (32,28,28) -> (64,14,14) -> (128,7,7)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DecoderSmall(nn.Module):
    """Small decoder: fc then 128 -> 64 -> 32 -> 16 -> 8 -> 3."""

    def __init__(self, latent_dim=32):
        super(DecoderSmall, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, latent):
        x = self.fc(latent)
        x = x.view(x.size(0), 128, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        return x


class ConvolutionalAutoencoderSmall(nn.Module):
    """Small CAE: same input/output shape as full model, fewer parameters."""

    def __init__(self, latent_dim=32):
        super(ConvolutionalAutoencoderSmall, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = EncoderSmall(latent_dim)
        self.decoder = DecoderSmall(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)


class CAEFederatedModel:
    """
    Wrapper for small CAE in Federated Learning.
    Same API as cae_model.CAEFederatedModel for drop-in testing.
    """

    def __init__(self, latent_dim=32, learning_rate=0.001):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self.model = ConvolutionalAutoencoderSmall(latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_step(self, noisy_images, clean_images):
        self.model.train()
        noisy_images = noisy_images.to(self.device)
        clean_images = clean_images.to(self.device)
        self.optimizer.zero_grad()
        reconstructed = self.model(noisy_images)
        loss = self.criterion(reconstructed, clean_images)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, noisy_images, clean_images):
        self.model.eval()
        with torch.no_grad():
            noisy_images = noisy_images.to(self.device)
            clean_images = clean_images.to(self.device)
            reconstructed = self.model(noisy_images)
            mse_loss = self.criterion(reconstructed, clean_images)
        return mse_loss.item()

    def reconstruct(self, noisy_images):
        self.model.eval()
        with torch.no_grad():
            noisy_images = noisy_images.to(self.device)
            reconstructed = self.model(noisy_images)
        return reconstructed

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_dim': self.latent_dim,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
