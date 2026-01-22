"""
Evaluation Script for Convolutional Autoencoder

Calculates:
- MSE (Mean Squared Error)
- SSIM (Structural Similarity Index)

Between reconstructed and clean images.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import importlib
import os

from data_preprocessing_cae import CAEDataPreprocessor
import config


def calculate_mse(img1, img2):
    """
    Calculate MSE between two images.

    Args:
        img1, img2: Torch tensors or numpy arrays

    Returns:
        mse: Mean squared error
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    mse = np.mean((img1 - img2) ** 2)
    return mse


def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.

    Args:
        img1, img2: Images in format (C, H, W) or (H, W, C)

    Returns:
        ssim_value: Structural similarity index
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # Convert from (C, H, W) to (H, W, C) if needed
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))

    # Calculate SSIM with multichannel support
    ssim_value = ssim(
        img1, img2,
        multichannel=True,
        channel_axis=2,
        data_range=1.0  # Images are in [0, 1] range
    )

    return ssim_value


class CAEEvaluator:
    """Evaluator for Convolutional Autoencoder"""

    def __init__(self, model_path=None):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model checkpoint (optional)
        """
        # Load model
        model_module = importlib.import_module(config.MODEL_PATH)
        model_class = getattr(model_module, config.MODEL_CLASS)
        self.model = model_class(**config.MODEL_CONFIG)

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load(model_path)

        self.device = self.model.device
        print(f"Model loaded on {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")

    def evaluate_on_dataset(self, dataloader, max_batches=None):
        """
        Evaluate model on dataset.

        Args:
            dataloader: PyTorch DataLoader with (noisy, clean) pairs
            max_batches: Maximum number of batches to evaluate (None = all)

        Returns:
            results: Dictionary with evaluation metrics
        """
        self.model.model.eval()

        total_mse = 0.0
        total_ssim = 0.0
        num_images = 0

        print("\nEvaluating model...")

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluation")

            for batch_idx, (noisy_batch, clean_batch) in enumerate(pbar):
                if max_batches and batch_idx >= max_batches:
                    break

                # Move to device
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)

                # Reconstruct
                reconstructed_batch = self.model.model(noisy_batch)

                # Calculate metrics for each image in batch
                batch_size = noisy_batch.size(0)

                for i in range(batch_size):
                    reconstructed = reconstructed_batch[i]
                    clean = clean_batch[i]

                    # MSE
                    mse = calculate_mse(reconstructed, clean)
                    total_mse += mse

                    # SSIM
                    ssim_val = calculate_ssim(reconstructed, clean)
                    total_ssim += ssim_val

                    num_images += 1

                # Update progress bar
                avg_mse = total_mse / num_images
                avg_ssim = total_ssim / num_images
                pbar.set_postfix({
                    'MSE': f'{avg_mse:.6f}',
                    'SSIM': f'{avg_ssim:.4f}'
                })

        # Calculate averages
        avg_mse = total_mse / num_images
        avg_ssim = total_ssim / num_images

        results = {
            'mse': avg_mse,
            'ssim': avg_ssim,
            'num_images': num_images
        }

        return results

    def evaluate_all_clients(self, clean_dir, noisy_dir, num_clients=3):
        """
        Evaluate on all clients' data.

        Args:
            clean_dir: Directory with clean images
            noisy_dir: Directory with noisy images
            num_clients: Number of clients

        Returns:
            all_results: Dictionary with results per client
        """
        preprocessor = CAEDataPreprocessor(
            clean_dir=clean_dir,
            noisy_dir=noisy_dir,
            img_size=config.AUTOENCODER_IMAGE_SIZE,
            num_clients=num_clients
        )

        all_results = {}

        for client_id in range(num_clients):
            print(f"\n{'='*60}")
            print(f"Evaluating Client {client_id + 1}")
            print(f"{'='*60}")

            # Get client dataloader
            dataloader, num_samples = preprocessor.get_dataloader(
                client_id=client_id,
                batch_size=config.BATCH_SIZE,
                shuffle=False
            )

            # Evaluate
            results = self.evaluate_on_dataset(dataloader)

            all_results[f'client_{client_id + 1}'] = results

            print(f"\nClient {client_id + 1} Results:")
            print(f"  Images: {results['num_images']}")
            print(f"  MSE:    {results['mse']:.6f}")
            print(f"  SSIM:   {results['ssim']:.4f}")

        # Calculate overall average
        avg_mse = np.mean([r['mse'] for r in all_results.values()])
        avg_ssim = np.mean([r['ssim'] for r in all_results.values()])
        total_images = sum([r['num_images'] for r in all_results.values()])

        all_results['overall'] = {
            'mse': avg_mse,
            'ssim': avg_ssim,
            'num_images': total_images
        }

        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Total Images: {total_images}")
        print(f"Average MSE:  {avg_mse:.6f}")
        print(f"Average SSIM: {avg_ssim:.4f}")

        return all_results

    def save_reconstructions(self, dataloader, output_dir, num_samples=10):
        """
        Save sample reconstructions.

        Args:
            dataloader: DataLoader with image pairs
            output_dir: Directory to save images
            num_samples: Number of samples to save
        """
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        self.model.model.eval()

        saved_count = 0

        with torch.no_grad():
            for noisy_batch, clean_batch in dataloader:
                if saved_count >= num_samples:
                    break

                noisy_batch = noisy_batch.to(self.device)
                reconstructed_batch = self.model.model(noisy_batch)

                # Save each image in batch
                batch_size = noisy_batch.size(0)

                for i in range(batch_size):
                    if saved_count >= num_samples:
                        break

                    # Convert to numpy for plotting
                    noisy = noisy_batch[i].cpu().permute(1, 2, 0).numpy()
                    clean = clean_batch[i].cpu().permute(1, 2, 0).numpy()
                    reconstructed = reconstructed_batch[i].cpu().permute(1, 2, 0).numpy()

                    # Create figure
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(noisy)
                    axes[0].set_title('Noisy Input')
                    axes[0].axis('off')

                    axes[1].imshow(reconstructed)
                    axes[1].set_title('Reconstructed')
                    axes[1].axis('off')

                    axes[2].imshow(clean)
                    axes[2].set_title('Clean Target')
                    axes[2].axis('off')

                    # Save
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'reconstruction_{saved_count+1}.png'))
                    plt.close()

                    saved_count += 1

        print(f"\n✓ Saved {saved_count} reconstructions to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CAE model")
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--clean-dir', type=str, default='../data/payload/images/clean',
                        help='Clean images directory')
    parser.add_argument('--noisy-dir', type=str, default='../data/payload/images/noise',
                        help='Noisy images directory')
    parser.add_argument('--num-clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--save-reconstructions', action='store_true',
                        help='Save sample reconstructions')
    parser.add_argument('--output-dir', type=str, default='./reconstructions',
                        help='Directory to save reconstructions')

    args = parser.parse_args()

    # Create evaluator
    evaluator = CAEEvaluator(model_path=args.model_path)

    # Evaluate on all clients
    results = evaluator.evaluate_all_clients(
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        num_clients=args.num_clients
    )

    # Save reconstructions if requested
    if args.save_reconstructions:
        preprocessor = CAEDataPreprocessor(
            clean_dir=args.clean_dir,
            noisy_dir=args.noisy_dir,
            img_size=config.AUTOENCODER_IMAGE_SIZE,
            num_clients=1
        )
        dataloader, _ = preprocessor.get_dataloader(client_id=0, batch_size=8, shuffle=False)
        evaluator.save_reconstructions(dataloader, args.output_dir, num_samples=10)


if __name__ == '__main__':
    main()
