"""
Simple Test Script for CAE FL System Components

Tests each component individually to verify system is working.
"""
import torch
import importlib
import sys
from data_preprocessing_cae import CAEDataPreprocessor
import config

print("="*70)
print("TESTING CONVOLUTIONAL AUTOENCODER FL SYSTEM")
print("="*70)

# Test 1: Configuration
print("\n[1/5] Testing Configuration...")
try:
    config.print_config()
    print("✓ Configuration loaded successfully")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)

# Test 2: Data Preprocessing
print("\n[2/5] Testing Data Preprocessing...")
try:
    preprocessor = CAEDataPreprocessor(
        clean_dir=config.CLEAN_IMAGE_DIR,
        noisy_dir=config.NOISY_IMAGE_DIR,
        img_size=config.AUTOENCODER_IMAGE_SIZE,
        num_clients=config.NUM_CLIENTS
    )

    # Get dataloader for client 0
    dataloader, num_samples = preprocessor.get_dataloader(
        client_id=0,
        batch_size=8,
        shuffle=True
    )

    print(f"  Client 0 has {num_samples} image pairs")
    print(f"  Number of batches: {len(dataloader)}")

    # Test one batch
    for noisy_batch, clean_batch in dataloader:
        print(f"  Batch shapes: Noisy {noisy_batch.shape}, Clean {clean_batch.shape}")
        break

    print("✓ Data preprocessing working")

except Exception as e:
    print(f"✗ Data preprocessing error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model Loading
print("\n[3/5] Testing CAE Model...")
try:
    model_module = importlib.import_module(config.MODEL_PATH)
    model_class = getattr(model_module, config.MODEL_CLASS)
    model = model_class(**config.MODEL_CONFIG)

    print(f"  Model device: {model.device}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Latent dimension: {model.latent_dim}")

    print("✓ CAE model loaded successfully")

except Exception as e:
    print(f"✗ Model loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Training Step
print("\n[4/5] Testing Training Step...")
try:
    # Get one batch
    noisy_batch, clean_batch = next(iter(dataloader))

    # Test forward pass
    print("  Running forward pass...")
    reconstructed = model.reconstruct(noisy_batch)
    print(f"  Reconstructed shape: {reconstructed.shape}")

    # Test training step
    print("  Running training step...")
    loss = model.train_step(noisy_batch, clean_batch)
    print(f"  Training loss: {loss:.6f}")

    print("✓ Training step working")

except Exception as e:
    print(f"✗ Training step error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Weight Serialization
print("\n[5/5] Testing Weight Serialization...")
try:
    # Get weights
    weights = model.get_weights()
    print(f"  Model state dict keys: {len(weights.keys())}")

    # Set weights (test FL aggregation simulation)
    model.set_weights(weights)
    print(f"  Weights restored successfully")

    # Test pickle serialization
    import pickle
    weights_bytes = pickle.dumps(weights)
    print(f"  Serialized size: {len(weights_bytes):,} bytes")

    weights_restored = pickle.loads(weights_bytes)
    model.set_weights(weights_restored)
    print("  Weights deserialized and loaded")

    print("✓ Weight serialization working")

except Exception as e:
    print(f"✗ Weight serialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Evaluation Metrics
print("\n[6/6] Testing Evaluation Metrics...")
try:
    from evaluate_cae import calculate_mse, calculate_ssim

    # Test on single image
    noisy_img = noisy_batch[0]
    clean_img = clean_batch[0]
    reconstructed_img = reconstructed[0]

    mse = calculate_mse(reconstructed_img, clean_img)
    ssim_val = calculate_ssim(reconstructed_img, clean_img)

    print(f"  MSE:  {mse:.6f}")
    print(f"  SSIM: {ssim_val:.4f}")

    print("✓ Evaluation metrics working")

except Exception as e:
    print(f"✗ Evaluation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nSystem Components Verified:")
print("  ✓ Configuration")
print("  ✓ Data Preprocessing (clean/noisy image pairs)")
print("  ✓ Convolutional Autoencoder Model")
print("  ✓ Training Step (forward + backward pass)")
print("  ✓ Weight Serialization (for FL aggregation)")
print("  ✓ Evaluation Metrics (MSE + SSIM)")
print("\n The FL-CAE system is ready for training!")
print("="*70)
