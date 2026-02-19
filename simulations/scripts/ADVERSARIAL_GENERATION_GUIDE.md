# Adversarial Image Generation Guide

## Overview

This directory contains code and data for generating **adversarial examples** using the **PGD (Projected Gradient Descent)** attack on object detection models. These adversarial images are used to test the robustness of the Federated Learning Convolutional Autoencoder (FL-CAE) system.

**Purpose**: Generate noisy/adversarial images that can evade object detection while appearing visually similar to clean images. These serve as the "noisy" inputs for the CAE to denoise and reconstruct.

---

## Dataset Source: VisDrone

### About VisDrone Dataset

**Official Repository**: [https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

**VisDrone** is a large-scale benchmark dataset for drone-based computer vision tasks, collected by the AISKYEYE team at Tianjin University, China. It contains images and videos captured by various drone platforms in different scenarios.

### Dataset Characteristics

- **Domain**: Aerial/drone imagery
- **Use Cases**: Object detection, tracking, crowd counting
- **Image Quality**: High-resolution images with various viewing angles
- **Object Categories**: Pedestrians, cars, vans, trucks, buses, bicycles, motorcycles, etc.
- **Scenes**: Urban, suburban, rural environments
- **Challenges**: Small objects, occlusion, varying illumination, complex backgrounds

---

## Dataset Used in This Project

### Sample Selection

From the VisDrone dataset, **502 images** were selected from the training/validation sets for this project.

**Selection Criteria**:
- Images with clear object instances (for effective adversarial attack)
- Diverse scenes (urban, highways, parking lots, etc.)
- Varying object densities (from sparse to crowded)
- Different times of day and lighting conditions

### Image Naming Convention

Images follow VisDrone's naming format:
```
SSSSSSSS_FFFFF_t_IIIIII.jpg

Where:
- SSSSSSSS = Sequence ID (video sequence number)
- FFFFF = Frame number within sequence
- t = Image type (d = detection, v = video tracking)
- IIIIII = Unique image ID
```

**Example**: `0000002_00005_d_0000014.jpg`
- Sequence: 0000002
- Frame: 00005
- Type: Detection (d)
- ID: 0000014


## Adversarial Attack: PGD (Projected Gradient Descent)

### What is PGD Attack?

PGD is an **iterative adversarial attack** that creates imperceptible perturbations to fool machine learning models.

**Key Concept**: Add small, carefully crafted noise to an image such that:
1. The image looks almost identical to humans
2. The ML model's predictions are significantly affected (detection evasion)

### Attack Parameters

The script uses the following PGD parameters:

```python
EPSILON = 8 / 255      # Maximum perturbation: ~3.1% of pixel range
STEP_SIZE = 2 / 255    # Step size per iteration: ~0.8%
NUM_STEPS = 10         # Number of PGD iterations
```

**What this means**:
- **EPSILON (ε)**: Maximum allowed noise per pixel = 8/255 ≈ 0.031
  - Each pixel value can be perturbed by at most ±8 (on 0-255 scale)
  - Ensures perturbations are imperceptible to human eyes

- **STEP_SIZE (α)**: How much to update per iteration = 2/255 ≈ 0.008
  - Smaller steps = more precise attack, but slower

- **NUM_STEPS**: 10 iterations to refine the adversarial example
  - More iterations = stronger attack, but diminishing returns

### Attack Objective

The PGD attack in `adversarial.py` aims to **minimize detection confidence**:

```python
# Minimize detector output to evade detection
loss = -outputs.abs().sum()
```

This causes YOLO to:
- Miss detecting objects (false negatives)
- Reduce confidence scores on detected objects
- Misclassify object categories

---

## How to Use the Code

### Prerequisites

1. **Install Dependencies**
```bash
cd /Users/deeraj/Projects/github_projects/AIML_concepts/FL/payload
pip install -r requirements.txt
```

Required packages:
- `opencv-python` - Image loading and processing
- `torch` - PyTorch for deep learning
- `numpy` - Numerical operations
- `ultralytics` - YOLOv8 implementation

2. **Download VisDrone Images** (Optional - already done)

If you want to get original source images:
```bash
# Visit: https://github.com/VisDrone/VisDrone-Dataset
# Download: VisDrone2019-DET-train or VisDrone2019-DET-val
# Extract and select 502 images
# Place in images/ directory
```

The current `images/` directory already contains 502 selected images.

### Running the Script

**Basic Usage**:
```bash
python adversarial.py
```

**What it does**:
1. Loads all `.jpg`, `.jpeg`, `.png` files from `images/` directory
2. For each image:
   - Resizes to 640×640 (YOLO input size)
   - Applies PGD attack for 10 iterations
   - Saves adversarial version to `outputs/` as `{original_name}_adv.png`
3. Prints progress for each image and iteration

**Expected Output**:
```
Processing 0000002_00005_d_0000014.jpg
Generating adversarial example...
  Iteration 1/10, Loss: -1234.5678
  Iteration 2/10, Loss: -2345.6789
  ...
  Iteration 10/10, Loss: -5678.9012

Processing 0000002_00448_d_0000015.jpg
...

Done. Check outputs/
```

### Configuration

Edit the script to customize attack parameters:

```python
# Line 10-16: Basic Config
IMAGE_DIR = "images"           # Input directory
OUTPUT_DIR = "outputs"         # Output directory
IMG_SIZE = 640                 # Resize dimension

# Line 14-16: Attack Strength
EPSILON = 8 / 255              # Max perturbation (increase = stronger, more visible)
STEP_SIZE = 2 / 255            # Step size (decrease = finer, slower)
NUM_STEPS = 10                 # Iterations (increase = stronger attack)
```

**Attack Strength Guide**:
- **Weak Attack**: `EPSILON=4/255, NUM_STEPS=5` - Subtle, harder to detect
- **Medium Attack**: `EPSILON=8/255, NUM_STEPS=10` - Current setting (balanced)
- **Strong Attack**: `EPSILON=16/255, NUM_STEPS=20` - Very noticeable, high success rate

### Optional Features (Currently Commented Out)

The script includes commented-out code for visualization:

```python
# Line 109-113: Save vanilla YOLO detection
save_detection(
    img_path,
    os.path.join(OUTPUT_DIR, f"{img_name}_vanilla.jpg"),
)

# Line 131-134: Save adversarial YOLO detection
save_detection(
    adv_path,
    os.path.join(OUTPUT_DIR, f"{img_name}_adv_detected.jpg"),
)
```

**To enable**:
1. Uncomment these sections
2. Run the script
3. Get comparison visualizations:
   - `{name}_vanilla.jpg` - YOLO detection on clean image
   - `{name}_adv.png` - Adversarial image
   - `{name}_adv_detected.jpg` - YOLO detection on adversarial image

---

## Technical Details

### PGD Attack Algorithm

```python
def pgd_attack(model, image, epsilon, alpha, num_iter):
    1. Initialize: x_adv = x (clean image)

    2. For each iteration:
        a. Forward pass: outputs = model(x_adv)
        b. Compute loss: loss = -sum(|outputs|)  # Minimize detections
        c. Backward pass: grad = ∂loss/∂x_adv
        d. Update: x_adv = x_adv + α · sign(grad)
        e. Project: x_adv = clip(x_adv, x-ε, x+ε)  # Stay in ε-ball
        f. Clamp: x_adv = clip(x_adv, 0, 1)        # Valid pixel range

    3. Return: x_adv (adversarial image)
```

### Image Processing Pipeline

```
Input Image (VisDrone)
    ↓
Load & Resize (640×640)
    ↓
Normalize ([0,1])
    ↓
Convert to PyTorch Tensor
    ↓
PGD Attack (10 iterations)
    ↓
Convert back to NumPy
    ↓
Denormalize ([0,255])
    ↓
Save as PNG
```

### Model Used

- **YOLO Version**: YOLOv8 Nano (`yolov8n.pt`)
- **Input Size**: 640×640×3 (RGB)
- **Framework**: Ultralytics YOLO
- **Device**: CUDA (GPU) if available, else CPU

**Why YOLOv8 Nano?**
- Fast inference (important for 502 images)
- Good detection performance
- Widely used in research
- Lightweight (6MB model file)

---

## How This Data is Used in FL-CAE

### Workflow

1. **Adversarial Generation** (This script):
   ```
   Clean Images (images/) → PGD Attack → Adversarial Images (outputs/)
   ```

2. **Data Organization for FL-CAE**:
   ```
   outputs/ → data/payload/images/noise/    # Noisy inputs
   images/ → data/payload/images/clean/     # Clean targets
   ```

3. **FL-CAE Training**:
   ```
   Noisy Input → CAE Encoder → Latent Space → CAE Decoder → Reconstructed Output
                                                                      ↓
                                                              Compare to Clean Target
                                                                      ↓
                                                                  MSE Loss
   ```

4. **Evaluation**:
   - **MSE (Mean Squared Error)**: How close is reconstruction to clean image?
   - **SSIM (Structural Similarity)**: How perceptually similar is reconstruction?

### Dataset Partition for FL

The 502 image pairs are distributed across 3 federated clients:

- **Client 1**: Indices 0-166 (167 pairs)
- **Client 2**: Indices 167-333 (167 pairs)
- **Client 3**: Indices 334-501 (168 pairs)

Each client trains on their local data and shares only model weights (not data) with the FL server.


## File Formats

### Input Images (Clean)
- **Format**: JPEG (`.jpg`)
- **Source**: VisDrone dataset
- **Size**: Original resolution (varies, typically 1920×1080)
- **Processing**: Resized to 640×640 during attack

### Output Images (Adversarial)
- **Format**: PNG (`.png`)
- **Naming**: `{original_filename}_adv.png`
  - Example: `0000002_00005_d_0000014.jpg` → `0000002_00005_d_0000014.jpg_adv.png`
- **Size**: 640×640 (resized during attack)
- **Channels**: RGB (3 channels)

---

### Quantitative Metrics

**Perturbation Magnitude**:
```python
# L∞ norm (max perturbation per pixel)
max_perturbation = np.abs(adv_image - clean_image).max()
# Should be ≤ EPSILON (8/255 ≈ 0.031)

# L2 norm (average perturbation)
l2_perturbation = np.linalg.norm(adv_image - clean_image)
```

**Detection Drop**:
```python
# Number of detections before/after attack
num_detections_clean = len(yolo(clean_image)[0].boxes)
num_detections_adv = len(yolo(adv_image)[0].boxes)
detection_drop = (num_detections_clean - num_detections_adv) / num_detections_clean
# Higher = more successful attack
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**:
```python
device = "cpu"  # Force CPU usage
# Or reduce image size: IMG_SIZE = 416
```

**2. No Images Found**
```
Done. Check outputs/
# But outputs/ is empty
```
**Solution**: Check `IMAGE_DIR` path and ensure images have correct extensions:
```python
print(os.listdir(IMAGE_DIR))  # Debug: list files
```

**3. Slow Performance**
```
# Taking >60 sec per image
```
**Solution**:
- Use GPU if available
- Reduce `NUM_STEPS` from 10 to 5
- Check if other processes are using GPU/CPU

**4. Import Errors**
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution**:
```bash
pip install -r requirements.txt
```

---


### Dataset Licenses

VisDrone dataset is released for academic research use. Please cite the VisDrone papers if you use this dataset in your research:

```bibtex
@article{zhu2020visdrone,
  title={VisDrone-DET2019: The vision meets drone object detection in image challenge results},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and others},
  journal={Proceedings of the IEEE/CVF ICCV Workshops},
  year={2019}
}
```

---

## Advanced Usage

### Custom Attack Objectives

Modify the loss function (line 69-79) for different attack goals:

**1. Targeted Misclassification**:
```python
# Force model to predict specific class (e.g., class 5)
target_class = 5
loss = -outputs[:, :, 5+4].sum()  # Maximize class 5 confidence
```

**2. Maximize Confidence (Opposite Goal)**:
```python
# Make model overconfident (confidence attacks)
loss = outputs.abs().sum()  # Note: no negative sign
```

**3. Spatial Attacks**:
```python
# Only perturb specific image regions (e.g., top-left quadrant)
mask = torch.zeros_like(x_adv)
mask[:, :, :320, :320] = 1  # Top-left 320×320
grad_sign = grad_sign * mask
```

---
