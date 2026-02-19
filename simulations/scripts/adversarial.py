import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --------------------------------
# Config
# --------------------------------
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"
IMG_SIZE = 640

EPSILON = 8 / 255
STEP_SIZE = 2 / 255
NUM_STEPS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# Load YOLOv8
# --------------------------------
yolo = YOLO("yolov8n.pt")
yolo.model.eval()
yolo.model.to(device)

# --------------------------------
# Helper functions
# --------------------------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


def save_detection(image_path, save_path):
    results = yolo(image_path, conf=0.25)
    plotted = results[0].plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, plotted)


def pgd_attack(model, image, epsilon, alpha, num_iter):
    """
    PGD attack on YOLO model
    Args:
        model: YOLO model
        image: numpy array (C, H, W) normalized [0,1]
        epsilon: maximum perturbation
        alpha: step size
        num_iter: number of iterations
    """
    # Convert to tensor
    x = torch.from_numpy(image).unsqueeze(0).to(device)
    x_adv = x.clone().detach()

    for i in range(num_iter):
        x_adv.requires_grad = True

        # Forward pass - YOLOv8 returns a list of tensors
        outputs = model(x_adv)

        # Compute loss to reduce detections
        # YOLOv8 outputs are in format [batch, num_boxes, 4+num_classes]
        # We want to minimize confidence to evade detection
        loss = 0

        if isinstance(outputs, (list, tuple)):
            for pred in outputs:
                if isinstance(pred, torch.Tensor):
                    # Sum absolute values to create gradient
                    loss -= pred.abs().sum()
        elif isinstance(outputs, torch.Tensor):
            loss -= outputs.abs().sum()

        # Backward pass
        loss.backward()

        # Get gradient sign
        grad_sign = x_adv.grad.data.sign()

        # Update adversarial example
        x_adv = x_adv.detach() + alpha * grad_sign

        # Project back to epsilon ball
        perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + perturbation, 0, 1).detach()

        print(f"  Iteration {i+1}/{num_iter}, Loss: {loss.item():.4f}")

    return x_adv.cpu().numpy()[0]


# --------------------------------
# Main loop
# --------------------------------
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    print(f"\nProcessing {img_name}")
    img_path = os.path.join(IMAGE_DIR, img_name)

    # ---- Vanilla detection
    #save_detection(
    #    img_path,
    #    os.path.join(OUTPUT_DIR, f"{img_name}_vanilla.jpg"),
    #)

    # ---- Load image for attack
    x = load_image(img_path)

    # ---- Generate adversarial example
    print("Generating adversarial example...")
    x_adv = pgd_attack(yolo.model, x, EPSILON, STEP_SIZE, NUM_STEPS)

    # ---- Save adversarial image
    adv_img = (
        np.transpose(x_adv, (1, 2, 0)) * 255
    ).astype(np.uint8)

    adv_path = os.path.join(OUTPUT_DIR, f"{img_name}_adv.png")
    cv2.imwrite(adv_path, cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR))

    # ---- Detection on adversarial image
    #save_detection(
    #    adv_path,
    #    os.path.join(OUTPUT_DIR, f"{img_name}_adv_detected.jpg"),
    #)

print("\nDone. Check outputs/")
