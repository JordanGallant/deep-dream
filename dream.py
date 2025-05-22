import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import os

# -------- CONFIG --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_index = 35  # VGG19 layer index to dream on (0–36 valid)
steps = 500       # Iterations
step_size = 0.03  # Learning rate
base_image = 'input.jpg'
guide_image_path = 'guide.png'  # Path to guide image

output_path = 'dreamified.jpg'
# ------------------------

# ImageNet normalization (crucial for VGG19)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Denormalization for saving
def denormalize(tensor):
    """Reverse ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

# Load image from disk and prepare for model
def load_image(path, requires_grad=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert('RGB')
    
    # Fix rotation by applying EXIF orientation tag
    image = ImageOps.exif_transpose(image)
    
    image = preprocess(image).unsqueeze(0)
    image = normalize(image)
    return image.to(device).requires_grad_(requires_grad)

# Save tensor as image
def save_image(tensor, path):
    # Denormalize first
    image = denormalize(tensor)
    image = image.detach().squeeze().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)
    print(f"Saved: {path}")

# Submodel from VGG19 up to a specific layer
def get_submodel(layer_idx=28):
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:layer_idx + 1]
    return vgg.to(device).eval()

# Core DeepDream loop with proper gradient processing
def deep_dream(image, model, guide_features=None, steps=100, lr=0.01):
    for i in range(steps):
        # Zero gradients
        if image.grad is not None:
            image.grad.zero_()
            
        # Forward pass
        features = model(image)
        
        # DeepDream objective
        if guide_features is not None:
            # Guided DeepDream: match guide image features
            loss = -torch.mean((features - guide_features) ** 2)
        else:
            # Standard DeepDream: maximize the L2 norm of activations
            loss = -torch.mean(features ** 2)
        
        # Backward pass
        loss.backward()
        
        # Normalize gradients to prevent explosion
        grad = image.grad.data
        grad = grad / (torch.std(grad) + 1e-8)  # Normalize by standard deviation
        
        # Update image
        image.data += lr * grad
        
        # Optional: clip values to reasonable range (helps stability)
        with torch.no_grad():
            # Don't clip too aggressively as it can hurt the effect
            image.data = torch.clamp(image.data, -3, 3)
        
        if i % 50 == 0:
            print(f"Step {i:03d} | Loss: {loss.item():.4f}")
    
    return image

# Multi-scale DeepDream (optional enhancement)
def deep_dream_multiscale(base_image_path, model, guide_features=None, scales=[224, 320, 448], steps=50, lr=0.01):
    """Apply DeepDream at multiple scales for better results"""
    
    # Load original image
    original = Image.open(base_image_path).convert('RGB')
    
    for scale in scales:
        print(f"\n--- Processing at scale {scale}x{scale} ---")
        
        # Resize image
        resize_transform = transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.ToTensor(),
        ])
        
        image = resize_transform(original).unsqueeze(0)
        image = normalize(image).to(device).requires_grad_(True)
        
        # Apply DeepDream
        dreamed = deep_dream(image, model, guide_features, steps, lr)
        
        # Save intermediate result
        save_image(dreamed, f'dreamed_scale_{scale}.jpg')
    
    return dreamed

# Main script
if __name__ == "__main__":
    try:
        print("Loading model...")
        model = get_submodel(layer_index)
        
        print(f"Processing image: {base_image}")
        print(f"Using layer index: {layer_index}")
        
        # Load guide image if it exists
        guide_features = None
        if os.path.exists(guide_image_path):
            print(f"Loading guide image: {guide_image_path}")
            guide_image = load_image(guide_image_path, requires_grad=False)
            with torch.no_grad():
                guide_features = model(guide_image).detach()
        else:
            print("No guide image found, using standard DeepDream")
        
        # Single scale version
        image = load_image(base_image)
        dreamed = deep_dream(image, model, guide_features, steps, step_size)
        save_image(dreamed, output_path)
        
        # Uncomment for multi-scale version (often produces better results)
        # dreamed = deep_dream_multiscale(base_image, model, guide_features, scales=[224, 320], steps=steps//2, lr=step_size)
        # save_image(dreamed, 'multiscale_' + output_path)
        
    except Exception as e:
        print("❌ Error:", e)
        import traceback
        traceback.print_exc()