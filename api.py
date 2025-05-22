from flask import Flask, request, jsonify
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import threading
import time
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import tempfile


app = Flask(__name__)

# Email configuration
SMTP_SERVER = "smtp.gmail.com"  # Change based on your email provider
SMTP_PORT = 587
EMAIL_ADDRESS = "jordab.gallant.v9p@gmail.com" # Your email address
EMAIL_PASSWORD = "crljxbhzpacemzqp"    # Your email app password

# DeepDream configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        intermediate_path = f'dreamed_scale_{scale}_{int(time.time())}.jpg'
        save_image(dreamed, intermediate_path)
    
    return dreamed

def send_email_with_image(recipient_email, image_path, subject="DeepDream Processing Complete"):
    """Send email with processed DeepDream image as attachment"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Email body
        body = f"""
        Your DeepDream image processing has been completed successfully!
        
        Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Please find your dreamified image attached.
        
        Best regards,
        DeepDream Service
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach image
        with open(image_path, 'rb') as f:
            img_data = f.read()
            image = MIMEImage(img_data)
            image.add_header('Content-Disposition', 
                           f'attachment; filename="{os.path.basename(image_path)}"')
            msg.attach(image)
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Email sending error: {e}")
        return False

def process_deepdream(input_path, output_path, recipient_email, layer_index, steps, step_size, guide_path=None, use_multiscale=False):
    """Background task to process DeepDream and send email"""
    try:
        print(f"Starting DeepDream processing...")
        print(f"Device: {device}")
        print(f"Layer index: {layer_index}, Steps: {steps}, Step size: {step_size}")
        
        # Load model
        model = get_submodel(layer_index)
        
        # Load guide image if provided
        guide_features = None
        if guide_path and os.path.exists(guide_path):
            print(f"Loading guide image: {guide_path}")
            guide_image = load_image(guide_path, requires_grad=False)
            with torch.no_grad():
                guide_features = model(guide_image).detach()
        else:
            print("No guide image, using standard DeepDream")
        
        if use_multiscale:
            # Multi-scale processing
            print("Using multi-scale processing...")
            dreamed = deep_dream_multiscale(
                input_path, model, guide_features, 
                scales=[224, 320], steps=steps//2, lr=step_size
            )
            save_image(dreamed, output_path)
        else:
            # Single scale processing
            print("Using single-scale processing...")
            image = load_image(input_path)
            dreamed = deep_dream(image, model, guide_features, steps, step_size)
            save_image(dreamed, output_path)
        
        # Send email with result
        send_email_with_image(recipient_email, output_path)
        
        # Clean up uploaded files
        if os.path.exists(input_path):
            os.remove(input_path)
        if guide_path and os.path.exists(guide_path):
            os.remove(guide_path)
            
        print("DeepDream processing completed successfully!")
        
    except Exception as e:
        print(f"DeepDream processing error: {e}")
        import traceback
        traceback.print_exc()

#clear files
@app.route('/clear-temp', methods=['POST'])
def clear_uploads_and_dreamified():
    """Delete all files in the uploads and dreamified folders (non-temp version)."""
    try:
        def clear_folder(folder):
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        clear_folder('uploads')
        clear_folder('dreamified')

        return jsonify({'message': 'uploads and dreamified folders cleared.'}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to clear folders: {str(e)}'}), 500

#conversion    
@app.route('/deepdream', methods=['POST'])
def deepdream_endpoint():
    """API endpoint to upload image for DeepDream processing"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters from form data
        recipient_email = request.form.get('email')
        layer_index = int(request.form.get('layer_index', 35))
        steps = int(request.form.get('steps', 500))
        step_size = float(request.form.get('step_size', 0.03))
        use_multiscale = request.form.get('multiscale', 'false').lower() == 'true'
        
        if not recipient_email:
            return jsonify({'error': 'Email address is required'}), 400
        
        # Validate parameters
        if not (0 <= layer_index <= 36):
            return jsonify({'error': 'Layer index must be between 0 and 36'}), 400
        
        if not (10 <= steps <= 2000):
            return jsonify({'error': 'Steps must be between 10 and 2000'}), 400
        
        if not (0.001 <= step_size <= 0.1):
            return jsonify({'error': 'Step size must be between 0.001 and 0.1'}), 400
        
        # Create directories
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('dreamified', exist_ok=True)
        
        # Save uploaded file
        timestamp = int(time.time())
        input_filename = f"input_{timestamp}_{file.filename}"
        input_path = os.path.join('uploads', input_filename)
        file.save(input_path)
        
        # Handle guide image if provided
        guide_path = None
        if 'guide_image' in request.files and request.files['guide_image'].filename != '':
            guide_file = request.files['guide_image']
            guide_filename = f"guide_{timestamp}_{guide_file.filename}"
            guide_path = os.path.join('uploads', guide_filename)
            guide_file.save(guide_path)
        
        # Generate output filename
        output_filename = f"dreamified_{timestamp}.jpg"
        output_path = os.path.join('dreamified', output_filename)
        
        # Start background processing
        thread = threading.Thread(
            target=process_deepdream,
            args=(input_path, output_path, recipient_email, layer_index, steps, step_size, guide_path, use_multiscale)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'DeepDream processing started. You will receive an email when complete.',
            'parameters': {
                'layer_index': layer_index,
                'steps': steps,
                'step_size': step_size,
                'multiscale': use_multiscale,
                'has_guide_image': guide_path is not None
            },
            'email': recipient_email,
            'estimated_time_minutes': steps / 100  # Rough estimate
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/', methods=['GET'])
def home():
    """Basic info endpoint"""
    return jsonify({
        'service': 'DeepDream Processing API',
        'endpoints': {
            'POST /deepdream': 'Upload image for DeepDream processing',
            'GET /health': 'Health check',
            'GET /': 'This endpoint'
        },
        'parameters': {
            'image': 'Required - Image file to process',
            'email': 'Required - Email to send result to',
            'layer_index': 'Optional - VGG19 layer (0-36, default: 35)',
            'steps': 'Optional - Processing steps (10-2000, default: 500)',
            'step_size': 'Optional - Learning rate (0.001-0.1, default: 0.03)',
            'guide_image': 'Optional - Guide image for guided DeepDream',
            'multiscale': 'Optional - Use multiscale processing (true/false, default: false)'
        },
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    })

if __name__ == '__main__':
    print(f"Starting DeepDream API on device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    app.run(debug=True, host='0.0.0.0', port=6969)