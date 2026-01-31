"""
Generate style preview images by applying trained models to the frame template
"""
import torch
from PIL import Image
from torchvision import transforms
import os
from style_transfer import TransformerNet

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_image_path = "content_images/frame-for-options.png"
output_dir = "style_previews"
os.makedirs(output_dir, exist_ok=True)

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Load and prepare input image
input_image = Image.open(input_image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)

# All available styles
styles = [
    'udnie', 'wave', 'rain_princess', 'monet',
    'candy', 'mosaic', 'picasso', 'scream', 'vangogh'
]

print(f"Generating style previews using {device}...")
print(f"Input image: {input_image_path}\n")

for style_name in styles:
    try:
        model_path = f"models/{style_name}.pth"
        
        # Load model
        model = TransformerNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Apply style transfer
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert back to image
        output = output.squeeze(0).clamp(0, 255).cpu()
        output = output.permute(1, 2, 0).numpy().astype('uint8')
        output_image = Image.fromarray(output)
        
        # Save
        output_path = os.path.join(output_dir, f"{style_name}.png")
        output_image.save(output_path)
        print(f"[OK] Generated {style_name}.png")
        
    except Exception as e:
        print(f"[FAIL] Failed to generate {style_name}: {e}")

# Also copy original for comparison
original = Image.open(input_image_path).convert('RGB')
original.thumbnail((256, 256))
original.save(os.path.join(output_dir, "original.png"))
print(f"[OK] Saved original.png")

print(f"\n[DONE] All preview images saved to {output_dir}/")
