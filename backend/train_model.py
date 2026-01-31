"""
Train a fast neural style transfer model using only provided style images.
Usage:
    python train_model.py --style-image "style_images/Van Gogh" --style-name van_gogh_master
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import argparse
import sys
from tqdm import tqdm
import random

# Import the model architecture
from style_transfer import TransformerNet


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception:
            # Return a random image if one fails to load
            return self.__getitem__(random.randint(0, len(self.image_files)-1))


class VGG16Features(nn.Module):
    """Extract features from VGG16 for perceptual loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Split VGG into blocks for extracting features at different levels
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # relu3_3
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)])  # relu4_3
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


def gram_matrix(y):
    """Compute Gram matrix for style loss"""
    b, c, h, w = y.size()
    features = y.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def normalize_batch(batch):
    """Normalize batch for VGG (ImageNet normalization)"""
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) * 255
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) * 255
    return (batch - mean) / std


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Preprocessing transforms
    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    # Initialize models
    transformer = TransformerNet().to(device)
    vgg = VGG16Features().to(device).eval()
    
    # Load content dataset
    content_dir = args.content_dir if args.content_dir and os.path.exists(args.content_dir) else args.style_image
    print(f"Loading content images from: {content_dir}")
    print(f"Loading style images from: {args.style_image}")
    
    dataset = ImageDataset(content_dir, img_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    style_image_paths = [os.path.join(args.style_image, f) for f in os.listdir(args.style_image) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def get_style_features(path):
        try:
            img = Image.open(path).convert('RGB')
            style_img = img_transform(img).unsqueeze(0).to(device)
            style_img_normalized = normalize_batch(style_img)
            features = vgg(style_img_normalized)
            return [gram_matrix(f).detach() for f in features]
        except Exception:
            return None

    # Optimizer
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Total images: {len(dataset)}")
    
    for epoch in range(args.epochs):
        transformer.train()
        
        # Use provided iterations or default to max of dataloader/500
        num_iterations = args.iterations if args.iterations else max(len(dataloader), 500)
        progress_bar = tqdm(range(num_iterations), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        data_iter = iter(dataloader)
        
        for i in progress_bar:
            try:
                content = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                content = next(data_iter)
                
            content = content.to(device)
            n_batch = content.size(0)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = transformer(content)
            
            # Normalize for VGG
            output_normalized = normalize_batch(output)
            content_normalized = normalize_batch(content)
            
            # Get features
            output_features = vgg(output_normalized)
            content_features = vgg(content_normalized)
            
            # 1. Content loss
            c_loss = args.content_weight * nn.functional.mse_loss(output_features[2], content_features[2])
            
            # 2. Style loss
            s_loss = 0.
            # Pick a random style image for this batch
            random_style_path = random.choice(style_image_paths)
            style_grams = get_style_features(random_style_path)
            
            if style_grams:
                for of, sg in zip(output_features, style_grams):
                    output_gram = gram_matrix(of)
                    s_loss += nn.functional.mse_loss(output_gram, sg.expand(n_batch, -1, -1))
                s_loss *= args.style_weight
            
            # 3. Total variation loss
            tv_loss = args.tv_weight * (
                torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) +
                torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]))
            )
            
            total_loss = c_loss + s_loss + tv_loss
            
            # Check for NaN
            if torch.isnan(total_loss):
                print(f"Warning: NaN loss detected at iteration {i}. Skipping.")
                continue
                
            total_loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.2e}",
                'style': f"{s_loss.item():.2e}",
                'content': f"{c_loss.item():.2e}"
            })

    # Save final model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'{args.style_name}.pth')
    
    torch.save(transformer.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Fast Neural Style Transfer Model")
    parser.add_argument("--style-image", required=True, help="Path to style image directory")
    parser.add_argument("--style-name", required=True, help="Name for the style")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--content-weight", type=float, default=1e5, help="Content loss weight")
    parser.add_argument("--style-weight", type=float, default=1e10, help="Style loss weight")
    parser.add_argument("--tv-weight", type=float, default=1e-7, help="Total variation weight")
    parser.add_argument("--content-dir", help="Path to content image directory (optional, defaults to style images)")
    parser.add_argument("--image-size", type=int, default=256, help="Training image size")
    parser.add_argument("--iterations", type=int, help="Number of iterations per epoch")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
