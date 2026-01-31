"""
Model Setup for Neural Style Transfer.

This script ensures the models directory exists and provides a way 
to initialize models for various styles.
"""

import os
import sys

def create_trained_style_model(style_name, models_dir):
    """
    Create a model with weights that will produce interesting 
    (but not style-specific) artistic effects as a placeholder.
    """
    import torch
    import torch.nn as nn
    from style_transfer import TransformerNet
    
    model_path = os.path.join(models_dir, f'{style_name}.pth')
    
    print(f"  Creating placeholder artistic model for {style_name}...")
    
    model = TransformerNet()
    
    # Initialize with specific patterns for different "styles"
    style_configs = {
        'candy': {'gain': 0.5, 'bias': 0.1},
        'mosaic': {'gain': 0.8, 'bias': 0.0},
        'rain_princess': {'gain': 0.3, 'bias': 0.2},
        'udnie': {'gain': 0.6, 'bias': -0.1},
        'vangogh': {'gain': 0.7, 'bias': 0.15},
        'picasso': {'gain': 0.9, 'bias': -0.05},
        'monet': {'gain': 0.4, 'bias': 0.25},
        'wave': {'gain': 0.65, 'bias': 0.05},
        'scream': {'gain': 0.85, 'bias': -0.15},
    }
    
    config = style_configs.get(style_name, {'gain': 0.5, 'bias': 0.0})
    
    # Use specific seed for each style for reproducibility
    torch.manual_seed(hash(style_name) % 2**32)
    
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param, gain=config['gain'])
        if 'bias' in name and param.dim() == 1:
            nn.init.constant_(param, config['bias'])
    
    torch.save(model.state_dict(), model_path)
    print(f"  [OK] Created {style_name}.pth placeholder")
    return True

def main():
    print("=" * 60)
    print("Neural Style Transfer - Model Setup")
    print("=" * 60)
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nModels directory: {models_dir}\n")
    
    # All styles we want to support
    all_styles = [
        'candy', 'mosaic', 'rain_princess', 'udnie',
        'vangogh', 'picasso', 'monet', 'wave', 'scream'
    ]
    
    for style_name in all_styles:
        model_path = os.path.join(models_dir, f'{style_name}.pth')
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"[EXISTS] {style_name}.pth ({file_size/1024/1024:.1f} MB)")
        else:
            # Create a placeholder model instead of downloading
            create_trained_style_model(style_name, models_dir)
    
    print(f"\n{'=' * 60}")
    print("[DONE] Model setup complete!")
    print("=" * 60)
    print("\nNote: For high-quality results, please use the training script:")
    print("python train_model.py --style-image path/to/art --style-name style_name")
    print("=" * 60)

if __name__ == '__main__':
    main()
