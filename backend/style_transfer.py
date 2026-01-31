import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np

class TransformerNet(nn.Module):
    """Fast Neural Style Transfer Network Architecture"""
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        
        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class StyleTransferModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.models = {}
        self.current_style = None
        self.current_model = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        
    def load_model(self, style_name):
        """Load a pre-trained style transfer model"""
        if style_name == 'none':
            self.current_style = 'none'
            self.current_model = None
            return
            
        if style_name in self.models:
            self.current_model = self.models[style_name]
            self.current_style = style_name
            return
        
        try:
            # Try to load pre-trained model
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", f"{style_name}.pth")
            model = TransformerNet()
            
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            except FileNotFoundError:
                print(f"Warning: Model file {model_path} not found. Using untrained model.")
                # Model will work but produce random output
            
            model.to(self.device)
            model.eval()
            
            self.models[style_name] = model
            self.current_model = model
            self.current_style = style_name
            
            print(f"Loaded model for style: {style_name}")
        except Exception as e:
            print(f"Error loading model {style_name}: {e}")
            self.current_model = None
            
    def process_frame(self, image_data, style_name='none', intensity=100):
        """Process a single frame with style transfer"""
        try:
            # Decode base64 image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                image = image_data
            
            # If no style or intensity is 0, return original
            if style_name == 'none' or intensity == 0:
                return self._image_to_base64(image)
            
            # Load model if needed
            if self.current_style != style_name:
                self.load_model(style_name)
            
            # If model failed to load, return original
            if self.current_model is None:
                return self._image_to_base64(image)
            
            # Resize for faster processing (optional)
            original_image = image.copy()  # Store original before any resizing
            original_size = image.size
            max_size = 640
            if max(original_size) > max_size:
                ratio = max_size / max(original_size)
                new_size = tuple(int(dim * ratio) for dim in original_size)
                image = image.resize(new_size, Image.LANCZOS)
            
            # Transform image to tensor
            content_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Apply style transfer
            with torch.no_grad():
                output_tensor = self.current_model(content_tensor)
            
            # Convert back to image
            output_tensor = output_tensor.cpu().squeeze(0)
            output_image = transforms.ToPILImage()(output_tensor.clamp(0, 255) / 255.0)
            
            # Resize back to original size if needed
            if output_image.size != original_size:
                output_image = output_image.resize(original_size, Image.LANCZOS)
            
            # Blend with original based on intensity
            if intensity < 100:
                alpha = intensity / 100.0
                # Ensure both images are same size and mode
                if output_image.size == original_image.size and output_image.mode == original_image.mode:
                    output_image = Image.blend(original_image, output_image, alpha)
                else:
                    print(f"Warning: Cannot blend - size or mode mismatch. Output: {output_image.size}/{output_image.mode}, Original: {original_image.size}/{original_image.mode}")
            
            return self._image_to_base64(output_image)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Return original image on error
            try:
                if isinstance(image_data, str):
                    return image_data
                else:
                    return self._image_to_base64(image_data)
            except:
                return None
    
    def _image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
