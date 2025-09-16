#!/usr/bin/env python3
"""
Script to download Stable Diffusion v1.5 model for FedKTL
"""
import os
import sys

def download_stable_diffusion():
    """Download Stable Diffusion v1.5 model"""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("Downloading Stable Diffusion v1.5 model...")
        print("This may take several minutes depending on your internet connection.")
        
        # Download and cache the model
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float32
        )
        
        print(f"✅ Successfully downloaded Stable Diffusion model: {model_id}")
        print("The model is now cached and ready to use.")
        
        # Test basic functionality
        print("Testing model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"✅ Model successfully loaded on {device}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: Missing required packages.")
        print("Please install required packages:")
        print("pip install diffusers transformers accelerate")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("This might be due to network issues or insufficient disk space.")
        return False

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['diffusers', 'transformers', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        if 'torch' in missing_packages:
            print("For PyTorch, visit: https://pytorch.org/get-started/locally/")
        return False
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("FedKTL Stable Diffusion Model Downloader")
    print("=" * 60)
    
    if not check_requirements():
        sys.exit(1)
    
    if download_stable_diffusion():
        print("\n🎉 Setup complete! You can now run FedKTL with Stable Diffusion.")
    else:
        print("\n💭 If you continue to have issues, you can also:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (>4GB)")
        print("3. Try running with a VPN if you're in a restricted region")
        sys.exit(1)
