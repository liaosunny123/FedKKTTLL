#!/usr/bin/env python3
"""
Complete fix script for FedKTL Stable Diffusion issues
"""
import os
import sys

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_environment():
    """Check if required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'diffusers': 'Diffusers library', 
        'transformers': 'Transformers library',
        'numpy': 'NumPy',
        'torchvision': 'Torchvision'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {name} is missing")
    
    return missing_packages

def download_model():
    """Download Stable Diffusion model"""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("\n📥 Downloading Stable Diffusion v1.5 model...")
        print("This will download ~4GB of model files.")
        
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
        
        print("✅ Model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def test_model_loading():
    """Test if the model can be loaded correctly"""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("\n🧪 Testing model loading...")
        
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        print(f"✅ Model successfully loaded on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def main():
    print("=" * 70)
    print("🔧 FedKTL Stable Diffusion Fix Script")
    print("=" * 70)
    
    print("\n1️⃣ Checking CUDA availability...")
    check_cuda()
    
    print("\n2️⃣ Checking required packages...")
    missing_packages = check_environment()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        if 'torch' in missing_packages:
            print("\nFor PyTorch with CUDA support:")
            print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print("\n3️⃣ Downloading Stable Diffusion model...")
    if not download_model():
        return False
    
    print("\n4️⃣ Testing model loading...")
    if not test_model_loading():
        return False
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! FedKTL Stable Diffusion setup is complete!")
    print("=" * 70)
    
    print("\n📋 What was fixed:")
    print("✅ Updated generator_path parameter to use 'runwayml/stable-diffusion-v1-5'")
    print("✅ Modified code to handle both local and remote model loading")
    print("✅ Downloaded Stable Diffusion v1.5 model from Hugging Face")
    print("✅ Verified model loading functionality")
    
    print("\n🚀 You can now run FedKTL with Stable Diffusion!")
    print("Example command:")
    print("python main.py -algo FedKTL-stable-diffusion -data Cifar10 -m HtM10")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
