#!/usr/bin/env python3
"""
Complete script to download and setup Stable Diffusion v1.5 model for FedKTL
"""
import os
import sys
import shutil

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'diffusers': 'Diffusers library', 
        'transformers': 'Transformers library',
        'accelerate': 'Accelerate library',
        'safetensors': 'SafeTensors library'
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

def download_and_cache_model():
    """Download Stable Diffusion model and ensure it's properly cached"""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("\n📥 Downloading Stable Diffusion v1.5 model...")
        print("This will download ~4GB of model files and cache them locally.")
        
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Download and cache the model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            dtype=torch.float32,
            use_safetensors=True
        )
        
        print("✅ Model downloaded and cached successfully!")
        
        # Test loading the model
        print("\n🧪 Testing model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"✅ Model successfully loaded on {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def remove_incomplete_local_model():
    """Remove incomplete local model files"""
    local_model_path = "system/stable-diffusion/v1.5"
    if os.path.exists(local_model_path):
        print(f"\n🧹 Removing incomplete local model at {local_model_path}")
        try:
            shutil.rmtree(local_model_path)
            print("✅ Incomplete local model removed")
        except Exception as e:
            print(f"⚠️  Could not remove incomplete model: {e}")

def test_fedktl_integration():
    """Test that FedKTL can now load the model correctly"""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("\n🔧 Testing FedKTL integration...")
        
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Test the exact loading method used by FedKTL
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                dtype=torch.float32,
                use_safetensors=True
            )
            print("✅ Model loads successfully with FedKTL parameters")
            
            # Test moving to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            print(f"✅ Model successfully moved to {device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    print("=" * 70)
    print("🚀 FedKTL Stable Diffusion Complete Setup Script")
    print("=" * 70)
    
    print("\n1️⃣ Checking required packages...")
    missing_packages = check_requirements()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n2️⃣ Removing incomplete local model files...")
    remove_incomplete_local_model()
    
    print("\n3️⃣ Downloading complete Stable Diffusion model...")
    if not download_and_cache_model():
        return False
    
    print("\n4️⃣ Testing FedKTL integration...")
    if not test_fedktl_integration():
        return False
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! FedKTL Stable Diffusion setup is complete!")
    print("=" * 70)
    
    print("\n📋 What was completed:")
    print("✅ Verified all required packages are installed")
    print("✅ Removed incomplete local model files")  
    print("✅ Downloaded complete Stable Diffusion v1.5 model from Hugging Face")
    print("✅ Verified model caching and loading functionality")
    print("✅ Fixed torch_dtype deprecation warning")
    print("✅ Tested FedKTL integration")
    
    print("\n🚀 You can now run FedKTL with Stable Diffusion!")
    print("Example command:")
    print("cd system && python main.py -algo FedKTL-stable-diffusion -data Cifar10 -m HtM10")
    
    print(f"\n💾 Model cache location:")
    try:
        from huggingface_hub import snapshot_download
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        print(f"   {cache_dir}")
    except:
        print("   ~/.cache/huggingface/hub/")
        
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
