#!/usr/bin/env python3
"""
Script to download StyleGAN-3 model for FedKTL from hf-mirror
"""
import os
import sys
import requests
import warnings
import argparse
from urllib.parse import urlparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'pickle': 'Pickle (built-in)',
        'numpy': 'NumPy'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        if package == 'pickle':  # pickle is built-in
            continue
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {name} is missing")
    
    return missing_packages

def set_hf_mirror_environment():
    """Set up environment variables for hf-mirror"""
    print("🔧 Setting up hf-mirror environment...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("✅ HF_ENDPOINT set to https://hf-mirror.com")

def download_from_url(url, target_path):
    """Download file from URL with progress tracking"""
    print(f"📥 Downloading from: {url}")
    print(f"📁 Target path: {target_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n✅ Successfully downloaded: {target_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download: {e}")
        return False

def download_using_huggingface_cli(model_name, local_dir):
    """Download model using huggingface-cli"""
    try:
        import subprocess
        
        print(f"📥 Downloading {model_name} using huggingface-cli...")
        
        # Check if huggingface_hub is installed
        try:
            import huggingface_hub
            print("✅ huggingface_hub is available")
        except ImportError:
            print("❌ huggingface_hub not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"], check=True)
            print("✅ huggingface_hub installed")
        
        # Create the command
        cmd = [
            "huggingface-cli", "download", 
            "--resume-download", 
            model_name,
            "--local-dir", local_dir
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Model downloaded successfully!")
        print(f"📁 Model saved to: {local_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def download_stylegan3_models(model_choice="1"):
    """Download StyleGAN-3 model files"""
    models_dir = "system/stylegan/stylegan-3-models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Known StyleGAN-3 model URLs (these are from NVIDIA's official releases)
    model_urls = {
        "stylegan3-t-afhqv2-512x512.pkl": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl",
        "stylegan3-r-ffhqu-256x256.pkl": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl",
        "stylegan3-t-ffhqu-256x256.pkl": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl"
    }
    
    print(f"\n📦 Available StyleGAN-3 models:")
    for i, (name, url) in enumerate(model_urls.items(), 1):
        print(f"  {i}. {name}")
    
    print(f"\n🔄 Using choice: {model_choice}")
    
    try:
        if model_choice.lower() == 'all':
            selected_models = list(model_urls.items())
        else:
            choice_num = int(model_choice) - 1
            if 0 <= choice_num < len(model_urls):
                selected_models = [list(model_urls.items())[choice_num]]
            else:
                print("❌ Invalid choice, using default (first model)")
                selected_models = [list(model_urls.items())[0]]
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Invalid input, using default (first model)")
        selected_models = [list(model_urls.items())[0]]
    
    success_count = 0
    for model_name, model_url in selected_models:
        target_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(target_path):
            print(f"⚠️  {model_name} already exists. Skipping...")
            success_count += 1
            continue
            
        if download_from_url(model_url, target_path):
            success_count += 1
        else:
            print(f"❌ Failed to download {model_name}")
    
    return success_count == len(selected_models)

def try_download_from_huggingface():
    """Try to download from Hugging Face using different possible model names"""
    models_dir = "system/stylegan/stylegan-3-models"
    
    # Possible model names on Hugging Face
    possible_models = [
        "FedKTL-stylegan-3",
        "nvidia/stylegan3-ffhq",
        "nvidia/stylegan3-afhqv2",
        "stylegan3/ffhq",
        "stylegan3/afhqv2"
    ]
    
    print("\n🔍 Trying to find StyleGAN-3 models on Hugging Face...")
    
    for model_name in possible_models:
        print(f"\n🧪 Trying model: {model_name}")
        if download_using_huggingface_cli(model_name, models_dir):
            print(f"✅ Successfully downloaded from Hugging Face: {model_name}")
            return True
        else:
            print(f"❌ Model {model_name} not found or failed to download")
    
    return False

def test_model_loading():
    """Test loading the downloaded models"""
    models_dir = "system/stylegan/stylegan-3-models"
    
    print("\n🧪 Testing model loading...")
    
    try:
        import torch
        import pickle
        
        pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not pkl_files:
            print("❌ No .pkl files found to test")
            return False
        
        for pkl_file in pkl_files:
            model_path = os.path.join(models_dir, pkl_file)
            print(f"🔍 Testing {pkl_file}...")
            
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Check if the model has the expected structure
                if isinstance(model_data, dict) and 'G_ema' in model_data:
                    print(f"✅ {pkl_file} has correct structure with 'G_ema' key")
                    G = model_data['G_ema']
                    print(f"   Generator type: {type(G)}")
                    
                    # Try to get some basic info
                    if hasattr(G, 'z_dim'):
                        print(f"   Z dimension: {G.z_dim}")
                    if hasattr(G, 'w_dim'):
                        print(f"   W dimension: {G.w_dim}")
                    if hasattr(G, 'img_resolution'):
                        print(f"   Image resolution: {G.img_resolution}")
                    
                else:
                    print(f"⚠️  {pkl_file} structure: {type(model_data)}")
                    if isinstance(model_data, dict):
                        print(f"   Keys: {list(model_data.keys())}")
            
            except Exception as e:
                print(f"❌ Error loading {pkl_file}: {e}")
                continue
        
        return True
        
    except ImportError as e:
        print(f"❌ Required packages missing: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download StyleGAN-3 models for FedKTL")
    parser.add_argument("--method", choices=["1", "2", "3"], default="1", 
                       help="Download method: 1=NVIDIA official, 2=Hugging Face, 3=Both")
    parser.add_argument("--model", default="1", 
                       help="Model choice: 1, 2, 3, or 'all'")
    parser.add_argument("--skip-test", action="store_true", 
                       help="Skip model loading test")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎨 FedKTL StyleGAN-3 Model Download Script")
    print("=" * 70)
    
    print("\n1️⃣ Checking required packages...")
    missing_packages = check_requirements()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n2️⃣ Setting up hf-mirror environment...")
    set_hf_mirror_environment()
    
    print(f"\n3️⃣ Using download method: {args.method}")
    print("  1. Download from NVIDIA's official releases (recommended)")
    print("  2. Try downloading from Hugging Face")
    print("  3. Both methods")
    
    try:
        success = False
        
        if args.method in ['1', '3']:
            print("\n📥 Downloading from NVIDIA official releases...")
            if download_stylegan3_models(args.model):
                success = True
        
        if args.method in ['2', '3']:
            print("\n📥 Trying Hugging Face models...")
            if try_download_from_huggingface():
                success = True
        
        if not success:
            print("\n❌ All download methods failed")
            return False
        
        if not args.skip_test:
            print("\n4️⃣ Testing model loading...")
            test_model_loading()
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 StyleGAN-3 Model Download Complete!")
    print("=" * 70)
    
    print("\n📋 Next steps:")
    print("1. Use the downloaded model in your FedKTL training:")
    print("   cd system")
    print("   python main.py -algo FedKTL-stylegan-3 -data Cifar10 -m HtM10 \\")
    print("                   -GPath system/stylegan/stylegan-3-models/<model_file.pkl>")
    
    print(f"\n💾 Models saved to: system/stylegan/stylegan-3-models/")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        print("\n❌ Download failed. Please check the error messages above.")
        sys.exit(1)
