#!/usr/bin/env python3
"""Install core dependencies for the LegalGPT FastAPI application."""

import subprocess
import sys

def install_package(package):
    """Install a single package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install core packages needed for the application."""
    core_packages = [
        "torch==2.6.0",
        "transformers==4.51.3", 
        "sentence-transformers==4.1.0",
        "chromadb==1.0.6",
        "fastapi==0.115.9",
        "uvicorn==0.34.2",
        "pydantic==2.11.3",
        "python-multipart==0.0.20",
        "aiofiles"
    ]
    
    print("Installing core dependencies...")
    for package in core_packages:
        if not install_package(package):
            print(f"Failed to install {package}, continuing...")
    
    print("Core installation complete!")

if __name__ == "__main__":
    main()