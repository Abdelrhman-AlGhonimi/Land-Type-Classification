"""
Setup verification script for Land Type Classification System.
Run this script to verify that all dependencies and files are correctly set up.
"""

import sys
import os

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - NOT FOUND")
        return False

def check_module_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        if package_name:
            __import__(package_name)
        else:
            __import__(module_name)
        print(f"✓ {module_name}: OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: FAILED - {e}")
        return False

def main():
    print("=" * 60)
    print("Land Type Classification System - Setup Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("Python Version:")
    print(f"  {sys.version}")
    print()
    
    # Check required files
    print("Required Files:")
    all_ok &= check_file_exists("best_efficientnet_model.pth", "Model file")
    all_ok &= check_file_exists("app.py", "Main application")
    all_ok &= check_file_exists("requirements.txt", "Requirements file")
    all_ok &= check_file_exists("config/recommendations.json", "Recommendations config")
    print()
    
    # Check module structure
    print("Module Structure:")
    all_ok &= check_file_exists("model_inference/__init__.py", "Model inference module")
    all_ok &= check_file_exists("video_processing/__init__.py", "Video processing module")
    all_ok &= check_file_exists("image_retrieval/__init__.py", "Image retrieval module")
    all_ok &= check_file_exists("geolocation/__init__.py", "Geolocation module")
    all_ok &= check_file_exists("recommendations/__init__.py", "Recommendations module")
    all_ok &= check_file_exists("streamlit_app/__init__.py", "Streamlit app module")
    print()
    
    # Check dependencies
    print("Python Dependencies:")
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("efficientnet_pytorch", "EfficientNet-PyTorch"),
        ("streamlit", "Streamlit"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("requests", "Requests"),
        ("matplotlib", "Matplotlib"),
    ]
    
    for module, name in dependencies:
        all_ok &= check_module_import(name, module)
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! System is ready to use.")
        print("\nTo start the application, run:")
        print("  streamlit run app.py")
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()

