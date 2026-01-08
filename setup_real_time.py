#!/usr/bin/env python3
"""
Setup script for real-time mental health assessment system
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "mediapipe>=0.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "streamlit>=1.28.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0"
    ]
    
    print("Installing requirements...")
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ All requirements installed successfully!")

def create_directory_structure():
    """Create necessary directory structure"""
    
    directories = [
        "src",
        "saved_models",
        "data",
        "reports",
        "assets",
        "logs"
    ]
    
    print("\nCreating directory structure...")
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}/")
        else:
            print(f"Exists: {directory}/")
    
    print("\n✅ Directory structure created!")

def check_webcam_access():
    """Check if webcam is accessible"""
    
    print("\nChecking webcam access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("✅ Webcam accessible and working")
                return True
            else:
                print("⚠ Webcam accessible but cannot read frames")
                return False
        else:
            print("❌ Webcam not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing webcam: {e}")
        return False

def main():
    """Main setup function"""
    
    print("=" * 60)
    print("Real-Time Mental Health Assessment System Setup")
    print("=" * 60)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Create directory structure
    create_directory_structure()
    
    # Step 3: Check webcam
    webcam_ok = check_webcam_access()
    
    # Step 4: Provide instructions
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Train or load a model:")
    print("   python train.py --data_path data.npz --model_type hybrid")
    print("\n2. Run the real-time application:")
    print("   streamlit run real_time_app.py")
    print("\n3. For webcam analysis:")
    print("   python webcam_demo.py")
    
    if not webcam_ok:
        print("\n⚠ Webcam issues detected. You may need to:")
        print("   - Grant camera permissions")
        print("   - Check if another application is using the camera")
        print("   - Try a different camera with --camera_id 1")
    
    print("\n📖 Documentation available in README.md")
    print("🆘 For help: python real_time_app.py --help")

if __name__ == "__main__":
    main()