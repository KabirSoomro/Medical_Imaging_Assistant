"""
Test script for Medical Imaging Assistant.
Run this to verify all modules are working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")
    try:
        from utils.model_loader import load_cnn_model, load_vgg16_model, get_model_info
        print("✅ model_loader imported successfully")
    except ImportError as e:
        print(f"❌ model_loader import failed: {e}")
        return False
    
    try:
        from utils.preprocess import load_image, preprocess_image
        print("✅ preprocess imported successfully")
    except ImportError as e:
        print(f"❌ preprocess import failed: {e}")
        return False
    
    try:
        from utils.predict import predict, DISEASE_CLASSES
        print("✅ predict imported successfully")
    except ImportError as e:
        print(f"❌ predict import failed: {e}")
        return False
    
    try:
        from utils.visualize import create_confidence_chart, create_confidence_gauge
        print("✅ visualize imported successfully")
    except ImportError as e:
        print(f"❌ visualize import failed: {e}")
        return False
    
    try:
        from utils.explainability import generate_explanation, get_key_factors
        print("✅ explainability imported successfully")
    except ImportError as e:
        print(f"❌ explainability import failed: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

def test_dependencies():
    """Test all dependencies are installed."""
    print("\nTesting dependencies...")
    dependencies = [
        'streamlit',
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'plotly',
        'cv2',
        'PIL',
        'sklearn'
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} installed")
        except ImportError:
            print(f"❌ {dep} not installed")
            all_ok = False
    
    return all_ok

def test_folder_structure():
    """Test folder structure is correct."""
    print("\nTesting folder structure...")
    required_folders = [
        'models',
        'utils',
        'data',
        'data/sample_images',
        'screenshots',
        'notebooks'
    ]
    
    all_ok = True
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✅ {folder}/ exists")
        else:
            print(f"❌ {folder}/ missing")
            all_ok = False
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'utils/__init__.py',
        'utils/model_loader.py',
        'utils/preprocess.py',
        'utils/predict.py',
        'utils/visualize.py',
        'utils/explainability.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests."""
    print("=" * 60)
    print("Medical Imaging Assistant - Verification Script")
    print("=" * 60)
    
    # Run tests
    imports_ok = test_imports()
    deps_ok = test_dependencies()
    folder_ok = test_folder_structure()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ Imports: {'OK' if imports_ok else 'FAILED'}")
    print(f"✅ Dependencies: {'OK' if deps_ok else 'FAILED'}")
    print(f"✅ Folder Structure: {'OK' if folder_ok else 'FAILED'}")
    
    if imports_ok and deps_ok and folder_ok:
        print("\n🎉 All checks passed! You can now run the app:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above.")
        print("\nTo install missing dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
