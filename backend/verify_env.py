# verify_env.py
import sys
from pathlib import Path


def verify_environment():
    """Verify that all necessary components are in place"""

    # Check directory structure
    required_dirs = [
        "app",
        "app/model",
        "app/utils",
        "models",
        "data/raw",
        "data/processed"
    ]

    print("Checking directory structure...")
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating directory: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)

    # Check required files
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/model/__init__.py",
        "app/model/cnn.py",
        "app/utils/__init__.py",
        "app/utils/preprocessing.py"
    ]

    print("\nChecking required files...")
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    # Check model file
    model_path = Path("models/doodle_model.pth")
    if not model_path.exists():
        print("\nWarning: Model file not found!")
        print("You may need to train the model first using:")
        print("python quick_train.py")

    return True


if __name__ == "__main__":
    if verify_environment():
        print("\nEnvironment verification completed successfully!")
    else:
        print("\nEnvironment verification failed!")
        sys.exit(1)
