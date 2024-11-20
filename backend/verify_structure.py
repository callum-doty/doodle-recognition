# verify_structure.py
from pathlib import Path
import sys


def create_init_file(path: Path):
    """Create an __init__.py file if it doesn't exist"""
    init_file = path / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Created {init_file}")


def verify_structure():
    """Verify and fix project structure"""
    # Create base directories
    directories = [
        "app",
        "app/model",
        "app/utils",
        "models",
        "data/raw",
        "data/processed"
    ]

    base_path = Path(".")
    for dir_name in directories:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"Created directory: {dir_path}")

        # Add __init__.py to Python packages
        if dir_name.startswith("app"):
            create_init_file(dir_path)

    # Verify key files
    key_files = {
        "app/main.py": lambda: print("Please create main.py from the previous code"),
        "app/model/cnn.py": lambda: print("Please create cnn.py from the previous code"),
        "app/utils/preprocessing.py": lambda: print("Please create preprocessing.py from the previous code")
    }

    missing_files = []
    for file_path, action in key_files.items():
        if not (base_path / file_path).exists():
            print(f"\nMissing file: {file_path}")
            action()
            missing_files.append(file_path)

    if missing_files:
        print("\nPlease create the missing files and try again.")
        return False

    return True


if __name__ == "__main__":
    if verify_structure():
        print("\nProject structure verified successfully!")
    else:
        print("\nProject structure verification failed!")
        sys.exit(1)
