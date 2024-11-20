# verify_model.py
import torch
from pathlib import Path
from app.model.cnn import DoodleNet


def verify_model():
    model_path = Path('models/doodle_model.pth')

    # Check if model file exists
    if not model_path.exists():
        print(f"Model file not found at {model_path}")
        return False

    try:
        # Load model
        model = DoodleNet(num_classes=15)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()

        # Test with dummy input
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            output = model(dummy_input)

        print("Model verified successfully!")
        print(f"Output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"Error verifying model: {e}")
        return False


if __name__ == "__main__":
    verify_model()
