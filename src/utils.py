import joblib
import os


def save_model(model, model_name, directory="models"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{model_name}.joblib")

    try:
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")


def load_model(model_name, directory="models"):
    filepath = os.path.join(directory, f"{model_name}.joblib")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
